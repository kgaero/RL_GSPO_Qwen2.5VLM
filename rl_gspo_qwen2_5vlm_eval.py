# -*- coding: utf-8 -*-
"""Notebook-friendly entrypoint for evaluation-only checkpoint reevaluation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from staged_rl.checkpointing import write_checkpoint_artifacts
from staged_rl.config import (
    DatasetFilterSpec,
    StageSpec,
    apply_hardware_profile,
    build_default_hardware_profiles,
    build_default_run_config,
)
from staged_rl.data import (
    analyze_dataset_records,
    build_eval_datasets,
    build_stage_dataset,
    dataset_to_records,
    load_mathvista_split,
    save_dataset_analysis,
)
from staged_rl.diagnostics import save_json, write_fatal_error
from staged_rl.evaluation import evaluate_checkpoint
from staged_rl.rewarding import RewardRuntimeContext, build_reward_functions
from staged_rl.trainer_runtime import (
    build_initial_reward_weights,
    create_model_and_tokenizer,
    log_cuda_environment,
)


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure process-wide logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI flags while staying notebook-friendly."""

    parser = argparse.ArgumentParser(description="Reevaluate saved LoRA checkpoints on MathVista splits.")
    parser.add_argument("--phase", default="phase_c", help="Default phase to assume when a target does not specify one.")
    parser.add_argument("--checkpoint", default=None, help="Single checkpoint or adapter directory to evaluate.")
    parser.add_argument("--label", default=None, help="Single target label used under the output root.")
    parser.add_argument(
        "--target-spec-json",
        default=None,
        help="Path to a JSON file containing a list of targets or an object with a 'targets' list.",
    )
    parser.add_argument("--output-root", default="outputs_full_testmini_reeval", help="Output root for evaluation artifacts.")
    parser.add_argument("--eval-split", default=None, help="Override the evaluation split name.")
    parser.add_argument(
        "--hardware-profile",
        default="default",
        choices=sorted(build_default_hardware_profiles().keys()),
        help="Runtime profile for smaller GPUs, for example kaggle_t4.",
    )
    parser.add_argument(
        "--max-eval-examples-per-subset",
        type=int,
        default=None,
        help="Limit evaluation examples per subset. Omit for the full selected subset.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=None,
        help="Single-target override for the generation/evaluation completion budget.",
    )
    parser.add_argument(
        "--reward-weights-json",
        default=None,
        help="Single-target override for a reward_weights.json path.",
    )
    parser.add_argument(
        "--save-full-completion-text",
        action="store_true",
        help="Persist full completion text in per-sample records.",
    )
    parser.add_argument(
        "--disable-stage",
        action="append",
        default=[],
        help="Disable a named stage before building evaluation subsets.",
    )
    parser.add_argument(
        "--enable-stage",
        action="append",
        default=[],
        help="Enable a named stage before building evaluation subsets.",
    )
    subset_mode = parser.add_mutually_exclusive_group()
    subset_mode.add_argument(
        "--overall-only",
        action="store_true",
        help="Evaluate only the numeric overall subset and skip stage-specific subsets.",
    )
    subset_mode.add_argument(
        "--full-split",
        action="store_true",
        help="Evaluate the full raw split, including both numeric free-form and multi-choice rows.",
    )
    return parser.parse_args()


def apply_cli_overrides(run_config, args: argparse.Namespace):
    """Update the default run config from CLI overrides."""

    run_config.phase_name = args.phase
    run_config = apply_hardware_profile(run_config, args.hardware_profile)
    run_config.output_root = args.output_root
    if args.eval_split:
        run_config.eval_split = args.eval_split
    if args.max_eval_examples_per_subset is not None:
        run_config.eval.max_eval_examples_per_subset = args.max_eval_examples_per_subset
    run_config.eval.save_full_completion_text = args.save_full_completion_text

    for stage_name in args.disable_stage:
        if stage_name in run_config.stages:
            run_config.stages[stage_name].enabled = False
    for stage_name in args.enable_stage:
        if stage_name in run_config.stages:
            run_config.stages[stage_name].enabled = True
    return run_config


def _resolve_optional_path(value: Any, base_dir: Path) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def load_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load one or more evaluation targets from CLI args."""

    if args.target_spec_json:
        spec_path = Path(args.target_spec_json).resolve()
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        targets = payload.get("targets", payload) if isinstance(payload, dict) else payload
        if not isinstance(targets, list):
            raise ValueError("Target spec JSON must be a list or an object with a 'targets' list.")
        base_dir = spec_path.parent
    else:
        if not args.checkpoint:
            raise ValueError("Either --checkpoint or --target-spec-json is required.")
        targets = [
            {
                "label": args.label or Path(args.checkpoint).name,
                "checkpoint": args.checkpoint,
                "phase": args.phase,
                "max_completion_length": args.max_completion_length,
                "reward_weights_json": args.reward_weights_json,
            }
        ]
        base_dir = Path.cwd()

    normalized_targets = []
    for target in targets:
        checkpoint = _resolve_optional_path(target.get("checkpoint"), base_dir)
        if checkpoint is None:
            raise ValueError(f"Target is missing 'checkpoint': {target}")
        normalized_targets.append(
            {
                **dict(target),
                "label": str(target.get("label") or Path(checkpoint).name),
                "checkpoint": checkpoint,
                "phase": str(target.get("phase") or args.phase),
                "reward_weights_json": _resolve_optional_path(target.get("reward_weights_json"), base_dir),
            }
        )
    return normalized_targets


def build_eval_datasets_for_mode(base_eval_dataset, run_config, tokenizer, *, overall_only: bool, full_split: bool) -> dict[str, Any]:
    """Build the evaluation subsets requested for this reevaluation run."""

    if full_split:
        full_stage = StageSpec(
            name="eval_full_split",
            description="Full mixed evaluation split without stage filtering.",
            answer_mode="mixed",
            filter_spec=DatasetFilterSpec(),
        )
        return {
            full_stage.name: build_stage_dataset(
                base_eval_dataset,
                full_stage,
                tokenizer,
                image_size=run_config.model.image_size,
            )
        }

    eval_datasets = build_eval_datasets(base_eval_dataset, run_config, tokenizer)
    if overall_only:
        return {"eval_overall_numeric": eval_datasets["eval_overall_numeric"]}
    return eval_datasets


def resolve_max_completion_length(target: dict[str, Any], run_config, phase_config) -> int:
    """Resolve the completion budget used during reevaluation."""

    explicit = target.get("max_completion_length")
    if explicit is not None:
        return int(explicit)
    phase_override = phase_config.trainer_overrides.get("max_completion_length")
    if phase_override is not None:
        return int(phase_override)
    return int(run_config.trainer_defaults.max_completion_length)


def resolve_reward_weights(target: dict[str, Any], phase_config) -> dict[str, float]:
    """Load target-specific reward weights when present, otherwise use phase defaults."""

    reward_weights = build_initial_reward_weights(phase_config)
    candidate_paths = []
    if target.get("reward_weights_json"):
        candidate_paths.append(Path(target["reward_weights_json"]))
    candidate_paths.append(Path(target["checkpoint"]) / "reward_weights.json")

    for path in candidate_paths:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            reward_weights.update({key: float(value) for key, value in payload.items()})
            return reward_weights
    return reward_weights


def load_checkpoint_metadata(checkpoint_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load optional checkpoint_info and controller_state payloads."""

    checkpoint_dir = Path(checkpoint_path)
    info_path = checkpoint_dir / "checkpoint_info.json"
    controller_path = checkpoint_dir / "controller_state.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    controller_state = json.loads(controller_path.read_text(encoding="utf-8")) if controller_path.exists() else {}
    return info, controller_state


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a compact CSV summary for the reevaluation run."""

    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Entrypoint used by both CLI runs and notebook execution."""

    configure_logging()
    args = parse_args()
    run_config = apply_cli_overrides(build_default_run_config(phase_name=args.phase), args)
    output_root = Path(run_config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    fatal_error_path = output_root / "fatal_error.txt"
    if fatal_error_path.exists():
        fatal_error_path.unlink()

    targets = load_targets(args)
    save_json(
        {
            "default_phase": args.phase,
            "output_root": run_config.output_root,
            "eval_split": run_config.eval_split,
            "hardware_profile": run_config.hardware_profile_name,
            "max_eval_examples_per_subset": run_config.eval.max_eval_examples_per_subset,
            "save_full_completion_text": run_config.eval.save_full_completion_text,
            "overall_only": args.overall_only,
            "full_split": args.full_split,
            "disabled_stages": args.disable_stage,
            "enabled_stages": args.enable_stage,
            "targets": targets,
        },
        output_root / "run_request.json",
    )

    try:
        log_cuda_environment()
        model, tokenizer = create_model_and_tokenizer(run_config)
        if hasattr(model, "for_inference"):
            model.for_inference()

        eval_base = load_mathvista_split(run_config, run_config.eval_split)
        save_dataset_analysis(
            analyze_dataset_records(dataset_to_records(eval_base), run_config.stages),
            output_root / "dataset_analysis_eval.json",
        )
        eval_datasets = build_eval_datasets_for_mode(
            eval_base,
            run_config,
            tokenizer,
            overall_only=args.overall_only,
            full_split=args.full_split,
        )
        save_json(
            {name: len(dataset) for name, dataset in eval_datasets.items()},
            output_root / "eval_subset_sizes.json",
        )

        summary_rows: list[dict[str, Any]] = []
        for target in targets:
            phase_name = target["phase"]
            if phase_name not in run_config.phases:
                raise ValueError(f"Unknown phase '{phase_name}' for target {target['label']}")
            phase_config = run_config.phases[phase_name]
            label = target["label"]
            checkpoint_path = target["checkpoint"]
            max_completion_length = resolve_max_completion_length(target, run_config, phase_config)
            reward_weights = resolve_reward_weights(target, phase_config)
            checkpoint_info, controller_state = load_checkpoint_metadata(checkpoint_path)

            reward_runtime = RewardRuntimeContext(
                tokenizer=tokenizer,
                max_completion_length=max_completion_length,
                phase_config=phase_config,
            )
            reward_funcs = build_reward_functions(reward_runtime)
            eval_results = evaluate_checkpoint(
                model=model,
                eval_datasets=eval_datasets,
                lora_path=checkpoint_path,
                runtime=reward_runtime,
                reward_funcs=reward_funcs,
                reward_weights=reward_weights,
                eval_config=run_config.eval,
            )

            target_output_dir = output_root / label
            target_output_dir.mkdir(parents=True, exist_ok=True)
            target_checkpoint_info = {
                "checkpoint_path": checkpoint_path,
                "global_step": checkpoint_info.get("global_step"),
                "phase_name": phase_name,
                "selector_phase_name": phase_name,
                "source_phase_name": checkpoint_info.get("phase_name"),
                "label": label,
                "eval_split": run_config.eval_split,
                "eval_mode": "full_split" if args.full_split else ("overall_only" if args.overall_only else "default"),
                "max_completion_length": max_completion_length,
            }
            registry_entry = write_checkpoint_artifacts(
                checkpoint_dir=target_output_dir,
                eval_results=eval_results,
                reward_weights=reward_weights,
                controller_state=controller_state,
                checkpoint_info=target_checkpoint_info,
                score_config=run_config.checkpoint_scores,
            )
            save_json(target, target_output_dir / "target_spec.json")

            metrics = registry_entry["metrics"]
            original_metrics = checkpoint_info.get("metrics", {})
            summary_rows.append(
                {
                    "label": label,
                    "phase": phase_name,
                    "checkpoint_path": checkpoint_path,
                    "output_dir": str(target_output_dir),
                    "eval_split": run_config.eval_split,
                    "eval_mode": target_checkpoint_info["eval_mode"],
                    "max_completion_length": max_completion_length,
                    "original_normalized_exact_match": original_metrics.get("normalized_exact_match"),
                    "reevaluated_normalized_exact_match": metrics.get("normalized_exact_match"),
                    "original_tolerance_accuracy": original_metrics.get("tolerance_accuracy"),
                    "reevaluated_tolerance_accuracy": metrics.get("tolerance_accuracy"),
                    "reevaluated_parseable_answer_rate": metrics.get("parseable_answer_rate"),
                    "reevaluated_malformed_answer_rate": metrics.get("malformed_answer_rate"),
                    "reevaluated_truncation_rate": metrics.get("truncation_rate"),
                    "reevaluated_average_completion_tokens": metrics.get("average_completion_tokens"),
                    "structure_score": metrics.get("structure_score"),
                    "correctness_score": metrics.get("correctness_score"),
                    "composite_score": metrics.get("composite_score"),
                }
            )

        save_json({"targets": summary_rows}, output_root / "reevaluation_summary.json")
        write_summary_csv(summary_rows, output_root / "reevaluation_summary.csv")
        print("=" * 80)
        print("REEVALUATION COMPLETE")
        print("=" * 80)
        print(json.dumps(summary_rows, indent=2, ensure_ascii=False, default=str))
    except Exception as exc:
        write_fatal_error(
            fatal_error_path,
            exc,
            {
                "output_root": run_config.output_root,
                "eval_split": run_config.eval_split,
                "hardware_profile": run_config.hardware_profile_name,
                "max_eval_examples_per_subset": run_config.eval.max_eval_examples_per_subset,
                "overall_only": args.overall_only,
                "full_split": args.full_split,
                "targets": targets if "targets" in locals() else None,
            },
        )
        LOGGER.exception("Fatal reevaluation error written to %s", fatal_error_path)
        raise


if __name__ == "__main__":
    main()
