#!/usr/bin/env python3
"""Generate report-ready CSV tables and plots from saved RL artifacts."""

from __future__ import annotations

import csv
import json
import math
import sys
import textwrap
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from staged_rl.config import apply_hardware_profile, build_default_run_config

RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"


RUN_SOURCES = [
    {
        "run_family": "smoke_testmini",
        "run_family_label": "Smoke Testmini",
        "notebook_slug": "mcgmcg1/rl-gspo-qwen2-5vlm-staged-train",
        "output_root_label": "outputs_staged",
        "root": Path("/tmp/rl-gspo-report-smoke/RL_GSPO_Qwen2.5VLM/outputs_staged"),
        "phases": ["phase_a", "phase_b", "phase_c", "phase_d"],
        "defaults": {
            "train_split": "testmini",
            "eval_split": "testmini",
            "hardware_profile": "kaggle_t4",
            "output_root": "outputs_staged",
        },
    },
    {
        "run_family": "large_split_continue",
        "run_family_label": "Large Split Continue",
        "notebook_slug": "mcgmcg1/rl-gspo-qwen2-5vlm-large-split-continue",
        "output_root_label": "outputs_staged_large_continue",
        "root": Path("/tmp/rl-gspo-qwen2-5vlm-large-full/RL_GSPO_Qwen2.5VLM/outputs_staged_large_continue"),
        "phases": ["phase_c", "phase_d"],
        "defaults": {
            "train_split": "test",
            "eval_split": "testmini",
            "hardware_profile": "kaggle_t4",
            "output_root": "outputs_staged_large_continue",
        },
    },
    {
        "run_family": "phase_d_dedicated",
        "run_family_label": "Dedicated Phase D",
        "notebook_slug": "mcgmcg1/rl-gspo-qwen2-5vlm-phase-d-large-continue",
        "output_root_label": "outputs_staged_phase_d_from_large_phase_c",
        "root": Path(
            "/tmp/rl-gspo-qwen2-5vlm-phase-d-final/RL_GSPO_Qwen2.5VLM/outputs_staged_phase_d_from_large_phase_c"
        ),
        "phases": ["phase_d"],
        "defaults": {
            "train_split": "test",
            "eval_split": "testmini",
            "hardware_profile": "kaggle_t4",
            "output_root": "outputs_staged_phase_d_from_large_phase_c",
        },
    },
]

PHASE_SEQUENCE = {
    ("smoke_testmini", "phase_a"): 1,
    ("smoke_testmini", "phase_b"): 2,
    ("smoke_testmini", "phase_c"): 3,
    ("smoke_testmini", "phase_d"): 4,
    ("large_split_continue", "phase_c"): 5,
    ("large_split_continue", "phase_d"): 6,
    ("phase_d_dedicated", "phase_d"): 7,
}

PHASE_LABELS = {
    ("smoke_testmini", "phase_a"): "Smoke Phase A",
    ("smoke_testmini", "phase_b"): "Smoke Phase B",
    ("smoke_testmini", "phase_c"): "Smoke Phase C",
    ("smoke_testmini", "phase_d"): "Smoke Phase D",
    ("large_split_continue", "phase_c"): "Large Phase C",
    ("large_split_continue", "phase_d"): "Large Phase D",
    ("phase_d_dedicated", "phase_d"): "Dedicated Phase D",
}

PHASE_STRATEGY_TEXT = {
    "phase_a": "Structure stabilization on Stage 1 easy numeric subset.",
    "phase_b": "Correctness strengthening on Stage 1+2 mix after structure stabilization.",
    "phase_c": "Precision and harder numeric reasoning on Stage 2+3 with larger split continuation.",
    "phase_d": "Hard numeric specialization on Stage 3-heavy subset.",
}

BASELINE_METRICS_PATH = ROOT / "grpo_eval_outputs" / "eval_metrics.json"
BASELINE_TRAIN_SUMMARY_PATH = ROOT / "grpo_eval_outputs" / "train_log_summary.json"

PRIMARY_FINAL_CHECKPOINT = "outputs_staged_large_continue/phase_c/checkpoint-120"

MASTER_TABLE_COLUMNS = [
    "Milestone ID",
    "Run Family",
    "Run Family Label",
    "Notebook / Run Slug",
    "Output Root",
    "Phase",
    "Phase Label",
    "Checkpoint",
    "Global Step",
    "Timeline Order",
    "X Label",
    "Artifact Status",
    "Metrics Source Kind",
    "Alias Role",
    "Alias Roles",
    "Keep / Discard",
    "Decision Reason",
    "Key Interpretation",
    "Train Split",
    "Eval Split",
    "Hardware Profile",
    "Seed / Resume Source",
    "Warm-Start Checkpoint",
    "Training Strategy Introduced",
    "Stage Mix / Curriculum",
    "Phase Description",
    "Phase Default Resume Selector",
    "Base Model",
    "4-bit Enabled",
    "Fast Inference Enabled",
    "Fast Inference Mode",
    "Compilation Mode",
    "max_seq_length",
    "image_size",
    "LoRA Rank",
    "Max LoRA Rank",
    "gradient_accumulation_steps",
    "gpu_memory_utilization",
    "num_generations",
    "max_prompt_length",
    "max_completion_length",
    "max_eval_examples_per_subset",
    "Configured Initial Correctness Weight",
    "Configured Initial Formatting Weight",
    "Configured Initial Parseability Weight",
    "Configured Initial Finished Weight",
    "Configured Initial Tolerance Weight",
    "Configured Initial Brevity Weight",
    "Correctness Weight",
    "Formatting Weight",
    "Parseability Weight",
    "Finished Weight",
    "Tolerance Weight",
    "Brevity Weight",
    "Correctness Reward Mean",
    "Format Reward Mean",
    "Parseable Reward Mean",
    "Finished Reward Mean",
    "Tolerance Reward Mean",
    "Exact Match",
    "Tolerance Accuracy",
    "Best-of-k Accuracy",
    "Best-of-k Tolerance Accuracy",
    "Sample-Level Exact Match",
    "Sample-Level Tolerance Accuracy",
    "Parseable Rate",
    "Reasoning Tag Compliance",
    "Solution Tag Compliance",
    "Well-Formed Rate",
    "Malformed Rate",
    "Completion Success Rate",
    "Truncation Rate",
    "Average Completion Tokens",
    "Repetition Rate",
    "Sample Diversity",
    "Average Total Reward",
    "Structure Score",
    "Correctness Score",
    "Composite Score",
    "KL Mean",
    "KL P95",
    "Checkpoint Rank Within Phase",
    "Is Best Composite",
    "Is Best Correctness",
    "Is Best Structure",
    "Is Latest",
    "Controller History Length",
    "Delta Exact vs Previous Milestone",
    "Delta Composite vs Previous Milestone",
    "Checkpoint Path",
    "Checkpoint Path Abs",
    "Evidence Root",
    "Metrics Source Path",
    "Checkpoint Info Path",
    "Reward Weights Path",
    "Controller State Path",
    "Run Config Path",
    "Run Request Path",
    "Diagnostics Path",
    "Train Log Summary Path",
    "Summary Path",
    "Notes",
]

CHECKPOINT_ONLY_COLUMNS = MASTER_TABLE_COLUMNS + [
    "Checkpoint Rank Within Phase",
    "Is Best Composite",
    "Is Best Correctness",
    "Is Best Structure",
    "Is Latest",
]


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def json_load(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text())


def maybe_path(path: Path) -> str:
    return str(path) if path.exists() else ""


def rel(path: Path | str | None) -> str:
    if not path:
        return ""
    path_obj = Path(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def parse_checkpoint_step(name: str) -> int | None:
    if not name.startswith("checkpoint-"):
        return None
    try:
        return int(name.split("-", 1)[1])
    except ValueError:
        return None


def phase_display(run_family: str, phase_name: str) -> str:
    return PHASE_LABELS.get((run_family, phase_name), phase_name)


def normalize_metric_name(metrics: dict[str, Any], key: str, *alternatives: str) -> Any:
    for name in (key,) + alternatives:
        if name in metrics:
            return metrics[name]
    return None


def compute_scores(metrics: dict[str, Any]) -> dict[str, float]:
    cfg = build_default_run_config().checkpoint_scores

    def score(mapping: dict[str, float]) -> float:
        total = 0.0
        for metric_name, weight in mapping.items():
            total += weight * float(metrics.get(metric_name, 0.0))
        return total

    return {
        "structure_score": score(dict(cfg.structure_weights)),
        "correctness_score": score(dict(cfg.correctness_weights)),
        "composite_score": score(dict(cfg.composite_weights)),
    }


def best_of_k_value(metrics: dict[str, Any]) -> Any:
    return normalize_metric_name(metrics, "best_of_k_accuracy", "best_of_4_accuracy")


def best_of_k_tol_value(metrics: dict[str, Any]) -> Any:
    return normalize_metric_name(metrics, "best_of_k_tolerance_accuracy", "best_of_4_tolerance_accuracy")


def format_reward_mean(metrics: dict[str, Any]) -> Any:
    return normalize_metric_name(metrics, "reward_component/format_reward_mean", "reward_component/formatting_reward_mean")


def total_reward_mean(metrics: dict[str, Any]) -> Any:
    return normalize_metric_name(metrics, "average_total_reward", "reward_component/total_reward_mean")


def select_primary_alias(alias_roles: Iterable[str]) -> str:
    priority = ["best_composite", "best_correctness", "best_structure", "latest"]
    alias_set = list(alias_roles)
    for role in priority:
        if role in alias_set:
            return role
    return "none"


def keep_discard(row: dict[str, Any]) -> tuple[str, str]:
    if row["Artifact Status"] == "baseline_snapshot":
        return "reference", "Baseline reference snapshot."
    if row["Artifact Status"] == "planned_milestone_missing":
        return "unavailable", "Planned milestone not present in current exported artifacts."
    if row["Checkpoint Path"] == PRIMARY_FINAL_CHECKPOINT:
        return "keep_primary", "Recommended final checkpoint: best larger-split Phase C composite result."
    if row["Is Best Composite"]:
        return "keep", "Phase-best composite checkpoint."
    if row["Is Best Correctness"] or row["Is Best Structure"]:
        return "keep_secondary", "Secondary alias checkpoint kept for analysis."
    if row["Is Latest"]:
        return "discard", "Latest checkpoint is not the recommended phase checkpoint."
    return "discard", "Non-alias checkpoint retained for audit trail only."


def phase_strategy_text(phase_name: str, run_family: str) -> str:
    base = PHASE_STRATEGY_TEXT.get(phase_name, phase_name)
    if run_family == "phase_d_dedicated":
        return base + " Dedicated notebook preserved earlier large-split outputs."
    return base


def flatten_stage_mix(stage_mix: dict[str, Any]) -> str:
    if not stage_mix:
        return ""
    parts = [f"{name}={weight:.2f}" for name, weight in sorted(stage_mix.items())]
    return "; ".join(parts)


def build_fallback_run_config(phase_name: str, defaults: dict[str, Any]) -> dict[str, Any]:
    run_config = build_default_run_config(phase_name)
    run_config.train_split = defaults["train_split"]
    run_config.eval_split = defaults["eval_split"]
    run_config.output_root = defaults["output_root"]
    run_config = apply_hardware_profile(run_config, defaults["hardware_profile"])
    return asdict(run_config)


def effective_phase_config(run_config: dict[str, Any], phase_name: str) -> dict[str, Any]:
    return dict(run_config.get("phases", {}).get(phase_name, {}))


def effective_trainer_defaults(run_config: dict[str, Any], phase_name: str) -> dict[str, Any]:
    trainer = dict(run_config.get("trainer_defaults", {}))
    trainer.update(effective_phase_config(run_config, phase_name).get("trainer_overrides", {}))
    return trainer


def extract_initial_reward_weights(run_config: dict[str, Any], phase_name: str) -> dict[str, Any]:
    phase_cfg = effective_phase_config(run_config, phase_name)
    reward_cfg = phase_cfg.get("reward_components", {})
    return {
        "Configured Initial Correctness Weight": reward_cfg.get("correctness_reward", {}).get("initial_weight"),
        "Configured Initial Formatting Weight": reward_cfg.get("format_reward", {}).get("initial_weight"),
        "Configured Initial Parseability Weight": reward_cfg.get("parseable_reward", {}).get("initial_weight"),
        "Configured Initial Finished Weight": reward_cfg.get("finished_reward", {}).get("initial_weight"),
        "Configured Initial Tolerance Weight": reward_cfg.get("tolerance_reward", {}).get("initial_weight"),
        "Configured Initial Brevity Weight": reward_cfg.get("brevity_reward", {}).get("initial_weight"),
    }


def build_phase_context(source: dict[str, Any], phase_name: str) -> dict[str, Any]:
    phase_dir = source["root"] / phase_name
    run_config_path = phase_dir / "run_config.json"
    run_request_path = phase_dir / "run_request.json"
    run_config = json_load(run_config_path) or build_fallback_run_config(phase_name, source["defaults"])
    run_request = json_load(run_request_path)
    trainer = effective_trainer_defaults(run_config, phase_name)
    model = run_config.get("model", {})
    phase_cfg = effective_phase_config(run_config, phase_name)
    reward_initials = extract_initial_reward_weights(run_config, phase_name)
    compilation_config = (model.get("fast_inference_kwargs") or {}).get("compilation_config") or {}
    compilation_mode = compilation_config.get("cudagraph_mode", "")
    compilation_level = compilation_config.get("level")
    compilation_display = ""
    if compilation_mode or compilation_level is not None:
        parts = []
        if compilation_mode:
            parts.append(str(compilation_mode))
        if compilation_level is not None:
            parts.append(f"level={compilation_level}")
        compilation_display = ", ".join(parts)
    return {
        "phase_dir": phase_dir,
        "run_config": run_config,
        "run_request": run_request,
        "phase_cfg": phase_cfg,
        "trainer": trainer,
        "model": model,
        "diagnostics_path": phase_dir / "post_training_diagnostics.json",
        "train_log_summary_path": phase_dir / "train_log_summary.json",
        "summary_defaults": {
            "Train Split": run_request.get("train_split") or run_config.get("train_split") or source["defaults"]["train_split"],
            "Eval Split": run_request.get("eval_split") or run_config.get("eval_split") or source["defaults"]["eval_split"],
            "Hardware Profile": run_request.get("hardware_profile")
            or run_config.get("hardware_profile_name")
            or source["defaults"]["hardware_profile"],
            "Seed / Resume Source": run_request.get("resume") or "",
            "Warm-Start Checkpoint": run_request.get("warm_start_checkpoint") or "",
            "Training Strategy Introduced": phase_strategy_text(phase_name, source["run_family"]),
            "Stage Mix / Curriculum": flatten_stage_mix(phase_cfg.get("stage_mix", {})),
            "Phase Description": phase_cfg.get("description", ""),
            "Phase Default Resume Selector": phase_cfg.get("default_resume", {}).get("selector", ""),
            "Base Model": model.get("base_model_name", ""),
            "4-bit Enabled": model.get("load_in_4bit"),
            "Fast Inference Enabled": model.get("fast_inference"),
            "Fast Inference Mode": "vLLM" if model.get("fast_inference") else "",
            "Compilation Mode": compilation_display,
            "max_seq_length": model.get("max_seq_length"),
            "image_size": model.get("image_size"),
            "LoRA Rank": model.get("lora_rank"),
            "Max LoRA Rank": model.get("max_lora_rank"),
            "gradient_accumulation_steps": trainer.get("gradient_accumulation_steps"),
            "gpu_memory_utilization": model.get("gpu_memory_utilization"),
            "num_generations": trainer.get("num_generations"),
            "max_prompt_length": trainer.get("max_prompt_length"),
            "max_completion_length": trainer.get("max_completion_length"),
            "max_eval_examples_per_subset": run_config.get("eval", {}).get("max_eval_examples_per_subset"),
        }
        | reward_initials,
        "run_config_path": run_config_path,
        "run_request_path": run_request_path,
        "compilation_display": compilation_display,
    }


def base_row_template(source: dict[str, Any], phase_name: str, context: dict[str, Any]) -> dict[str, Any]:
    return {
        "Milestone ID": "",
        "Run Family": source["run_family"],
        "Run Family Label": source["run_family_label"],
        "Notebook / Run Slug": source["notebook_slug"],
        "Output Root": source["output_root_label"],
        "Phase": phase_name,
        "Phase Label": phase_display(source["run_family"], phase_name),
        "Checkpoint": "",
        "Global Step": "",
        "Timeline Order": "",
        "X Label": "",
        "Artifact Status": "",
        "Metrics Source Kind": "",
        "Alias Role": "",
        "Alias Roles": "",
        "Keep / Discard": "",
        "Decision Reason": "",
        "Key Interpretation": "",
        "Checkpoint Rank Within Phase": "",
        "Is Best Composite": False,
        "Is Best Correctness": False,
        "Is Best Structure": False,
        "Is Latest": False,
        "Controller History Length": "",
        "Evidence Root": str(source["root"]),
        "Run Config Path": maybe_path(context["run_config_path"]),
        "Run Request Path": maybe_path(context["run_request_path"]),
        "Diagnostics Path": maybe_path(context["diagnostics_path"]),
        "Train Log Summary Path": maybe_path(context["train_log_summary_path"]),
        "Notes": "",
        **context["summary_defaults"],
    }


def enrich_metrics(row: dict[str, Any], metrics: dict[str, Any]) -> None:
    structure_score = metrics.get("structure_score")
    correctness_score = metrics.get("correctness_score")
    composite_score = metrics.get("composite_score")
    if structure_score is None or correctness_score is None or composite_score is None:
        computed = compute_scores(metrics)
        structure_score = structure_score if structure_score is not None else computed["structure_score"]
        correctness_score = correctness_score if correctness_score is not None else computed["correctness_score"]
        composite_score = composite_score if composite_score is not None else computed["composite_score"]

    malformed = metrics.get("malformed_answer_rate")
    truncation = metrics.get("truncation_rate")
    row.update(
        {
            "Correctness Reward Mean": metrics.get("reward_component/correctness_reward_mean"),
            "Format Reward Mean": format_reward_mean(metrics),
            "Parseable Reward Mean": metrics.get("reward_component/parseable_reward_mean"),
            "Finished Reward Mean": metrics.get("reward_component/finished_reward_mean"),
            "Tolerance Reward Mean": metrics.get("reward_component/tolerance_reward_mean"),
            "Exact Match": metrics.get("normalized_exact_match"),
            "Tolerance Accuracy": metrics.get("tolerance_accuracy"),
            "Best-of-k Accuracy": best_of_k_value(metrics),
            "Best-of-k Tolerance Accuracy": best_of_k_tol_value(metrics),
            "Sample-Level Exact Match": metrics.get("sample_level_normalized_exact_match"),
            "Sample-Level Tolerance Accuracy": metrics.get("sample_level_tolerance_accuracy"),
            "Parseable Rate": metrics.get("parseable_answer_rate"),
            "Reasoning Tag Compliance": metrics.get("reasoning_tag_compliance"),
            "Solution Tag Compliance": metrics.get("solution_tag_compliance"),
            "Well-Formed Rate": None if malformed is None else 1.0 - float(malformed),
            "Malformed Rate": malformed,
            "Completion Success Rate": None if truncation is None else 1.0 - float(truncation),
            "Truncation Rate": truncation,
            "Average Completion Tokens": metrics.get("average_completion_tokens"),
            "Repetition Rate": metrics.get("repetition_rate"),
            "Sample Diversity": metrics.get("sampled_answer_diversity"),
            "Average Total Reward": total_reward_mean(metrics),
            "Structure Score": structure_score,
            "Correctness Score": correctness_score,
            "Composite Score": composite_score,
        }
    )


def enrich_train_summary(row: dict[str, Any], train_summary_path: Path) -> None:
    train_summary = json_load(train_summary_path)
    row["KL Mean"] = train_summary.get("KL_mean")
    row["KL P95"] = train_summary.get("KL_p95")


def enrich_reward_weights(row: dict[str, Any], reward_weights_path: Path, controller_state_path: Path) -> None:
    reward_weights = json_load(reward_weights_path)
    controller_state = json_load(controller_state_path)
    row.update(
        {
            "Correctness Weight": reward_weights.get("correctness_reward"),
            "Formatting Weight": reward_weights.get("format_reward"),
            "Parseability Weight": reward_weights.get("parseable_reward"),
            "Finished Weight": reward_weights.get("finished_reward"),
            "Tolerance Weight": reward_weights.get("tolerance_reward"),
            "Brevity Weight": reward_weights.get("brevity_reward"),
            "Reward Weights Path": maybe_path(reward_weights_path),
            "Controller State Path": maybe_path(controller_state_path),
            "Controller History Length": len(controller_state.get("history", [])) if controller_state else "",
        }
    )


def collect_phase_rows(source: dict[str, Any], phase_name: str) -> list[dict[str, Any]]:
    context = build_phase_context(source, phase_name)
    phase_dir = context["phase_dir"]
    rows_by_checkpoint: dict[str, dict[str, Any]] = {}

    for checkpoint_dir in sorted(phase_dir.glob("checkpoint-*"), key=lambda path: parse_checkpoint_step(path.name) or -1):
        eval_metrics_path = checkpoint_dir / "eval_metrics.json"
        if not eval_metrics_path.exists():
            continue
        checkpoint_info_path = checkpoint_dir / "checkpoint_info.json"
        checkpoint_info = json_load(checkpoint_info_path)
        metrics = checkpoint_info.get("metrics") or json_load(eval_metrics_path)
        row = base_row_template(source, phase_name, context)
        checkpoint_rel = f"{source['output_root_label']}/{phase_name}/{checkpoint_dir.name}"
        row.update(
            {
                "Checkpoint": checkpoint_dir.name,
                "Global Step": checkpoint_info.get("global_step") or parse_checkpoint_step(checkpoint_dir.name),
                "Artifact Status": "actual_checkpoint",
                "Metrics Source Kind": "checkpoint_artifact",
                "Checkpoint Path": checkpoint_info.get("checkpoint_path", checkpoint_rel),
                "Checkpoint Path Abs": str(checkpoint_dir.resolve()),
                "Metrics Source Path": str(eval_metrics_path),
                "Checkpoint Info Path": maybe_path(checkpoint_info_path),
                "Summary Path": maybe_path(checkpoint_dir / "summary.txt"),
            }
        )
        enrich_metrics(row, metrics)
        enrich_train_summary(row, context["train_log_summary_path"])
        enrich_reward_weights(row, checkpoint_dir / "reward_weights.json", checkpoint_dir / "controller_state.json")
        rows_by_checkpoint[row["Checkpoint Path"]] = row

    alias_dir = phase_dir / "aliases"
    for alias_path in sorted(alias_dir.glob("*.json")):
        alias_data = json_load(alias_path)
        checkpoint_rel = alias_data.get("checkpoint_path", "")
        alias_role = alias_path.stem
        row = rows_by_checkpoint.get(checkpoint_rel)
        if row is None:
            row = base_row_template(source, phase_name, context)
            row.update(
                {
                    "Checkpoint": Path(checkpoint_rel).name if checkpoint_rel else "",
                    "Global Step": alias_data.get("global_step", ""),
                    "Artifact Status": "alias_metadata_only",
                    "Metrics Source Kind": "alias_metadata",
                    "Checkpoint Path": checkpoint_rel,
                    "Checkpoint Path Abs": str((phase_dir.parent.parent / checkpoint_rel).resolve()) if checkpoint_rel else "",
                    "Metrics Source Path": str(alias_path),
                    "Notes": "Checkpoint metrics are available only through alias metadata in the current export bundle.",
                }
            )
            enrich_metrics(row, alias_data.get("metrics", {}))
            enrich_train_summary(row, context["train_log_summary_path"])
            rows_by_checkpoint[checkpoint_rel] = row
        alias_roles = set(filter(None, row.get("Alias Roles", "").split(";")))
        alias_roles.add(alias_role)
        row["Alias Roles"] = ";".join(sorted(alias_roles))

    rows = list(rows_by_checkpoint.values())
    rows.sort(key=lambda item: int(item["Global Step"]) if item["Global Step"] != "" else 10**9)
    phase_rank = {
        row["Checkpoint Path"]: index
        for index, row in enumerate(
            sorted(rows, key=lambda item: int(item["Global Step"]) if item["Global Step"] != "" else 10**9), start=1
        )
    }
    for row in rows:
        alias_roles = set(filter(None, row.get("Alias Roles", "").split(";")))
        row["Alias Role"] = select_primary_alias(alias_roles)
        row["Is Best Composite"] = "best_composite" in alias_roles
        row["Is Best Correctness"] = "best_correctness" in alias_roles
        row["Is Best Structure"] = "best_structure" in alias_roles
        row["Is Latest"] = "latest" in alias_roles
        row["Checkpoint Rank Within Phase"] = phase_rank.get(row["Checkpoint Path"], "")
        row["X Label"] = f"{phase_name}:{row['Global Step']}" if row["Global Step"] != "" else phase_name
        row["Notes"] = row["Notes"] or ""
        row["Keep / Discard"], row["Decision Reason"] = keep_discard(row)
    return rows


def collect_baseline_row() -> dict[str, Any]:
    metrics = json_load(BASELINE_METRICS_PATH)
    row = {column: "" for column in MASTER_TABLE_COLUMNS}
    row.update(
        {
            "Milestone ID": "",
            "Run Family": "baseline_pre_refactor",
            "Run Family Label": "Pre-Refactor Baseline",
            "Notebook / Run Slug": "local_pre_refactor_snapshot",
            "Output Root": "grpo_eval_outputs",
            "Phase": "baseline",
            "Phase Label": "Baseline",
            "Checkpoint": "baseline",
            "Global Step": 0,
            "Artifact Status": "baseline_snapshot",
            "Metrics Source Kind": "local_eval_snapshot",
            "Checkpoint Path": "grpo_eval_outputs/eval_metrics.json",
            "Checkpoint Path Abs": str(BASELINE_METRICS_PATH.resolve()),
            "Evidence Root": str(ROOT / "grpo_eval_outputs"),
            "Metrics Source Path": str(BASELINE_METRICS_PATH),
            "Train Log Summary Path": maybe_path(BASELINE_TRAIN_SUMMARY_PATH),
            "Train Split": "unknown_pre_refactor",
            "Eval Split": "unknown_pre_refactor",
            "Hardware Profile": "local_pre_refactor",
            "Training Strategy Introduced": "Pre-refactor RL baseline without staged curriculum or metric-aware checkpointing.",
            "Stage Mix / Curriculum": "mixed numeric free-form baseline",
            "Phase Description": "Legacy local evaluation snapshot before the staged RL refactor.",
            "Base Model": "unsloth/Qwen2.5-VL-7B-Instruct",
            "Keep / Discard": "reference",
            "Decision Reason": "Reference-only baseline used for before/after comparison.",
            "Key Interpretation": "Structure was still weak and completions were long before the staged RL refactor.",
            "X Label": "baseline",
        }
    )
    enrich_metrics(row, metrics)
    enrich_train_summary(row, BASELINE_TRAIN_SUMMARY_PATH)
    row["Timeline Order"] = 0
    return row


def build_all_checkpoint_rows() -> list[dict[str, Any]]:
    rows = [collect_baseline_row()]
    for source in RUN_SOURCES:
        for phase_name in source["phases"]:
            rows.extend(collect_phase_rows(source, phase_name))

    rows.sort(
        key=lambda item: (
            0 if item["Run Family"] == "baseline_pre_refactor" else PHASE_SEQUENCE.get((item["Run Family"], item["Phase"]), 99),
            int(item["Global Step"]) if item["Global Step"] != "" else 10**9,
            item["Artifact Status"],
        )
    )
    for index, row in enumerate(rows):
        row["Timeline Order"] = index
    return rows


def find_row(
    rows: list[dict[str, Any]],
    *,
    run_family: str,
    phase: str,
    global_step: int | None = None,
    alias_role: str | None = None,
    checkpoint_path: str | None = None,
) -> dict[str, Any] | None:
    for row in rows:
        if row["Run Family"] != run_family or row["Phase"] != phase:
            continue
        if checkpoint_path and row["Checkpoint Path"] != checkpoint_path:
            continue
        if global_step is not None and str(row["Global Step"]) != str(global_step):
            continue
        if alias_role and alias_role not in row.get("Alias Roles", "").split(";"):
            continue
        return dict(row)
    return None


def build_milestone_rows(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        find_row(all_rows, run_family="baseline_pre_refactor", phase="baseline", global_step=0),
        find_row(all_rows, run_family="smoke_testmini", phase="phase_a", alias_role="best_composite"),
        find_row(all_rows, run_family="smoke_testmini", phase="phase_b", alias_role="best_composite"),
        find_row(all_rows, run_family="smoke_testmini", phase="phase_b", alias_role="latest"),
        find_row(all_rows, run_family="smoke_testmini", phase="phase_c", alias_role="best_composite"),
        find_row(all_rows, run_family="smoke_testmini", phase="phase_d", alias_role="best_composite"),
        find_row(all_rows, run_family="large_split_continue", phase="phase_c", alias_role="best_composite"),
        find_row(all_rows, run_family="large_split_continue", phase="phase_c", alias_role="latest"),
        find_row(all_rows, run_family="large_split_continue", phase="phase_d", alias_role="best_composite"),
        find_row(all_rows, run_family="large_split_continue", phase="phase_d", alias_role="latest"),
        find_row(all_rows, run_family="phase_d_dedicated", phase="phase_d", global_step=120),
        find_row(all_rows, run_family="phase_d_dedicated", phase="phase_d", alias_role="best_composite"),
    ]
    milestone_rows = [row for row in selected if row]
    milestone_ids = [
        "baseline_local_snapshot",
        "smoke_phase_a_best",
        "smoke_phase_b_best",
        "smoke_phase_b_latest_regression",
        "smoke_phase_c_best",
        "smoke_phase_d_best",
        "large_phase_c_best_recommended",
        "large_phase_c_latest",
        "large_phase_d_same_notebook_best",
        "large_phase_d_same_notebook_latest",
        "dedicated_phase_d_early_regression",
        "dedicated_phase_d_best",
    ]
    interpretations = {
        "baseline_local_snapshot": (
            "reference",
            "Reference-only baseline used for before/after comparison.",
            "Pre-refactor baseline: structure was unreliable, truncation was common, and completions were long.",
        ),
        "smoke_phase_a_best": (
            "keep_secondary",
            "Smoke Phase A best checkpoint kept as the first stable structure milestone.",
            "Phase A began structure stabilization but still had 50% malformed and truncated outputs on the tiny smoke eval.",
        ),
        "smoke_phase_b_best": (
            "keep_secondary",
            "Smoke Phase B best checkpoint showed the ceiling of the small-split smoke run.",
            "Phase B achieved perfect structure on the smoke eval, but correctness plateaued at 0.5.",
        ),
        "smoke_phase_b_latest_regression": (
            "discard",
            "Latest smoke Phase B checkpoint regressed correctness while structure stayed perfect.",
            "This is the clearest smoke-run example that later checkpoints were not automatically better.",
        ),
        "smoke_phase_c_best": (
            "keep_secondary",
            "Smoke Phase C best checkpoint kept because it seeded the later larger-split continuation.",
            "Smoke Phase C preserved perfect structure but still sat at the 0.5 correctness plateau that the larger split later broke.",
        ),
        "smoke_phase_d_best": (
            "keep_secondary",
            "Smoke Phase D best checkpoint kept as the last small-split specialization milestone.",
            "Smoke Phase D did not improve over Smoke Phase C; structure regressed back to 0.25 despite unchanged 0.5 exact match.",
        ),
        "large_phase_c_best_recommended": (
            "keep_primary",
            "Recommended final checkpoint: best larger-split Phase C composite result.",
            "Larger-split Phase C is the primary deployable checkpoint: structure stayed perfect and exact/tolerance improved to 0.75.",
        ),
        "large_phase_c_latest": (
            "keep",
            "Latest larger-split Phase C checkpoint matched the best composite result.",
            "In this specific run, latest happened to tie the best checkpoint, but the non-monotonic curve still justifies alias-based selection.",
        ),
        "large_phase_d_same_notebook_best": (
            "keep_secondary",
            "Same-notebook Phase D best checkpoint kept as a specialization branch reference.",
            "Phase D in the same notebook recovered to 0.75 early, but the run later regressed.",
        ),
        "large_phase_d_same_notebook_latest": (
            "discard",
            "Latest same-notebook Phase D checkpoint regressed below the recommended Phase C checkpoint.",
            "This row shows why the dedicated Phase D rerun was needed: the same-notebook branch finished weaker than its own earlier checkpoint.",
        ),
        "dedicated_phase_d_early_regression": (
            "discard",
            "Dedicated Phase D early checkpoint kept to show the temporary dip before recovery.",
            "Dedicated Phase D started from a weaker 0.25 exact checkpoint while structure remained perfect, then recovered later.",
        ),
        "dedicated_phase_d_best": (
            "keep_secondary",
            "Dedicated Phase D best checkpoint preserved as a specialization branch outcome.",
            "Dedicated Phase D recovered to 0.75 exact/tolerance, matching but not exceeding the larger-split Phase C best.",
        ),
    }

    for milestone_id, row in zip(milestone_ids, milestone_rows):
        decision, reason, interpretation = interpretations[milestone_id]
        row["Milestone ID"] = milestone_id
        row["Keep / Discard"] = decision
        row["Decision Reason"] = reason
        row["Key Interpretation"] = interpretation

    previous_exact = None
    previous_composite = None
    for row in milestone_rows:
        exact = row.get("Exact Match")
        composite = row.get("Composite Score")
        if isinstance(exact, (int, float)) and isinstance(previous_exact, (int, float)):
            row["Delta Exact vs Previous Milestone"] = exact - previous_exact
        else:
            row["Delta Exact vs Previous Milestone"] = ""
        if isinstance(composite, (int, float)) and isinstance(previous_composite, (int, float)):
            row["Delta Composite vs Previous Milestone"] = composite - previous_composite
        else:
            row["Delta Composite vs Previous Milestone"] = ""
        if isinstance(exact, (int, float)):
            previous_exact = exact
        if isinstance(composite, (int, float)):
            previous_composite = composite

    return milestone_rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            output = {column: row.get(column, "") for column in columns}
            writer.writerow(output)


def checkpoint_rows_for_plot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["Artifact Status"] != "planned_milestone_missing"]


def minmax_normalize(values: list[float | None], invert: bool = False) -> list[float | None]:
    valid = [value for value in values if isinstance(value, (int, float))]
    if not valid:
        return [None for _ in values]
    lo = min(valid)
    hi = max(valid)
    normalized: list[float | None] = []
    for value in values:
        if not isinstance(value, (int, float)):
            normalized.append(None)
            continue
        scaled = 1.0 if math.isclose(hi, lo) else (float(value) - lo) / (hi - lo)
        normalized.append(1.0 - scaled if invert else scaled)
    return normalized


def plot_main_evolution(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_rows = checkpoint_rows_for_plot(rows)
    x = [row["Timeline Order"] for row in plot_rows]
    labels = [row["X Label"] for row in plot_rows]

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(22, 20),
        sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.2, 1.2, 1.2, 0.55], "hspace": 0.08},
    )

    segments: list[tuple[int, int, str]] = []
    if plot_rows:
        current_label = plot_rows[0]["Phase Label"]
        start = x[0]
        previous_x = x[0]
        for row in plot_rows[1:]:
            if row["Phase Label"] != current_label:
                segments.append((start, previous_x, current_label))
                current_label = row["Phase Label"]
                start = row["Timeline Order"]
            previous_x = row["Timeline Order"]
        segments.append((start, previous_x, current_label))

    def add_segments(ax: Any) -> None:
        for index, (start, end, label) in enumerate(segments):
            ax.axvspan(start - 0.5, end + 0.5, color=("#f7f7f7" if index % 2 == 0 else "#eef3f9"), alpha=0.7, zorder=0)
            ax.axvline(end + 0.5, color="#bbbbbb", linestyle="--", linewidth=0.7)
            ax.text((start + end) / 2, 1.02, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)

    def extract_series(key: str) -> list[float | None]:
        return [row.get(key) if isinstance(row.get(key), (int, float)) else None for row in plot_rows]

    panel_specs = [
        (
            axes[0],
            "Reward Weight Evolution",
            [
                ("Correctness Weight", "#d62728"),
                ("Formatting Weight", "#1f77b4"),
                ("Parseability Weight", "#2ca02c"),
                ("Finished Weight", "#9467bd"),
                ("Tolerance Weight", "#ff7f0e"),
                ("Brevity Weight", "#8c564b"),
            ],
            True,
            "weight",
        ),
        (
            axes[1],
            "Structure Evolution",
            [
                ("Parseable Rate", "#2ca02c"),
                ("Reasoning Tag Compliance", "#1f77b4"),
                ("Solution Tag Compliance", "#ff7f0e"),
                ("Well-Formed Rate", "#9467bd"),
                ("Completion Success Rate", "#d62728"),
            ],
            False,
            "rate",
        ),
        (
            axes[2],
            "Accuracy Evolution",
            [
                ("Exact Match", "#d62728"),
                ("Tolerance Accuracy", "#ff7f0e"),
                ("Best-of-k Accuracy", "#1f77b4"),
                ("Composite Score", "#2ca02c"),
            ],
            False,
            "score",
        ),
        (
            axes[3],
            "Generation Behavior (normalized mixed-unit view)",
            [
                ("Average Completion Tokens", "#d62728"),
                ("Repetition Rate", "#1f77b4"),
                ("Sample Diversity", "#2ca02c"),
                ("Average Total Reward", "#9467bd"),
            ],
            False,
            "normalized",
        ),
    ]

    for ax, title, series_specs, step_mode, y_label in panel_specs:
        add_segments(ax)
        for key, color in series_specs:
            values = extract_series(key)
            if title == "Generation Behavior (normalized mixed-unit view)":
                values = minmax_normalize(values, invert=(key == "Average Completion Tokens"))
            if step_mode:
                ax.step(x, [math.nan if value is None else value for value in values], where="mid", label=key, color=color, linewidth=2)
                ax.scatter(x, values, color=color, s=25, zorder=3)
            else:
                ax.plot(x, values, marker="o", label=key, color=color, linewidth=2)
        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=12, loc="left")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper left", ncols=3, fontsize=8)

    decision_ax = axes[4]
    add_segments(decision_ax)
    y_map = {
        "discard": 0.0,
        "unavailable": 0.2,
        "reference": 0.5,
        "keep_secondary": 0.9,
        "keep": 1.0,
        "keep_primary": 1.0,
    }
    marker_map = {
        "best_composite": "*",
        "best_correctness": "D",
        "best_structure": "s",
        "latest": "o",
        "none": "P",
    }
    color_map = {
        "discard": "#d62728",
        "unavailable": "#ff7f0e",
        "reference": "#7f7f7f",
        "keep_secondary": "#1f77b4",
        "keep": "#2ca02c",
        "keep_primary": "#2ca02c",
    }
    for row in plot_rows:
        decision_ax.scatter(
            row["Timeline Order"],
            y_map.get(row["Keep / Discard"], 0.0),
            marker=marker_map.get(row["Alias Role"], "o"),
            color=color_map.get(row["Keep / Discard"], "#333333"),
            s=180 if row["Keep / Discard"] == "keep_primary" else 110,
            zorder=3,
            edgecolor="black",
            linewidth=0.5,
        )
    decision_ax.set_yticks([0.0, 0.5, 1.0])
    decision_ax.set_yticklabels(["discard", "reference", "keep"])
    decision_ax.set_ylim(-0.15, 1.2)
    decision_ax.set_title("Checkpoint Decisions / Alias Roles", fontsize=12, loc="left")
    decision_ax.grid(True, axis="y", alpha=0.25)
    decision_ax.set_xticks(x)
    decision_ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    legend_items = [
        Line2D([0], [0], marker="*", color="w", label="best_composite", markerfacecolor="black", markeredgecolor="black", markersize=12),
        Line2D([0], [0], marker="D", color="w", label="best_correctness", markerfacecolor="black", markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="s", color="w", label="best_structure", markerfacecolor="black", markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="latest", markerfacecolor="black", markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="keep", markerfacecolor="#2ca02c", markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="discard", markerfacecolor="#d62728", markeredgecolor="black", markersize=9),
    ]
    decision_ax.legend(handles=legend_items, loc="upper left", ncols=3, fontsize=8)

    fig.suptitle("RL Fine-Tuning Evolution Across Smoke, Large-Split, and Dedicated Phase D Runs", fontsize=16)
    fig.savefig(PLOTS_DIR / "evolution_panels.png", dpi=220, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "evolution_panels.svg", bbox_inches="tight")
    plt.close(fig)


def build_resource_rows(milestones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    default_cfg = asdict(build_default_run_config("phase_a"))
    default_phase = default_cfg["phases"]["phase_a"]
    default_trainer = dict(default_cfg["trainer_defaults"])
    default_model = dict(default_cfg["model"])
    default_eval = dict(default_cfg["eval"])

    def row_for_milestone(milestone_id: str) -> dict[str, Any]:
        for row in milestones:
            if row["Milestone ID"] == milestone_id:
                return row
        raise KeyError(milestone_id)

    stable_rows = [
        row_for_milestone("smoke_phase_a_best"),
        row_for_milestone("large_phase_c_best_recommended"),
        row_for_milestone("large_phase_d_same_notebook_best"),
        row_for_milestone("dedicated_phase_d_best"),
    ]

    def diff_summary(target: dict[str, Any]) -> str:
        changes = []
        comparisons = [
            ("max_seq_length", default_model.get("max_seq_length"), target.get("max_seq_length")),
            ("image_size", default_model.get("image_size"), target.get("image_size")),
            ("LoRA Rank", default_model.get("lora_rank"), target.get("LoRA Rank")),
            ("num_generations", default_trainer.get("num_generations"), target.get("num_generations")),
            ("max_prompt_length", default_trainer.get("max_prompt_length"), target.get("max_prompt_length")),
            ("max_completion_length", default_trainer.get("max_completion_length"), target.get("max_completion_length")),
            ("gpu_memory_utilization", default_model.get("gpu_memory_utilization"), target.get("gpu_memory_utilization")),
        ]
        for label, before, after in comparisons:
            if before != after:
                changes.append(f"{label} {before}\u2192{after}")
        return "; ".join(changes)

    common_memory_help = (
        "Lowered sequence, image, prompt, and generation budgets reduce KV-cache, activation, and vision-forward memory; "
        "lower LoRA rank and conservative vLLM utilization reduce trainable-state and allocator pressure."
    )
    common_tradeoff = (
        "Longer wall-clock time per unit of signal, shorter reasoning budget, lower visual detail, and lower generation diversity."
    )

    rows = [
        {
            "Run Family": "reference_default_profile",
            "Notebook / Run Slug": "code_default_reference",
            "Hardware": "Reference high-capacity profile",
            "Observed Constraint / Failure Mode": "Reference configuration used for comparison against the stable Kaggle T4 profile.",
            "Train Split": "test",
            "Eval Split": "testmini",
            "Hardware Profile": "default",
            "Base Model": default_model.get("base_model_name"),
            "4-bit Enabled": default_model.get("load_in_4bit"),
            "LoRA Rank": default_model.get("lora_rank"),
            "Max LoRA Rank": default_model.get("max_lora_rank"),
            "max_seq_length": default_model.get("max_seq_length"),
            "image_size": default_model.get("image_size"),
            "num_generations": default_trainer.get("num_generations"),
            "max_prompt_length": default_trainer.get("max_prompt_length"),
            "max_completion_length": default_trainer.get("max_completion_length"),
            "gradient_accumulation_steps": default_trainer.get("gradient_accumulation_steps"),
            "gpu_memory_utilization": default_model.get("gpu_memory_utilization"),
            "fast_inference enabled": default_model.get("fast_inference"),
            "vLLM version": "not_logged",
            "cudagraph / compilation mode": "default / not explicitly pinned",
            "Warm start used?": False,
            "What knob changed": "Reference row only.",
            "Why this helps memory": "",
            "Tradeoff introduced": "",
            "Outcome": "Reference profile only; not the stable Kaggle T4 execution profile.",
        }
    ]

    outcomes = {
        "smoke_phase_a_best": "Completed smoke-run validation on testmini; first stable end-to-end Kaggle T4 Phase A checkpoint.",
        "large_phase_c_best_recommended": "Completed larger-split continuation and produced the recommended final checkpoint at 0.75 exact / 0.75 composite.",
        "large_phase_d_same_notebook_best": "Completed same-notebook Phase D branch; best checkpoint reached 0.75 but latest regressed.",
        "dedicated_phase_d_best": "Completed dedicated Phase D notebook; recovered to 0.75 and preserved earlier notebook outputs.",
    }
    constraint_text = {
        "smoke_phase_a_best": "Needed a low-VRAM proof run to verify the staged RL loop on Kaggle T4.",
        "large_phase_c_best_recommended": "Needed to scale to the larger split without exceeding the same per-step VRAM budget.",
        "large_phase_d_same_notebook_best": "Needed to continue into Phase D inside the same Kaggle notebook budget.",
        "dedicated_phase_d_best": "Needed a separate Phase D branch while preserving earlier notebook artifacts for analysis.",
    }

    milestone_by_id = {row["Milestone ID"]: row for row in milestones}
    for milestone_id in [
        "smoke_phase_a_best",
        "large_phase_c_best_recommended",
        "large_phase_d_same_notebook_best",
        "dedicated_phase_d_best",
    ]:
        row = milestone_by_id[milestone_id]
        rows.append(
            {
                "Run Family": row["Run Family"],
                "Notebook / Run Slug": row["Notebook / Run Slug"],
                "Hardware": "Kaggle NvidiaTeslaT4",
                "Observed Constraint / Failure Mode": constraint_text[milestone_id],
                "Train Split": row["Train Split"],
                "Eval Split": row["Eval Split"],
                "Hardware Profile": row["Hardware Profile"],
                "Base Model": row["Base Model"],
                "4-bit Enabled": row["4-bit Enabled"],
                "LoRA Rank": row["LoRA Rank"],
                "Max LoRA Rank": row["Max LoRA Rank"],
                "max_seq_length": row["max_seq_length"],
                "image_size": row["image_size"],
                "num_generations": row["num_generations"],
                "max_prompt_length": row["max_prompt_length"],
                "max_completion_length": row["max_completion_length"],
                "gradient_accumulation_steps": row["gradient_accumulation_steps"],
                "gpu_memory_utilization": row["gpu_memory_utilization"],
                "fast_inference enabled": row["Fast Inference Enabled"],
                "vLLM version": "not_logged",
                "cudagraph / compilation mode": row["Compilation Mode"],
                "Warm start used?": bool(row["Warm-Start Checkpoint"] or row["Seed / Resume Source"]),
                "What knob changed": diff_summary(row),
                "Why this helps memory": common_memory_help,
                "Tradeoff introduced": common_tradeoff
                + (" Smaller eval coverage per checkpoint was also accepted." if row["max_eval_examples_per_subset"] else ""),
                "Outcome": outcomes[milestone_id],
            }
        )

    return rows


def build_knob_tradeoff_rows() -> list[dict[str, Any]]:
    default_cfg = build_default_run_config("phase_a")
    kaggle_cfg = build_default_run_config("phase_a")
    kaggle_cfg = apply_hardware_profile(kaggle_cfg, "kaggle_t4")

    default_model = default_cfg.model
    kaggle_model = kaggle_cfg.model
    default_trainer = default_cfg.trainer_defaults
    kaggle_trainer = kaggle_cfg.trainer_defaults
    default_eval = default_cfg.eval
    kaggle_eval = kaggle_cfg.eval

    return [
        {
            "Knob": "max_seq_length",
            "Default": default_model.max_seq_length,
            "Kaggle T4": kaggle_model.max_seq_length,
            "Why it reduced memory pressure": "Shorter sequence length lowers attention KV-cache and activation footprint.",
            "Tradeoff accepted": "Less room for long-context reasoning.",
        },
        {
            "Knob": "image_size",
            "Default": default_model.image_size,
            "Kaggle T4": kaggle_model.image_size,
            "Why it reduced memory pressure": "Smaller images make the vision encoder cheaper during generation and training.",
            "Tradeoff accepted": "Lower visual detail fidelity.",
        },
        {
            "Knob": "LoRA Rank",
            "Default": default_model.lora_rank,
            "Kaggle T4": kaggle_model.lora_rank,
            "Why it reduced memory pressure": "Lower-rank adapters shrink trainable state and optimizer footprint.",
            "Tradeoff accepted": "Lower adaptation capacity.",
        },
        {
            "Knob": "num_generations",
            "Default": default_trainer.num_generations,
            "Kaggle T4": kaggle_trainer.num_generations,
            "Why it reduced memory pressure": "Fewer sampled completions reduce generation-time memory pressure and runtime.",
            "Tradeoff accepted": "Lower exploration / diversity per prompt.",
        },
        {
            "Knob": "max_prompt_length",
            "Default": default_trainer.max_prompt_length,
            "Kaggle T4": kaggle_trainer.max_prompt_length,
            "Why it reduced memory pressure": "Shorter prompt budget reduces prefill memory and attention cache size.",
            "Tradeoff accepted": "Less room for long question context.",
        },
        {
            "Knob": "max_completion_length",
            "Default": default_trainer.max_completion_length,
            "Kaggle T4": kaggle_trainer.max_completion_length,
            "Why it reduced memory pressure": "Shorter completion budget caps generation-time memory and reduces truncation risk.",
            "Tradeoff accepted": "Less room for verbose reasoning.",
        },
        {
            "Knob": "gpu_memory_utilization",
            "Default": default_model.gpu_memory_utilization,
            "Kaggle T4": kaggle_model.gpu_memory_utilization,
            "Why it reduced memory pressure": "A more conservative allocator target reduces vLLM cache pressure on T4.",
            "Tradeoff accepted": "Leaves less headroom for aggressive throughput.",
        },
        {
            "Knob": "compilation mode",
            "Default": "default / unspecified",
            "Kaggle T4": "PIECEWISE, level=3",
            "Why it reduced memory pressure": "T4-safe piecewise cudagraph mode avoided unstable full-graph capture paths.",
            "Tradeoff accepted": "More conservative runtime path than the highest-throughput configuration.",
        },
        {
            "Knob": "max_eval_examples_per_subset",
            "Default": default_eval.max_eval_examples_per_subset,
            "Kaggle T4": kaggle_eval.max_eval_examples_per_subset,
            "Why it reduced memory pressure": "Smaller checkpoint-time eval batches reduce memory and runtime at save/eval boundaries.",
            "Tradeoff accepted": "Noisier checkpoint selection on tiny smoke runs.",
        },
    ]


def build_runtime_timeline_rows(milestones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = [
        "baseline_local_snapshot",
        "smoke_phase_a_best",
        "smoke_phase_b_best",
        "smoke_phase_c_best",
        "smoke_phase_d_best",
        "large_phase_c_best_recommended",
        "large_phase_d_same_notebook_best",
        "dedicated_phase_d_best",
    ]
    status_map = {
        "baseline_local_snapshot": "reference",
        "smoke_phase_a_best": "validated",
        "smoke_phase_b_best": "stabilized_structure",
        "smoke_phase_c_best": "ready_for_scale_up",
        "smoke_phase_d_best": "small_split_limit_reached",
        "large_phase_c_best_recommended": "improved_correctness",
        "large_phase_d_same_notebook_best": "matched_then_regressed",
        "dedicated_phase_d_best": "matched_specialization",
    }
    rows = []
    lookup = {row["Milestone ID"]: row for row in milestones}
    for order, milestone_id in enumerate(ids):
        row = lookup[milestone_id]
        rows.append(
            {
                "Timeline Order": order,
                "Event Label": row["Phase Label"] if milestone_id != "baseline_local_snapshot" else "Baseline",
                "Milestone ID": milestone_id,
                "Run Family": row["Run Family"],
                "Phase": row["Phase"],
                "Train Split": row["Train Split"],
                "Eval Split": row["Eval Split"],
                "Warm Start": row["Warm-Start Checkpoint"] or row["Seed / Resume Source"],
                "Status": status_map[milestone_id],
                "Best Exact": row["Exact Match"],
                "Best Composite": row["Composite Score"],
                "Key Change": row["Training Strategy Introduced"],
                "Interpretation": row["Key Interpretation"],
            }
        )
    return rows


def plot_runtime_timeline(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["Timeline Order"] for row in rows]
    colors = {
        "reference": "#7f7f7f",
        "validated": "#1f77b4",
        "stabilized_structure": "#2ca02c",
        "ready_for_scale_up": "#17becf",
        "small_split_limit_reached": "#bcbd22",
        "improved_correctness": "#d62728",
        "matched_then_regressed": "#ff7f0e",
        "matched_specialization": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(x, [0] * len(x), color="#6c757d", linewidth=2, zorder=1)
    for index, row in enumerate(rows):
        y = 0
        color = colors[row["Status"]]
        ax.scatter(row["Timeline Order"], y, s=180, color=color, edgecolor="black", zorder=3)
        dy = 0.36 if index % 2 == 0 else -0.48
        va = "bottom" if dy > 0 else "top"
        exact = row["Best Exact"]
        composite = row["Best Composite"]
        metrics_text = ""
        if isinstance(exact, (int, float)) and isinstance(composite, (int, float)):
            metrics_text = f"\nexact={exact:.2f} | composite={composite:.2f}"
        text = f"{row['Event Label']}\n{row['Train Split']}\n{row['Status']}{metrics_text}"
        ax.text(
            row["Timeline Order"],
            dy,
            text,
            ha="center",
            va=va,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": color, "alpha": 0.95},
        )

    ax.set_yticks([])
    ax.set_xticks(x)
    ax.set_xticklabels([row["Event Label"] for row in rows], rotation=20, ha="right")
    ax.set_title("Runtime Stabilization Timeline", loc="left", fontsize=14)
    ax.set_xlabel("Chronological milestone order")
    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
    ax.set_ylim(-0.9, 0.8)
    ax.grid(False)
    fig.savefig(PLOTS_DIR / "runtime_stabilization_timeline.png", dpi=220, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "runtime_stabilization_timeline.svg", bbox_inches="tight")
    plt.close(fig)


def plot_frontier(all_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [
        row
        for row in all_rows
        if isinstance(row.get("Structure Score"), (int, float))
        and isinstance(row.get("Correctness Score"), (int, float))
        and row["Artifact Status"] != "planned_milestone_missing"
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    color_map = {
        "baseline_pre_refactor": "#7f7f7f",
        "smoke_testmini": "#1f77b4",
        "large_split_continue": "#d62728",
        "phase_d_dedicated": "#9467bd",
    }
    for row in usable:
        size = 60
        if row["Checkpoint Path"] == PRIMARY_FINAL_CHECKPOINT:
            size = 200
        ax.scatter(
            row["Structure Score"],
            row["Correctness Score"],
            s=size,
            c=color_map.get(row["Run Family"], "#333333"),
            alpha=0.8,
            edgecolor="black",
        )
        if row["Checkpoint Path"] == PRIMARY_FINAL_CHECKPOINT or row["Milestone ID"] in {"dedicated_phase_d_best", "smoke_phase_b_best"}:
            ax.annotate(row["X Label"], (row["Structure Score"], row["Correctness Score"]), xytext=(6, 6), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Structure Score")
    ax.set_ylabel("Correctness Score")
    ax.set_title("Checkpoint Frontier: Structure vs Correctness", loc="left", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.savefig(PLOTS_DIR / "checkpoint_frontier_scatter.png", dpi=220, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "checkpoint_frontier_scatter.svg", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(all_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_rows = checkpoint_rows_for_plot(all_rows)
    metrics = [
        ("Exact Match", False),
        ("Tolerance Accuracy", False),
        ("Parseable Rate", False),
        ("Well-Formed Rate", False),
        ("Completion Success Rate", False),
        ("Average Completion Tokens", True),
        ("Composite Score", False),
        ("Correctness Weight", False),
    ]
    matrix: list[list[float]] = []
    row_labels: list[str] = []
    normalized_columns = {}
    for key, invert in metrics:
        normalized_columns[key] = minmax_normalize(
            [row.get(key) if isinstance(row.get(key), (int, float)) else None for row in plot_rows], invert=invert
        )
    for index, row in enumerate(plot_rows):
        matrix.append([
            normalized_columns[key][index] if normalized_columns[key][index] is not None else math.nan for key, _ in metrics
        ])
        row_labels.append(row["X Label"])
    fig, ax = plt.subplots(figsize=(12, max(6, len(row_labels) * 0.4)))
    im = ax.imshow(np.array(matrix), aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([key for key, _ in metrics], rotation=45, ha="right")
    ax.set_title("Checkpoint Heatmap (higher is better after normalization)", loc="left", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.savefig(PLOTS_DIR / "checkpoint_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "checkpoint_heatmap.svg", bbox_inches="tight")
    plt.close(fig)


def plot_lineage() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    nodes = {
        "smoke_c": (0.12, 0.7, "Smoke Phase C best\ncheckpoint-119\nartifact missing locally"),
        "large_c": (0.42, 0.7, "Large Phase C best\ncheckpoint-120\nrecommended final"),
        "large_d": (0.72, 0.7, "Large Phase D best\ncheckpoint-60\nsame notebook"),
        "dedicated_d": (0.72, 0.3, "Dedicated Phase D best\ncheckpoint-130\nmatched Phase C"),
    }
    for x, y, label in nodes.values():
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#3a3a3a"},
        )
    ax.annotate("", xy=(0.33, 0.7), xytext=(0.21, 0.7), arrowprops={"arrowstyle": "->", "linewidth": 2})
    ax.annotate("", xy=(0.63, 0.7), xytext=(0.51, 0.7), arrowprops={"arrowstyle": "->", "linewidth": 2})
    ax.annotate("", xy=(0.63, 0.34), xytext=(0.46, 0.63), arrowprops={"arrowstyle": "->", "linewidth": 2})
    ax.text(0.27, 0.75, "warm-start", ha="center", fontsize=9)
    ax.text(0.57, 0.75, "resume best_composite", ha="center", fontsize=9)
    ax.text(0.55, 0.46, "warm-start\nseparate notebook", ha="center", fontsize=9)
    ax.set_title("Resume / Warm-Start Lineage", loc="left", fontsize=14)
    fig.savefig(PLOTS_DIR / "resume_lineage.png", dpi=220, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "resume_lineage.svg", bbox_inches="tight")
    plt.close(fig)


def write_readme(milestones: list[dict[str, Any]], all_rows: list[dict[str, Any]], resource_rows: list[dict[str, Any]]) -> None:
    recommended = next(row for row in milestones if row["Milestone ID"] == "large_phase_c_best_recommended")
    lines = [
        "# Results Package",
        "",
        "This folder was generated from the current local and Kaggle-exported staged RL artifacts.",
        "",
        "Recommended final checkpoint:",
        f"- `{recommended['Checkpoint Path']}`",
        f"- exact match `{recommended['Exact Match']}`",
        f"- tolerance accuracy `{recommended['Tolerance Accuracy']}`",
        f"- composite score `{recommended['Composite Score']}`",
        "",
        "Generated tables:",
        "- `tables/master_table_milestones.csv`",
        "- `tables/master_table_all_checkpoints.csv`",
        "- `tables/resource_runtime_tuning.csv`",
        "- `tables/resource_knob_tradeoffs.csv`",
        "- `tables/runtime_stabilization_timeline.csv`",
        "",
        "Generated plots:",
        "- `plots/evolution_panels.png`",
        "- `plots/runtime_stabilization_timeline.png`",
        "- `plots/checkpoint_frontier_scatter.png`",
        "- `plots/checkpoint_heatmap.png`",
        "- `plots/resume_lineage.png`",
        "",
        "Regenerate with:",
        "- `.venv-results/bin/python scripts/generate_results_report.py`",
        "",
        "Evidence caveats:",
        "- The larger-split Phase C best composite checkpoint remains the recommended final checkpoint.",
        "- Dedicated Phase D matched the larger-split Phase C best score but did not exceed it.",
        "",
        f"Row counts: {len(all_rows)} checkpoint/audit rows, {len(milestones)} milestone rows, {len(resource_rows)} resource-tuning rows.",
    ]
    (RESULTS_DIR / "README.md").write_text("\n".join(lines) + "\n")


def write_summary(milestones: list[dict[str, Any]]) -> None:
    recommended = next(row for row in milestones if row["Milestone ID"] == "large_phase_c_best_recommended")
    dedicated = next(row for row in milestones if row["Milestone ID"] == "dedicated_phase_d_best")
    summary = [
        "# RL Fine-Tuning Result Summary",
        "",
        f"- Recommended final checkpoint: `{recommended['Checkpoint Path']}`",
        f"- Recommended final metrics: exact `{recommended['Exact Match']}`, tolerance `{recommended['Tolerance Accuracy']}`, parseable `{recommended['Parseable Rate']}`, malformed `{recommended['Malformed Rate']}`, truncation `{recommended['Truncation Rate']}`",
        f"- Dedicated Phase D best matched but did not beat it: exact `{dedicated['Exact Match']}`, composite `{dedicated['Composite Score']}`",
        "- Main causal read: staged RL fixed structure first; the larger split then produced the main correctness gain.",
        "- Resource-constrained read: the larger split became feasible on Kaggle T4 because per-step memory pressure was reduced by the stable `kaggle_t4` profile, not because the problem became smaller.",
    ]
    (RESULTS_DIR / "report_summary.md").write_text("\n".join(summary) + "\n")


def write_data_sources() -> None:
    payload = {
        "repo_root": str(ROOT),
        "baseline_metrics": str(BASELINE_METRICS_PATH),
        "baseline_train_log_summary": str(BASELINE_TRAIN_SUMMARY_PATH),
        "run_sources": [
            {
                "run_family": source["run_family"],
                "notebook_slug": source["notebook_slug"],
                "root": str(source["root"]),
                "phases": source["phases"],
                "exists": source["root"].exists(),
            }
            for source in RUN_SOURCES
        ],
        "notes": [
            "Smoke, larger-split, and dedicated Phase D bundles are present locally and were used directly for table and plot generation.",
            "The larger-split Phase C best composite checkpoint is the recommended final checkpoint in the generated summary outputs.",
        ],
    }
    (RESULTS_DIR / "data_sources.json").write_text(json.dumps(payload, indent=2) + "\n")


def sanitize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = []
    for row in rows:
        clean = {}
        for key, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                clean[key] = ""
            elif isinstance(value, bool):
                clean[key] = value
            else:
                clean[key] = value
        sanitized.append(clean)
    return sanitized


def main() -> None:
    ensure_dirs()

    all_rows = sanitize_rows(build_all_checkpoint_rows())
    milestone_rows = sanitize_rows(build_milestone_rows(all_rows))
    resource_rows = build_resource_rows(milestone_rows)
    knob_rows = build_knob_tradeoff_rows()
    timeline_rows = build_runtime_timeline_rows(milestone_rows)

    write_csv(TABLES_DIR / "master_table_all_checkpoints.csv", all_rows, MASTER_TABLE_COLUMNS)
    write_csv(TABLES_DIR / "master_table_milestones.csv", milestone_rows, MASTER_TABLE_COLUMNS)
    write_csv(TABLES_DIR / "resource_runtime_tuning.csv", resource_rows, list(resource_rows[0].keys()))
    write_csv(TABLES_DIR / "resource_knob_tradeoffs.csv", knob_rows, list(knob_rows[0].keys()))
    write_csv(TABLES_DIR / "runtime_stabilization_timeline.csv", timeline_rows, list(timeline_rows[0].keys()))

    plot_main_evolution(all_rows)
    plot_runtime_timeline(timeline_rows)
    plot_frontier(all_rows)
    plot_heatmap(all_rows)
    plot_lineage()

    write_readme(milestone_rows, all_rows, resource_rows)
    write_summary(milestone_rows)
    write_data_sources()


if __name__ == "__main__":
    main()
