"""Runtime integration with Unsloth and TRL GRPO."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from .checkpointing import CheckpointRegistry, build_resume_plan, write_checkpoint_artifacts
from .config import RunConfig, dataclass_to_dict
from .controller import RewardController
from .data import (
    analyze_dataset_records,
    build_eval_datasets,
    build_phase_train_dataset,
    dataset_to_records,
    load_mathvista_split,
    save_dataset_analysis,
)
from .diagnostics import build_post_training_diagnostics, save_json, summarize_training_logs
from .evaluation import evaluate_checkpoint
from .rewarding import RewardRuntimeContext, build_reward_functions


LOGGER = logging.getLogger(__name__)


def build_component_bounds(phase_config) -> dict[str, tuple[float, float]]:
    """Return min/max bounds for every reward component."""

    return {
        name: (component.min_weight, component.max_weight)
        for name, component in phase_config.reward_components.items()
    }


def build_initial_reward_weights(phase_config) -> dict[str, float]:
    """Return the initial per-component weights for a phase."""

    return {
        name: (component.initial_weight if component.enabled else 0.0)
        for name, component in phase_config.reward_components.items()
    }


def reward_weight_list(reward_funcs, reward_weights: Mapping[str, float]) -> list[float]:
    """Return weights aligned to the reward-function order."""

    return [float(reward_weights.get(func.__name__, 0.0)) for func in reward_funcs]


def apply_reward_weights(trainer, reward_funcs, reward_weights: Mapping[str, float]) -> None:
    """Update the trainer reward tensor in-place."""

    trainer.reward_weights = torch.tensor(
        reward_weight_list(reward_funcs, reward_weights),
        dtype=torch.float32,
        device=trainer.accelerator.device if hasattr(trainer, "accelerator") else None,
    )


def create_model_and_tokenizer(run_config: RunConfig, model_name_or_path: Optional[str] = None):
    """Load the model and tokenizer, then attach LoRA if needed."""

    from unsloth import FastVisionModel  # pylint: disable=import-error

    model_config = run_config.model
    resolved_name = model_name_or_path or model_config.base_model_name
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=resolved_name,
        max_seq_length=model_config.max_seq_length,
        load_in_4bit=model_config.load_in_4bit,
        fast_inference=model_config.fast_inference,
        gpu_memory_utilization=model_config.gpu_memory_utilization,
    )

    if not hasattr(model, "peft_config"):
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=model_config.finetune_vision_layers,
            finetune_language_layers=model_config.finetune_language_layers,
            finetune_attention_modules=model_config.finetune_attention_modules,
            finetune_mlp_modules=model_config.finetune_mlp_modules,
            r=model_config.lora_rank,
            lora_alpha=model_config.lora_alpha,
            bias=model_config.bias,
            random_state=model_config.random_state,
            use_rslora=model_config.use_rslora,
            loftq_config=model_config.loftq_config,
            use_gradient_checkpointing=model_config.use_gradient_checkpointing,
        )
    return model, tokenizer


def build_grpo_args(run_config: RunConfig, phase_config, reward_funcs, output_dir: Path):
    """Create GRPOConfig for the current phase."""

    from trl import GRPOConfig  # pylint: disable=import-error

    defaults = dataclass_to_dict(run_config.trainer_defaults)
    defaults.update(dict(phase_config.trainer_overrides))
    defaults["output_dir"] = str(output_dir)
    defaults["max_completion_length"] = defaults.get("max_completion_length", run_config.trainer_defaults.max_completion_length)
    defaults["reward_weights"] = reward_weight_list(reward_funcs, build_initial_reward_weights(phase_config))
    return GRPOConfig(**defaults)


class MetricAwareGRPOTrainerMixin:
    """Mixin that evaluates and ranks every checkpoint."""

    eval_datasets: Mapping[str, Any]
    reward_runtime: RewardRuntimeContext
    reward_funcs_list: list[Any]
    reward_controller: RewardController
    checkpoint_registry: CheckpointRegistry
    run_config: RunConfig
    phase_name: str
    latest_eval_results: Optional[dict[str, Any]]

    def _metric_aware_save(self, checkpoint_dir: Path) -> None:
        if not checkpoint_dir.exists():
            return

        if hasattr(self.model, "for_inference"):
            self.model.for_inference()

        eval_results = evaluate_checkpoint(
            model=self.model,
            eval_datasets=self.eval_datasets,
            lora_path=str(checkpoint_dir),
            runtime=self.reward_runtime,
            reward_funcs=self.reward_funcs_list,
            reward_weights=self.reward_controller.current_weights(),
            eval_config=self.run_config.eval,
        )

        checkpoint_entry = write_checkpoint_artifacts(
            checkpoint_dir=checkpoint_dir,
            eval_results=eval_results,
            reward_weights=self.reward_controller.current_weights(),
            controller_state=self.reward_controller.to_dict(),
            checkpoint_info={
                "checkpoint_path": str(checkpoint_dir),
                "global_step": self.state.global_step,
                "phase_name": self.phase_name,
                "selector_phase_name": self.phase_name,
            },
            score_config=self.run_config.checkpoint_scores,
        )
        self.checkpoint_registry.register(checkpoint_entry)

        updated_weights = self.reward_controller.update_from_metrics(
            eval_results["metrics"],
            max_completion_length=self.reward_runtime.max_completion_length,
        )
        apply_reward_weights(self, self.reward_funcs_list, updated_weights)
        (checkpoint_dir / "reward_weights.json").write_text(json.dumps(updated_weights, indent=2), encoding="utf-8")
        (checkpoint_dir / "controller_state.json").write_text(
            json.dumps(self.reward_controller.to_dict(), indent=2),
            encoding="utf-8",
        )
        self.latest_eval_results = eval_results

        if hasattr(self.model, "for_training"):
            self.model.for_training(use_gradient_checkpointing=self.run_config.model.use_gradient_checkpointing)


def build_metric_trainer_class(base_cls):
    """Create the concrete trainer subclass without importing TRL at module import time."""

    class MetricAwareGRPOTrainer(MetricAwareGRPOTrainerMixin, base_cls):
        """GRPOTrainer with checkpoint-side evaluation and reward control."""

        def __init__(
            self,
            *args,
            eval_datasets,
            reward_runtime,
            reward_funcs_list,
            reward_controller,
            checkpoint_registry,
            run_config,
            phase_name,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.eval_datasets = eval_datasets
            self.reward_runtime = reward_runtime
            self.reward_funcs_list = reward_funcs_list
            self.reward_controller = reward_controller
            self.checkpoint_registry = checkpoint_registry
            self.run_config = run_config
            self.phase_name = phase_name
            self.latest_eval_results = None

        def _save_checkpoint(self, model, trial):
            super()._save_checkpoint(model, trial)
            checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            LOGGER.info("Running checkpoint evaluation for %s", checkpoint_dir)
            self._metric_aware_save(checkpoint_dir)

    return MetricAwareGRPOTrainer


def _load_controller_state_from_checkpoint(checkpoint_path: Optional[str], current_phase: str) -> Optional[dict[str, Any]]:
    if not checkpoint_path:
        return None
    checkpoint_dir = Path(checkpoint_path)
    info_path = checkpoint_dir / "checkpoint_info.json"
    state_path = checkpoint_dir / "controller_state.json"
    if not info_path.exists() or not state_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("phase_name") != current_phase:
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def run_phase(run_config: RunConfig, phase_name: Optional[str] = None, resume_selector: Optional[str] = None) -> dict[str, Any]:
    """Run one explicit phase of staged RL training."""

    from trl import GRPOTrainer  # pylint: disable=import-error

    resolved_phase = phase_name or run_config.phase_name
    phase_config = run_config.phases[resolved_phase]
    output_dir = run_config.output_dir_for_phase(resolved_phase)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(dataclass_to_dict(run_config), output_dir / "run_config.json")

    search_dirs = [run_config.output_dir_for_phase(name) for name in run_config.phases]
    selector = resume_selector if resume_selector is not None else phase_config.default_resume.selector
    resume_plan = build_resume_plan(
        selector=selector,
        current_phase=resolved_phase,
        current_phase_dir=output_dir,
        search_dirs=search_dirs,
        default_model_name=run_config.model.base_model_name,
    )

    model, tokenizer = create_model_and_tokenizer(run_config, model_name_or_path=resume_plan.model_load_path)
    train_base = load_mathvista_split(run_config, run_config.train_split)
    eval_base = load_mathvista_split(run_config, run_config.eval_split)

    save_dataset_analysis(
        analyze_dataset_records(dataset_to_records(train_base), run_config.stages),
        output_dir / "dataset_analysis_train.json",
    )
    save_dataset_analysis(
        analyze_dataset_records(dataset_to_records(eval_base), run_config.stages),
        output_dir / "dataset_analysis_eval.json",
    )

    train_dataset, stage_datasets = build_phase_train_dataset(train_base, phase_config, run_config.stages, tokenizer)
    eval_datasets = build_eval_datasets(eval_base, run_config, tokenizer)

    reward_runtime = RewardRuntimeContext(
        tokenizer=tokenizer,
        max_completion_length=int(
            phase_config.trainer_overrides.get(
                "max_completion_length",
                run_config.trainer_defaults.max_completion_length,
            )
        ),
        phase_config=phase_config,
    )
    reward_funcs = build_reward_functions(reward_runtime)
    initial_weights = build_initial_reward_weights(phase_config)
    controller_state = _load_controller_state_from_checkpoint(resume_plan.trainer_resume_path, resolved_phase)
    reward_controller = RewardController.from_state(
        gate_config=run_config.reward_gate,
        component_bounds=build_component_bounds(phase_config),
        initial_weights=initial_weights,
        state_dict=controller_state,
    )

    args = build_grpo_args(run_config, phase_config, reward_funcs, output_dir)
    MetricAwareGRPOTrainer = build_metric_trainer_class(GRPOTrainer)
    trainer = MetricAwareGRPOTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_datasets=eval_datasets,
        reward_runtime=reward_runtime,
        reward_funcs_list=reward_funcs,
        reward_controller=reward_controller,
        checkpoint_registry=CheckpointRegistry(output_dir),
        run_config=run_config,
        phase_name=resolved_phase,
    )
    apply_reward_weights(trainer, reward_funcs, reward_controller.current_weights())

    train_result = trainer.train(resume_from_checkpoint=resume_plan.trainer_resume_path)

    final_lora_dir = output_dir / "final_lora"
    model.save_lora(str(final_lora_dir))

    log_summary = summarize_training_logs(trainer.state.log_history)
    save_json(log_summary, output_dir / "train_log_summary.json")
    registry = CheckpointRegistry(output_dir)
    diagnostics = build_post_training_diagnostics(registry.data, trainer.latest_eval_results or {})
    save_json(diagnostics, output_dir / "post_training_diagnostics.json")

    return {
        "train_result": train_result,
        "output_dir": str(output_dir),
        "final_lora_dir": str(final_lora_dir),
        "resume_plan": dataclass_to_dict(resume_plan),
        "stage_names": list(stage_datasets.keys()),
        "latest_eval_results": trainer.latest_eval_results,
        "train_log_summary": log_summary,
        "post_training_diagnostics": diagnostics,
    }
