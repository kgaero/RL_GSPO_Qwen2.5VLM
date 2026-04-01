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

from staged_rl.config import RewardGateConfig, apply_hardware_profile, build_default_run_config
from staged_rl.controller import RewardController

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

MASTER_COLUMN_GROUPS = [
    (
        "Identity and Selection",
        [
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
        ],
    ),
    (
        "Training Setup",
        [
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
        ],
    ),
    (
        "Reward Configuration",
        [
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
        ],
    ),
    (
        "Evaluation Metrics",
        [
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
        ],
    ),
    (
        "Checkpoint Selection Diagnostics",
        [
            "Checkpoint Rank Within Phase",
            "Is Best Composite",
            "Is Best Correctness",
            "Is Best Structure",
            "Is Latest",
            "Controller History Length",
            "Delta Exact vs Previous Milestone",
            "Delta Composite vs Previous Milestone",
        ],
    ),
    (
        "Evidence Paths",
        [
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
        ],
    ),
]

MASTER_COLUMN_DESCRIPTIONS = {
    "Milestone ID": "Stable short name used to refer to a key row in the report.",
    "Run Family": "Internal run bucket showing which notebook family produced the row.",
    "Run Family Label": "Human-friendly label for the run family.",
    "Notebook / Run Slug": "Kaggle or local run identifier that produced the artifacts.",
    "Output Root": "Top-level output folder where this run wrote checkpoints and reports.",
    "Phase": "Training phase name used by the staged RL pipeline.",
    "Phase Label": "Human-friendly phase name used in tables and plots.",
    "Checkpoint": "Checkpoint directory name.",
    "Global Step": "Trainer step number at which the checkpoint was saved.",
    "Timeline Order": "Chronological order used in the report figures.",
    "X Label": "Short checkpoint label shown on the x-axis of plots.",
    "Artifact Status": "Whether this row came from a real checkpoint, alias metadata, or a baseline snapshot.",
    "Metrics Source Kind": "Which file type supplied the metrics for the row.",
    "Alias Role": "Primary alias assigned to the checkpoint, such as best composite or latest.",
    "Alias Roles": "All alias labels that point to the checkpoint.",
    "Keep / Discard": "Report-level recommendation for whether to keep this checkpoint for future use.",
    "Decision Reason": "Short explanation for the keep/discard recommendation.",
    "Key Interpretation": "Plain-language take on what the row means.",
    "Train Split": "Dataset split used for training in this run.",
    "Eval Split": "Dataset split used for checkpoint evaluation.",
    "Hardware Profile": "Named runtime profile applied to fit the hardware budget.",
    "Seed / Resume Source": "Resume selector used to continue from a previous checkpoint alias.",
    "Warm-Start Checkpoint": "Explicit checkpoint path loaded before training started.",
    "Training Strategy Introduced": "Short summary of what this phase was trying to teach the model.",
    "Stage Mix / Curriculum": "Curriculum subset mix used in the phase.",
    "Phase Description": "Longer phase description from the run config.",
    "Phase Default Resume Selector": "Default alias that this phase expects to resume from.",
    "Base Model": "Base pretrained model used before LoRA adaptation.",
    "4-bit Enabled": "Whether 4-bit loading was used to reduce memory.",
    "Fast Inference Enabled": "Whether the fast generation path was enabled during RL.",
    "Fast Inference Mode": "Short name for the inference backend or mode.",
    "Compilation Mode": "vLLM/Unsloth compilation setting used for stable generation.",
    "max_seq_length": "Maximum total sequence length allowed by the runtime profile.",
    "image_size": "Target image resolution used for visual inputs.",
    "LoRA Rank": "Adapter rank used for LoRA fine-tuning.",
    "Max LoRA Rank": "Upper LoRA rank bound passed to the runtime when supported.",
    "gradient_accumulation_steps": "How many micro-batches were accumulated before an optimizer step.",
    "gpu_memory_utilization": "Target GPU memory fraction reserved for the fast generation backend.",
    "num_generations": "How many completions were sampled per prompt during RL.",
    "max_prompt_length": "Maximum prompt token budget.",
    "max_completion_length": "Maximum completion token budget.",
    "max_eval_examples_per_subset": "Maximum checkpoint-eval examples per subset for that run.",
    "Configured Initial Correctness Weight": "Starting correctness reward weight from the phase config.",
    "Configured Initial Formatting Weight": "Starting formatting reward weight from the phase config.",
    "Configured Initial Parseability Weight": "Starting parseability reward weight from the phase config.",
    "Configured Initial Finished Weight": "Starting finished-answer reward weight from the phase config.",
    "Configured Initial Tolerance Weight": "Starting tolerance reward weight from the phase config.",
    "Configured Initial Brevity Weight": "Starting brevity reward weight from the phase config.",
    "Correctness Weight": "Effective correctness reward weight saved with the checkpoint.",
    "Formatting Weight": "Effective formatting reward weight saved with the checkpoint.",
    "Parseability Weight": "Effective parseability reward weight saved with the checkpoint.",
    "Finished Weight": "Effective completion-finished reward weight saved with the checkpoint.",
    "Tolerance Weight": "Effective tolerance reward weight saved with the checkpoint.",
    "Brevity Weight": "Effective brevity reward weight saved with the checkpoint.",
    "Correctness Reward Mean": "Mean correctness reward observed during eval for this checkpoint.",
    "Format Reward Mean": "Mean formatting reward observed during eval for this checkpoint.",
    "Parseable Reward Mean": "Mean parseability reward observed during eval for this checkpoint.",
    "Finished Reward Mean": "Mean finished-answer reward observed during eval for this checkpoint.",
    "Tolerance Reward Mean": "Mean tolerance reward observed during eval for this checkpoint.",
    "Exact Match": "Fraction of prompts whose final answer matched exactly.",
    "Tolerance Accuracy": "Fraction of prompts counted correct under numeric tolerance.",
    "Best-of-k Accuracy": "Best completion accuracy across the sampled completions per prompt.",
    "Best-of-k Tolerance Accuracy": "Best completion tolerance accuracy across sampled completions.",
    "Sample-Level Exact Match": "Exact-match rate measured at the sampled-completion level rather than prompt level.",
    "Sample-Level Tolerance Accuracy": "Tolerance accuracy measured at the sampled-completion level.",
    "Parseable Rate": "Fraction of outputs whose answer could be parsed successfully.",
    "Reasoning Tag Compliance": "Fraction of outputs that included the required reasoning tags correctly.",
    "Solution Tag Compliance": "Fraction of outputs that included the required solution tags correctly.",
    "Well-Formed Rate": "Positive-form version of malformed rate: higher means fewer malformed outputs.",
    "Malformed Rate": "Fraction of outputs with malformed structure or tags.",
    "Completion Success Rate": "Positive-form version of truncation rate: higher means fewer truncated outputs.",
    "Truncation Rate": "Fraction of outputs that ended before producing a usable final answer.",
    "Average Completion Tokens": "Mean output length in completion tokens.",
    "Repetition Rate": "Simple repetition score; higher means the model repeated itself more.",
    "Sample Diversity": "How varied the sampled answers were across multiple generations.",
    "Average Total Reward": "Mean total reward for the evaluated completions.",
    "Structure Score": "Composite structure-oriented checkpoint score used in selection.",
    "Correctness Score": "Composite correctness-oriented checkpoint score used in selection.",
    "Composite Score": "Main combined checkpoint score used for best-composite selection.",
    "KL Mean": "Mean KL divergence logged during training for that phase.",
    "KL P95": "95th percentile KL divergence logged during training for that phase.",
    "Checkpoint Rank Within Phase": "Checkpoint order within its phase after sorting by global step.",
    "Is Best Composite": "Whether the checkpoint is tagged as best composite for its phase.",
    "Is Best Correctness": "Whether the checkpoint is tagged as best correctness for its phase.",
    "Is Best Structure": "Whether the checkpoint is tagged as best structure for its phase.",
    "Is Latest": "Whether the checkpoint is tagged as the latest saved checkpoint for its phase.",
    "Controller History Length": "How many eval events were stored in the reward-controller history.",
    "Delta Exact vs Previous Milestone": "Change in exact match compared with the previous milestone row.",
    "Delta Composite vs Previous Milestone": "Change in composite score compared with the previous milestone row.",
    "Checkpoint Path": "Relative checkpoint path recorded in the saved artifacts.",
    "Checkpoint Path Abs": "Absolute local filesystem path to the checkpoint directory or source file.",
    "Evidence Root": "Root folder containing the evidence for the row.",
    "Metrics Source Path": "File path that directly supplied the row metrics.",
    "Checkpoint Info Path": "Path to the saved checkpoint metadata JSON.",
    "Reward Weights Path": "Path to the saved effective reward weights JSON.",
    "Controller State Path": "Path to the saved reward-controller state JSON.",
    "Run Config Path": "Path to the saved run configuration JSON.",
    "Run Request Path": "Path to the saved CLI request JSON for the run.",
    "Diagnostics Path": "Path to the saved post-training diagnostics JSON for the phase.",
    "Train Log Summary Path": "Path to the saved training-log summary JSON for the phase.",
    "Summary Path": "Path to the short human-readable checkpoint summary file.",
    "Notes": "Extra caveats or context that did not fit cleanly into the other columns.",
}

RESOURCE_RUNTIME_COLUMN_DESCRIPTIONS = {
    "Run Family": "Run family or reference profile represented by the row.",
    "Notebook / Run Slug": "Notebook or run identifier tied to the row.",
    "Hardware": "Hardware used, or reference hardware label for comparison rows.",
    "Observed Constraint / Failure Mode": "Short statement of the memory or runtime constraint being addressed.",
    "Train Split": "Training split used by that run.",
    "Eval Split": "Evaluation split used by that run.",
    "Hardware Profile": "Named runtime profile applied to the run.",
    "Base Model": "Base pretrained model behind the run.",
    "4-bit Enabled": "Whether 4-bit loading was enabled.",
    "LoRA Rank": "LoRA rank used in that profile.",
    "Max LoRA Rank": "Maximum LoRA rank setting if present.",
    "max_seq_length": "Maximum sequence length allowed by the profile.",
    "image_size": "Image resolution used in the profile.",
    "num_generations": "Number of sampled completions per prompt during RL.",
    "max_prompt_length": "Maximum prompt token budget.",
    "max_completion_length": "Maximum completion token budget.",
    "gradient_accumulation_steps": "Gradient accumulation count used during training.",
    "gpu_memory_utilization": "Target vLLM/fast-generation memory reservation fraction.",
    "fast_inference enabled": "Whether the fast generation path remained enabled.",
    "vLLM version": "vLLM version if logged; otherwise not logged.",
    "cudagraph / compilation mode": "Compilation mode used to keep fast generation stable.",
    "Warm start used?": "Whether the run started from an earlier checkpoint rather than from scratch.",
    "What knob changed": "Short summary of the main profile differences versus the reference profile.",
    "Why this helps memory": "Plain-language reason the profile choices reduce memory pressure.",
    "Tradeoff introduced": "What capability or convenience was sacrificed to stay within the VRAM budget.",
    "Outcome": "What the row achieved under that profile.",
}

RESOURCE_KNOB_COLUMN_DESCRIPTIONS = {
    "Knob": "The runtime or training setting that changed.",
    "Default": "Reference value from the default higher-capacity profile.",
    "Kaggle T4": "Value used in the stable Kaggle T4 profile.",
    "Why it reduced memory pressure": "Simple reason this change lowered VRAM demand or stabilized runtime.",
    "Tradeoff accepted": "What was sacrificed in exchange for the lower memory demand.",
}

TIMELINE_COLUMN_DESCRIPTIONS = {
    "Timeline Order": "Chronological order used in the timeline figure and table.",
    "Event Label": "Short label for the milestone event.",
    "Milestone ID": "Stable internal milestone key.",
    "Run Family": "Run family that produced the milestone.",
    "Phase": "Training phase tied to the milestone.",
    "Train Split": "Training split used at that milestone.",
    "Eval Split": "Eval split used at that milestone.",
    "Warm Start": "Resume alias or explicit warm-start checkpoint used before training.",
    "Status": "Short status label describing what changed at that milestone.",
    "Best Exact": "Best exact-match value represented by the milestone.",
    "Best Composite": "Best composite score represented by the milestone.",
    "Key Change": "Short description of what the training setup changed at that point.",
    "Interpretation": "Plain-language read of why the milestone matters.",
}

CONTROLLER_AUDIT_COLUMNS = [
    "Timeline Order",
    "X Label",
    "Run Family",
    "Run Family Label",
    "Phase",
    "Phase Label",
    "Checkpoint",
    "Global Step",
    "Phase Reset",
    "Parseable Guard Fired",
    "Format Guard Fired",
    "Finish Guard Fired",
    "Correctness Rule Fired",
    "Stable Structure",
    "Stable Window Ready",
    "Correctness Plateau",
    "Triggered Rules",
    "Changed Components",
    "Clamped Components",
    "Exact Previous",
    "Exact Current",
    "Exact Delta",
    "Parseable Rate",
    "Solution Tag Compliance",
    "Reasoning Tag Compliance",
    "Malformed Rate",
    "Truncation Rate",
    "Average Completion Tokens",
    "Average Token Fraction",
    "Max Completion Length",
    "Correctness Before",
    "Correctness After",
    "Correctness Delta",
    "Formatting Before",
    "Formatting After",
    "Formatting Delta",
    "Parseability Before",
    "Parseability After",
    "Parseability Delta",
    "Finished Before",
    "Finished After",
    "Finished Delta",
    "Controller Decision Path",
    "Controller Decision Source",
    "Controller Decision Match",
]

CONTROLLER_AUDIT_COLUMN_DESCRIPTIONS = {
    "Timeline Order": "Chronological order used in the main checkpoint plots.",
    "X Label": "Short checkpoint label used in plots.",
    "Run Family": "Internal run family for the checkpoint.",
    "Run Family Label": "Human-friendly run family label.",
    "Phase": "Training phase for the checkpoint.",
    "Phase Label": "Human-friendly phase label.",
    "Checkpoint": "Checkpoint directory name.",
    "Global Step": "Trainer global step at save time.",
    "Phase Reset": "Whether this row is the first controller update in its phase run.",
    "Parseable Guard Fired": "Whether the low-parseability guard condition fired.",
    "Format Guard Fired": "Whether the tag/malformed formatting guard condition fired.",
    "Finish Guard Fired": "Whether the truncation or overlength guard condition fired.",
    "Correctness Rule Fired": "Whether correctness weight escalation fired after a stable plateau.",
    "Stable Structure": "Whether all structure stability thresholds were satisfied.",
    "Stable Window Ready": "Whether enough checkpoint history existed to test the correctness plateau rule.",
    "Correctness Plateau": "Whether exact-match gain stayed below the plateau threshold over the stable window.",
    "Triggered Rules": "Semicolon-separated list of controller rules that fired on this checkpoint.",
    "Changed Components": "Semicolon-separated list of reward weights that actually changed.",
    "Clamped Components": "Semicolon-separated list of weights clipped by min/max bounds.",
    "Exact Previous": "Previous exact-match value used by the plateau rule.",
    "Exact Current": "Current exact-match value seen by the controller.",
    "Exact Delta": "Current exact minus previous exact used by the plateau rule.",
    "Parseable Rate": "Current parseable-answer rate seen by the controller.",
    "Solution Tag Compliance": "Current solution-tag compliance seen by the controller.",
    "Reasoning Tag Compliance": "Current reasoning-tag compliance seen by the controller.",
    "Malformed Rate": "Current malformed-answer rate seen by the controller.",
    "Truncation Rate": "Current truncation rate seen by the controller.",
    "Average Completion Tokens": "Current average completion length seen by the controller.",
    "Average Token Fraction": "Average completion tokens divided by max completion length.",
    "Max Completion Length": "Max completion length used for the controller decision.",
    "Correctness Before": "Correctness weight before the controller update.",
    "Correctness After": "Correctness weight after the controller update.",
    "Correctness Delta": "Change applied to correctness weight on this checkpoint.",
    "Formatting Before": "Formatting weight before the controller update.",
    "Formatting After": "Formatting weight after the controller update.",
    "Formatting Delta": "Change applied to formatting weight on this checkpoint.",
    "Parseability Before": "Parseability weight before the controller update.",
    "Parseability After": "Parseability weight after the controller update.",
    "Parseability Delta": "Change applied to parseability weight on this checkpoint.",
    "Finished Before": "Finished-answer weight before the controller update.",
    "Finished After": "Finished-answer weight after the controller update.",
    "Finished Delta": "Change applied to finished-answer weight on this checkpoint.",
    "Controller Decision Path": "Path to the saved per-checkpoint controller decision artifact when present.",
    "Controller Decision Source": "Whether the audit row came from a saved artifact or report-side reconstruction.",
    "Controller Decision Match": "Whether the reconstructed post-update weights matched the saved checkpoint weights.",
}


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def json_load(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text())


def coerce_csv_scalar(value: str) -> Any:
    text = str(value).strip()
    if text == "":
        return ""
    if text == "True":
        return True
    if text == "False":
        return False
    try:
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        return float(text)
    except ValueError:
        return value


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                key: coerce_csv_scalar(value)
                for key, value in row.items()
            }
            for row in reader
        ]


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


def _default_run_config_dict(phase_name: str) -> dict[str, Any]:
    return asdict(build_default_run_config(phase_name))


def _build_controller_from_run_config(run_config_data: dict[str, Any], phase_name: str) -> RewardController:
    phases = run_config_data.get("phases", {})
    phase_cfg = phases.get(phase_name) or _default_run_config_dict(phase_name)["phases"][phase_name]
    reward_gate_data = run_config_data.get("reward_gate") or _default_run_config_dict(phase_name)["reward_gate"]
    reward_components = phase_cfg.get("reward_components", {})
    component_bounds = {
        name: (float(component["min_weight"]), float(component["max_weight"]))
        for name, component in reward_components.items()
    }
    initial_weights = {
        name: float(component["initial_weight"]) if component.get("enabled", True) else 0.0
        for name, component in reward_components.items()
    }
    return RewardController(
        gate_config=RewardGateConfig(**reward_gate_data),
        component_bounds=component_bounds,
        initial_weights=initial_weights,
    )


def _controller_metric_payload(row: dict[str, Any]) -> dict[str, float]:
    return {
        "parseable_answer_rate": float(row.get("Parseable Rate") or 0.0),
        "solution_tag_compliance": float(row.get("Solution Tag Compliance") or 0.0),
        "reasoning_tag_compliance": float(row.get("Reasoning Tag Compliance") or 0.0),
        "malformed_answer_rate": float(row.get("Malformed Rate") or 0.0),
        "truncation_rate": float(row.get("Truncation Rate") or 0.0),
        "average_completion_tokens": float(row.get("Average Completion Tokens") or 0.0),
        "normalized_exact_match": float(row.get("Exact Match") or 0.0),
    }


def _checkpoint_weight_snapshot(row: dict[str, Any]) -> dict[str, float]:
    return {
        "correctness_reward": float(row.get("Correctness Weight") or 0.0),
        "format_reward": float(row.get("Formatting Weight") or 0.0),
        "parseable_reward": float(row.get("Parseability Weight") or 0.0),
        "finished_reward": float(row.get("Finished Weight") or 0.0),
        "tolerance_reward": float(row.get("Tolerance Weight") or 0.0),
        "brevity_reward": float(row.get("Brevity Weight") or 0.0),
    }


def _weights_match(left: dict[str, Any], right: dict[str, Any], tolerance: float = 1e-9) -> bool:
    if not left or not right:
        return False
    for name, value in right.items():
        if name not in left:
            return False
        if not math.isclose(float(left[name]), float(value), abs_tol=tolerance, rel_tol=tolerance):
            return False
    return True


def build_controller_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit_rows = []
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("Artifact Status") != "actual_checkpoint" or row.get("Phase") == "baseline":
            continue
        grouped_rows[(row["Run Family"], row["Phase"])].append(row)

    for (_, phase_name), phase_rows in grouped_rows.items():
        phase_rows.sort(key=lambda item: int(item["Global Step"]))
        run_config_data = json_load(Path(phase_rows[0]["Run Config Path"])) or _default_run_config_dict(phase_name)
        controller = _build_controller_from_run_config(run_config_data, phase_name)

        for row in phase_rows:
            max_completion_length = int(row.get("max_completion_length") or 0)
            controller.update_from_metrics(
                _controller_metric_payload(row),
                max_completion_length=max_completion_length,
            )
            reconstructed_decision = controller.latest_decision()
            decision_path = Path(row["Checkpoint Path Abs"]) / "controller_decision.json"
            saved_decision = json_load(decision_path)
            decision = (
                saved_decision
                if saved_decision.get("post_update_weights") and saved_decision.get("weight_deltas")
                else reconstructed_decision
            )
            decision_source = "artifact" if decision is saved_decision else "reconstructed"
            weights = decision.get("weight_deltas", {})
            rule_status = decision.get("rule_status", {})
            checkpoint_weights = _checkpoint_weight_snapshot(row)
            post_update_weights = decision.get("post_update_weights", {})
            triggered_rules = [event["rule_key"] for event in decision.get("rule_events", [])]
            changed_components = decision.get("changed_components", [])
            clamped_components = [
                item["component"] if isinstance(item, dict) else str(item)
                for item in decision.get("clamped_components", [])
            ]

            audit_rows.append(
                {
                    "Timeline Order": row["Timeline Order"],
                    "X Label": row["X Label"],
                    "Run Family": row["Run Family"],
                    "Run Family Label": row["Run Family Label"],
                    "Phase": row["Phase"],
                    "Phase Label": row["Phase Label"],
                    "Checkpoint": row["Checkpoint"],
                    "Global Step": row["Global Step"],
                    "Phase Reset": bool(decision.get("history_length_before", 0) == 0),
                    "Parseable Guard Fired": bool(rule_status.get("parseable_guard", False)),
                    "Format Guard Fired": bool(rule_status.get("format_guard", False)),
                    "Finish Guard Fired": bool(rule_status.get("finish_guard", False)),
                    "Correctness Rule Fired": bool(rule_status.get("correctness_escalation", False)),
                    "Stable Structure": bool(rule_status.get("stable_structure", False)),
                    "Stable Window Ready": bool(rule_status.get("stable_window_ready", False)),
                    "Correctness Plateau": bool(rule_status.get("correctness_plateau", False)),
                    "Triggered Rules": ";".join(triggered_rules),
                    "Changed Components": ";".join(changed_components),
                    "Clamped Components": ";".join(clamped_components),
                    "Exact Previous": decision.get("exact_previous", ""),
                    "Exact Current": decision.get("exact_current", ""),
                    "Exact Delta": decision.get("exact_delta", ""),
                    "Parseable Rate": row.get("Parseable Rate", ""),
                    "Solution Tag Compliance": row.get("Solution Tag Compliance", ""),
                    "Reasoning Tag Compliance": row.get("Reasoning Tag Compliance", ""),
                    "Malformed Rate": row.get("Malformed Rate", ""),
                    "Truncation Rate": row.get("Truncation Rate", ""),
                    "Average Completion Tokens": row.get("Average Completion Tokens", ""),
                    "Average Token Fraction": decision.get("avg_token_fraction", ""),
                    "Max Completion Length": max_completion_length,
                    "Correctness Before": weights.get("correctness_reward", {}).get("before", ""),
                    "Correctness After": weights.get("correctness_reward", {}).get("after", ""),
                    "Correctness Delta": weights.get("correctness_reward", {}).get("delta", ""),
                    "Formatting Before": weights.get("format_reward", {}).get("before", ""),
                    "Formatting After": weights.get("format_reward", {}).get("after", ""),
                    "Formatting Delta": weights.get("format_reward", {}).get("delta", ""),
                    "Parseability Before": weights.get("parseable_reward", {}).get("before", ""),
                    "Parseability After": weights.get("parseable_reward", {}).get("after", ""),
                    "Parseability Delta": weights.get("parseable_reward", {}).get("delta", ""),
                    "Finished Before": weights.get("finished_reward", {}).get("before", ""),
                    "Finished After": weights.get("finished_reward", {}).get("after", ""),
                    "Finished Delta": weights.get("finished_reward", {}).get("delta", ""),
                    "Controller Decision Path": maybe_path(decision_path),
                    "Controller Decision Source": decision_source,
                    "Controller Decision Match": _weights_match(post_update_weights, checkpoint_weights),
                }
            )

    audit_rows.sort(key=lambda item: int(item["Timeline Order"]))
    return audit_rows


def analysis_for_controller_audit(rows: list[dict[str, Any]]) -> list[str]:
    parse_hits = sum(bool(row["Parseable Guard Fired"]) for row in rows)
    format_hits = sum(bool(row["Format Guard Fired"]) for row in rows)
    finish_hits = sum(bool(row["Finish Guard Fired"]) for row in rows)
    correctness_hits = sum(bool(row["Correctness Rule Fired"]) for row in rows)
    resets = sum(bool(row["Phase Reset"]) for row in rows)
    regressive_correctness_hits = sum(
        bool(row["Correctness Rule Fired"]) and isinstance(row.get("Exact Delta"), (int, float)) and float(row["Exact Delta"]) < 0.0
        for row in rows
    )
    return [
        f"The controller audit covers {len(rows)} evaluated checkpoints across {resets} phase resets.",
        f"Parseability guard fired {parse_hits} times, format guard fired {format_hits} times, and finish guard fired {finish_hits} times.",
        f"Correctness escalation fired {correctness_hits} times after structure was already stable.",
        f"{regressive_correctness_hits} correctness escalations happened on checkpoints where exact match had actually regressed, which makes the plateau rule's behavior explicit.",
        "The audit rows distinguish phase-default resets from controller-triggered updates, which the weight-evolution plot alone does not show.",
    ]


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


def render_column_glossary(groups: list[tuple[str, list[str]]], descriptions: dict[str, str]) -> str:
    sections = []
    for heading, columns in groups:
        sections.append(f"### {heading}\n")
        sections.append("| Column | Simple meaning |")
        sections.append("| --- | --- |")
        for column in columns:
            sections.append(f"| `{column}` | {descriptions.get(column, column.replace('_', ' '))} |")
        sections.append("")
    return "\n".join(sections)


def write_markdown_doc(
    *,
    markdown_path: Path,
    title: str,
    csv_name: str,
    purpose: str,
    glossary_markdown: str,
    analysis_lines: list[str],
) -> None:
    lines = [
        f"# {title}",
        "",
        f"CSV source: `{csv_name}`",
        "",
        "## Purpose",
        "",
        purpose,
        "",
        "## Column Glossary",
        "",
        glossary_markdown.rstrip(),
        "",
        "## Analysis",
        "",
    ]
    for line in analysis_lines:
        lines.append(f"- {line}")
    lines.append("")
    markdown_path.write_text("\n".join(lines))


def format_float(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{value:.3f}".rstrip("0").rstrip(".")


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


def run_family_plot_code(run_family: str) -> str:
    return {
        "baseline_pre_refactor": "base",
        "smoke_testmini": "smk",
        "large_split_continue": "lgc",
        "phase_d_dedicated": "dpd",
    }.get(run_family, "run")


def compact_phase_header(label: str, span: int) -> str:
    if label == "Baseline":
        return "Base"
    if label.startswith("Smoke Phase "):
        suffix = label.removeprefix("Smoke Phase ")
        return f"Smoke {suffix}" if span <= 3 else label
    if label.startswith("Large Phase "):
        suffix = label.removeprefix("Large Phase ")
        return f"Large {suffix}" if span <= 3 else label
    if label == "Dedicated Phase D":
        return "Dedicated D" if span <= 4 else label
    return label


KEY_CURRICULUM_MILESTONE_IDS = [
    "baseline_local_snapshot",
    "smoke_phase_a_best",
    "smoke_phase_b_best",
    "smoke_phase_c_best",
    "smoke_phase_d_best",
    "large_phase_c_best_recommended",
    "large_phase_d_same_notebook_best",
    "dedicated_phase_d_best",
]

PHASE_CURRICULUM_MILESTONE_IDS = KEY_CURRICULUM_MILESTONE_IDS[1:]

STAGE_ORDER = [
    "stage1_easy_numeric",
    "stage2_float_numeric",
    "stage3_hard_numeric",
]

STAGE_DISPLAY = {
    "stage1_easy_numeric": "Stage 1\nEasy Numeric",
    "stage2_float_numeric": "Stage 2\nMedium / Float",
    "stage3_hard_numeric": "Stage 3\nHard Numeric",
}

STAGE_SHORT = {
    "stage1_easy_numeric": "S1",
    "stage2_float_numeric": "S2",
    "stage3_hard_numeric": "S3",
}

STAGE_COLORS = {
    "stage1_easy_numeric": "#4c78a8",
    "stage2_float_numeric": "#f58518",
    "stage3_hard_numeric": "#e45756",
}

RUN_FAMILY_COLORS = {
    "baseline_pre_refactor": "#7f7f7f",
    "smoke_testmini": "#4c78a8",
    "large_split_continue": "#e45756",
    "phase_d_dedicated": "#72b7b2",
}

SPLIT_DISPLAY = {
    "unknown_pre_refactor": "unknown",
    "testmini": "testmini",
    "test": "test",
}


def key_curriculum_rows(milestone_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {row["Milestone ID"]: row for row in milestone_rows}
    return [lookup[milestone_id] for milestone_id in KEY_CURRICULUM_MILESTONE_IDS if milestone_id in lookup]


def phase_curriculum_rows(milestone_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {row["Milestone ID"]: row for row in milestone_rows}
    return [lookup[milestone_id] for milestone_id in PHASE_CURRICULUM_MILESTONE_IDS if milestone_id in lookup]


def parse_stage_mix_text(stage_mix_text: str) -> dict[str, float]:
    text = (stage_mix_text or "").strip()
    if not text or "baseline" in text:
        return {}
    result: dict[str, float] = {}
    for part in text.split(";"):
        chunk = part.strip()
        if not chunk or "=" not in chunk:
            continue
        name, value = chunk.split("=", 1)
        try:
            result[name.strip()] = float(value.strip())
        except ValueError:
            continue
    return result


def compact_stage_mix_text(stage_mix_text: str) -> str:
    mix = parse_stage_mix_text(stage_mix_text)
    if not mix:
        return "no staged curriculum"
    return ", ".join(f"{STAGE_SHORT.get(stage_name, stage_name)} {weight:.0%}" for stage_name, weight in mix.items())


def stage_mix_vector(stage_mix_text: str) -> list[float]:
    mix = parse_stage_mix_text(stage_mix_text)
    return [float(mix.get(stage_name, 0.0)) for stage_name in STAGE_ORDER]


def save_plot_dual(fig: Any, stem: str) -> None:
    fig.savefig(PLOTS_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / f"{stem}.svg", bbox_inches="tight")


def plot_main_evolution(
    rows: list[dict[str, Any]],
    *,
    stem: str = "evolution_panels",
    include_notebook_panel: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_rows = checkpoint_rows_for_plot(rows)
    x = [row["Timeline Order"] for row in plot_rows]
    plot_labels = []
    for row in plot_rows:
        if row["Phase"] == "baseline":
            plot_labels.append("base")
            continue
        phase_short = row["Phase"].replace("phase_", "")
        plot_labels.append(f"{run_family_plot_code(row['Run Family'])}:{phase_short}:{row['Global Step']}")

    x_tick_labels = list(plot_labels)
    notebook_order = list(dict.fromkeys(row["Notebook / Run Slug"] for row in plot_rows))
    notebook_to_y = {label: index for index, label in enumerate(notebook_order)}
    notebook_y = [notebook_to_y[row["Notebook / Run Slug"]] for row in plot_rows]
    plot_row_lookup = {
        (row["Run Family"], row["Phase"], row["Checkpoint"]): row
        for row in plot_rows
    }
    phase_start_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in plot_rows:
        phase_start_rows.setdefault((row["Run Family"], row["Phase"]), row)

    split_to_y = {"unknown_pre_refactor": 0, "testmini": 1, "test": 2}
    stage_series_lookup = {
        "Stage 1 (Easy Numeric)": [],
        "Stage 2 (Medium/Float)": [],
        "Stage 3 (Hard Numeric)": [],
    }
    for row in plot_rows:
        if row["Phase"] == "baseline":
            mix_values = [math.nan, math.nan, math.nan]
        else:
            mix_values = stage_mix_vector(row["Stage Mix / Curriculum"])
        stage_series_lookup["Stage 1 (Easy Numeric)"].append(mix_values[0])
        stage_series_lookup["Stage 2 (Medium/Float)"].append(mix_values[1])
        stage_series_lookup["Stage 3 (Hard Numeric)"].append(mix_values[2])

    audit_lookup = {row["X Label"]: row for row in build_controller_audit_rows(rows)}
    rule_heatmap_specs = [
        ("Phase Reset", "reset"),
        ("Parseable Guard Fired", "parse"),
        ("Format Guard Fired", "format"),
        ("Finish Guard Fired", "finish"),
        ("Stable Structure", "stable"),
        ("Stable Window Ready", "window"),
        ("Correctness Plateau", "plateau"),
        ("Correctness Rule Fired", "correct"),
    ]
    rule_marker_specs = {
        "Phase Reset": {"marker": "P", "color": "#bcbd22", "size": 110},
        "Parseable Guard Fired": {"marker": "s", "color": "#1f77b4", "size": 95},
        "Format Guard Fired": {"marker": "D", "color": "#2ca02c", "size": 95},
        "Finish Guard Fired": {"marker": "^", "color": "#9467bd", "size": 95},
        "Stable Structure": {"marker": "o", "color": "#7f7f7f", "size": 90},
        "Stable Window Ready": {"marker": "v", "color": "#17becf", "size": 95},
        "Correctness Plateau": {"marker": "X", "color": "#ff7f0e", "size": 100},
        "Correctness Rule Fired": {"marker": "*", "color": "#d62728", "size": 160},
    }
    delta_heatmap_specs = [
        ("Correctness Delta", "Correctness Weight"),
        ("Parseability Delta", "Parseability Weight"),
        ("Finished Delta", "Finished Weight"),
        ("Formatting Delta", "Formatting Weight"),
    ]
    reset_value_specs = [
        ("Configured Initial Correctness Weight", "Correctness Weight"),
        ("Configured Initial Parseability Weight", "Parseability Weight"),
        ("Configured Initial Finished Weight", "Finished Weight"),
        ("Configured Initial Tolerance Weight", "Tolerance Weight"),
        ("Configured Initial Brevity Weight", "Brevity Weight"),
        ("Configured Initial Formatting Weight", "Formatting Weight"),
    ]
    delta_marker_style = {
        "positive_fill": "#cfe8dc",
        "negative_fill": "#f4dddd",
        "edge": "#6f6f6f",
        "boxstyle": "square,pad=0.18",
        "alpha": 0.98,
        "fontsize": 6.8,
    }
    reset_marker_style = {
        "fill": "#dbe9f6",
        "edge": "#6f6f6f",
        "boxstyle": "square,pad=0.18",
        "alpha": 0.98,
        "fontsize": 6.8,
    }
    height_ratios = [1.0, 1.0, 0.9, 1.0, 1.0, 0.95, 0.9, 0.95, 1.25, 0.75, 0.85, 0.9, 0.65]
    if include_notebook_panel:
        height_ratios.append(0.95)

    def wrap_axis_label(label: str, width: int = 18) -> str:
        wrap_ready = label.replace("/", "/ ").replace("_", "_ ")
        wrapped = textwrap.fill(wrap_ready, width=width, break_long_words=False, break_on_hyphens=True)
        return wrapped.replace("/ ", "/").replace("_ ", "_")

    def extract_dataset_slug(path_text: str) -> str:
        path_value = str(path_text or "").strip()
        if not path_value:
            return ""
        parts = Path(path_value).parts
        if "datasets" in parts:
            dataset_index = parts.index("datasets")
            if dataset_index + 2 < len(parts):
                return parts[dataset_index + 2]
        return Path(path_value).parent.name

    def dataset_node_label(warm_start_path: str) -> str:
        slug = extract_dataset_slug(warm_start_path)
        short_slug = slug.removeprefix("rl-gspo-qwen2-5vlm-")
        if short_slug.endswith("-checkpoint"):
            short_slug = short_slug[: -len("-checkpoint")]
        checkpoint_name = Path(str(warm_start_path)).name or "checkpoint"
        display_slug = short_slug or slug or "attached-checkpoint"
        return "\n".join(
            [
                "Kaggle dataset",
                wrap_axis_label(display_slug, width=18),
                checkpoint_name,
            ]
        )

    notebook_dataset_transfers: list[dict[str, Any]] = []
    for source_key, target_key in [
        (("smoke_testmini", "phase_c", "checkpoint-119"), ("large_split_continue", "phase_c")),
        (("large_split_continue", "phase_c", "checkpoint-120"), ("phase_d_dedicated", "phase_d")),
    ]:
        source_row = plot_row_lookup.get(source_key)
        target_row = phase_start_rows.get(target_key)
        if source_row is None or target_row is None:
            continue
        warm_start_path = str(target_row.get("Warm-Start Checkpoint") or "").strip()
        if not warm_start_path:
            continue
        notebook_dataset_transfers.append(
            {
                "source_row": source_row,
                "target_row": target_row,
                "warm_start_path": warm_start_path,
                "node_label": dataset_node_label(warm_start_path),
            }
        )

    fig, axes = plt.subplots(
        14 if include_notebook_panel else 13,
        1,
        figsize=(22, 46 if include_notebook_panel else 43),
        sharex=True,
        gridspec_kw={
            "height_ratios": height_ratios,
            "hspace": 0.10,
        },
    )
    fig.subplots_adjust(right=0.84)

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

    def add_segments(ax: Any, label_position: str | None = None) -> None:
        for index, (start, end, label) in enumerate(segments):
            ax.axvspan(start - 0.5, end + 0.5, color=("#f7f7f7" if index % 2 == 0 else "#eef3f9"), alpha=0.7, zorder=0)
            ax.axvline(end + 0.5, color="#bbbbbb", linestyle="--", linewidth=0.7)
            if label_position:
                span = end - start + 1
                display_label = compact_phase_header(label, span)
                if label_position == "top":
                    y_offset = 1.085
                    va = "bottom"
                else:
                    y_offset = -0.98
                    va = "top"
                ax.text(
                    (start + end) / 2,
                    y_offset,
                    display_label,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va=va,
                    fontsize=8.5,
                    linespacing=0.95,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.6},
                    clip_on=False,
                )

    def extract_series(key: str) -> list[float | None]:
        if key in stage_series_lookup:
            return stage_series_lookup[key]
        return [row.get(key) if isinstance(row.get(key), (int, float)) else None for row in plot_rows]

    panel_specs = [
        (
            axes[0],
            "Exact Match",
            [
                ("Exact Match", "#d62728"),
            ],
            False,
            "score",
        ),
        (
            axes[1],
            "Reasoning Tag Compliance / Solution Tag Compliance / Truncation Rate",
            [
                ("Reasoning Tag Compliance", "#1f77b4"),
                ("Solution Tag Compliance", "#2ca02c"),
                ("Truncation Rate", "#d62728"),
            ],
            False,
            "rate",
        ),
        (
            axes[2],
            "KL Mean / KL P95",
            [
                ("KL Mean", "#9467bd"),
                ("KL P95", "#ff7f0e"),
            ],
            False,
            "kl",
        ),
        (
            axes[3],
            "Generation Length / Reward",
            [
                ("Average Completion Tokens", "#d62728"),
                ("Average Total Reward", "#9467bd"),
            ],
            False,
            "",
        ),
        (
            axes[4],
            "Generation Diversity / Repetition",
            [
                ("Sample Diversity", "#2ca02c"),
                ("Repetition Rate", "#1f77b4"),
            ],
            False,
            "rate",
        ),
        (
            axes[5],
            "Rule Triggers",
            [],
            False,
            "rule",
        ),
        (
            axes[6],
            "Weight Deltas - controller applied",
            [],
            False,
            "delta",
        ),
        (
            axes[7],
            "Weight Reset - phase initial default",
            [],
            False,
            "reset",
        ),
        (
            axes[8],
            "Reward Weight Evolution",
            [
                ("Correctness Weight", "#d62728"),
                ("Parseability Weight", "#2ca02c"),
                ("Finished Weight", "#9467bd"),
                ("Tolerance Weight", "#ff7f0e"),
                ("Brevity Weight", "#8c564b"),
            ],
            True,
            "weight",
        ),
        (
            axes[9],
            "Reward Weight Evolution continued",
            [
                ("Formatting Weight", "#1f77b4"),
            ],
            True,
            "weight",
        ),
        (
            axes[10],
            "Checkpointwise Split Transition",
            [],
            False,
            "split",
        ),
        (
            axes[11],
            "Checkpointwise Stage Relationship / Mix",
            [
                ("Stage 1 (Easy Numeric)", STAGE_COLORS["stage1_easy_numeric"]),
                ("Stage 2 (Medium/Float)", STAGE_COLORS["stage2_float_numeric"]),
                ("Stage 3 (Hard Numeric)", STAGE_COLORS["stage3_hard_numeric"]),
            ],
            True,
            "mix",
        ),
    ]

    for index, (ax, title, series_specs, step_mode, y_label) in enumerate(panel_specs):
        add_segments(ax, label_position="top" if index == 0 else None)
        if title == "Generation Length / Reward":
            left_key, left_color = series_specs[0]
            right_key, right_color = series_specs[1]
            left_values = extract_series(left_key)
            right_values = extract_series(right_key)
            ax.plot(x, left_values, marker="o", label=left_key, color=left_color, linewidth=2)
            ax.set_ylabel("tokens", color=left_color)
            ax.tick_params(axis="y", labelcolor=left_color)
            right_ax = ax.twinx()
            right_ax.plot(x, right_values, marker="o", label=right_key, color=right_color, linewidth=2)
            right_ax.set_ylabel("reward", color=right_color)
            right_ax.tick_params(axis="y", labelcolor=right_color)
            right_ax.grid(False)
            legend_items = [
                Line2D([0], [0], color=left_color, marker="o", linewidth=2, label=left_key),
                Line2D([0], [0], color=right_color, marker="o", linewidth=2, label=right_key),
            ]
            ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0.0)
        elif title == "Rule Triggers":
            ax.set_ylabel("rule")
            ax.set_yticks(range(len(rule_heatmap_specs)))
            ax.set_yticklabels([label for _, label in rule_heatmap_specs], fontsize=8)
            ax.set_ylim(-0.5, len(rule_heatmap_specs) - 0.5)
            ax.invert_yaxis()
            for row in plot_rows:
                audit_row = audit_lookup.get(row["X Label"])
                if audit_row is None:
                    continue
                x_pos = row["Timeline Order"]
                for trigger_index, (column_name, _) in enumerate(rule_heatmap_specs):
                    if bool(audit_row.get(column_name)):
                        marker_cfg = rule_marker_specs[column_name]
                        ax.scatter(
                            x_pos,
                            trigger_index,
                            marker=marker_cfg["marker"],
                            color=marker_cfg["color"],
                            s=marker_cfg["size"],
                            zorder=3,
                            edgecolor="black",
                            linewidth=0.5,
                        )
            rule_legend_items = [
                Line2D(
                    [0],
                    [0],
                    marker=rule_marker_specs[column_name]["marker"],
                    color="w",
                    label=label,
                    markerfacecolor=rule_marker_specs[column_name]["color"],
                    markeredgecolor="black",
                    markersize=9 if rule_marker_specs[column_name]["size"] < 140 else 12,
                )
                for column_name, label in rule_heatmap_specs
            ]
            ax.legend(
                handles=rule_legend_items,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                fontsize=7.5,
                borderaxespad=0.0,
            )
        elif title == "Weight Deltas - controller applied":
            ax.set_ylabel("")
            ax.set_yticks(range(len(delta_heatmap_specs)))
            ax.set_yticklabels([label for _, label in delta_heatmap_specs], fontsize=8)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.tick_params(axis="y", labelright=True, labelleft=False, pad=8)
            ax.set_ylim(-0.5, len(delta_heatmap_specs) - 0.5)
            ax.invert_yaxis()
            for delta_index in range(len(delta_heatmap_specs)):
                ax.axhline(delta_index, color="#d7d7d7", linewidth=0.8, zorder=1)
            for row in plot_rows:
                audit_row = audit_lookup.get(row["X Label"])
                if audit_row is None:
                    continue
                x_pos = row["Timeline Order"]
                for delta_index, (column_name, _) in enumerate(delta_heatmap_specs):
                    value = audit_row.get(column_name)
                    if isinstance(value, (int, float)) and abs(float(value)) > 1e-9:
                        marker_color = delta_marker_style["positive_fill"] if float(value) > 0 else delta_marker_style["negative_fill"]
                        label = f"{float(value):+.2f}".rstrip("0").rstrip(".")
                        ax.text(
                            x_pos,
                            delta_index,
                            label,
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=delta_marker_style["fontsize"],
                            zorder=4,
                            bbox={
                                "boxstyle": delta_marker_style["boxstyle"],
                                "facecolor": marker_color,
                                "edgecolor": delta_marker_style["edge"],
                                "linewidth": 0.9,
                                "alpha": delta_marker_style["alpha"],
                            },
                        )
        elif title == "Weight Reset - phase initial default":
            ax.set_ylabel("")
            ax.set_yticks(range(len(reset_value_specs)))
            ax.set_yticklabels([label for _, label in reset_value_specs], fontsize=8)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.tick_params(axis="y", labelright=True, labelleft=False, pad=8)
            ax.set_ylim(-0.5, len(reset_value_specs) - 0.5)
            ax.invert_yaxis()
            for reset_index in range(len(reset_value_specs)):
                ax.axhline(reset_index, color="#d7d7d7", linewidth=0.8, zorder=1)
            for row in plot_rows:
                audit_row = audit_lookup.get(row["X Label"])
                if audit_row is None or not bool(audit_row.get("Phase Reset")):
                    continue
                x_pos = row["Timeline Order"]
                for reset_index, (column_name, _) in enumerate(reset_value_specs):
                    value = row.get(column_name)
                    if isinstance(value, (int, float)):
                        label = f"{float(value):.2f}".rstrip("0").rstrip(".")
                        ax.text(
                            x_pos,
                            reset_index,
                            label,
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=reset_marker_style["fontsize"],
                            zorder=4,
                            bbox={
                                "boxstyle": reset_marker_style["boxstyle"],
                                "facecolor": reset_marker_style["fill"],
                                "edgecolor": reset_marker_style["edge"],
                                "linewidth": 0.9,
                                "alpha": reset_marker_style["alpha"],
                            },
                        )
        elif title == "Checkpointwise Split Transition":
            train_values = [split_to_y.get(str(row["Train Split"]), math.nan) for row in plot_rows]
            eval_values = [split_to_y.get(str(row["Eval Split"]), math.nan) for row in plot_rows]
            ax.step(x, train_values, where="mid", label="train split", color="#1f77b4", linewidth=2.6)
            ax.plot(x, train_values, marker="o", color="#1f77b4", linewidth=0, markersize=4.8)
            ax.step(x, eval_values, where="mid", label="eval split", color="#ff7f0e", linewidth=2.0, linestyle="--")
            ax.plot(
                x,
                eval_values,
                marker="o",
                color="#ff7f0e",
                linewidth=0,
                markersize=4.8,
                markerfacecolor="white",
            )
            ax.set_ylabel("split")
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(["unknown", "testmini", "test"])
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0.0)
        else:
            for key, color in series_specs:
                values = extract_series(key)
                if step_mode:
                    ax.step(
                        x,
                        [math.nan if value is None else value for value in values],
                        where="mid",
                        label=key,
                        color=color,
                        linewidth=2,
                    )
                    ax.scatter(x, values, color=color, s=25, zorder=3)
                else:
                    ax.plot(x, values, marker="o", label=key, color=color, linewidth=2)
            ax.set_ylabel(y_label)
            if series_specs:
                ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0.0)
        title_y = 1.005 if index == 0 else 1.0
        ax.set_title(title, fontsize=12, loc="left", y=title_y)
        ax.grid(True, axis="y", alpha=0.25)
        if title in {"Reward Weight Evolution", "Reward Weight Evolution continued"}:
            ax.set_ylim(0.0, 8.01)
        elif title == "Checkpointwise Split Transition":
            ax.set_ylim(-0.2, 2.2)
        elif title == "Exact Match":
            ax.set_ylim(0.0, 0.8)
        elif title in {
            "Reasoning Tag Compliance / Solution Tag Compliance / Truncation Rate",
            "Generation Diversity / Repetition",
            "Checkpointwise Stage Relationship / Mix",
        }:
            ax.set_ylim(-0.05, 1.05)
        elif y_label == "rate":
            ax.set_ylim(-0.05, 1.05)

    decision_ax = axes[12]
    add_segments(decision_ax, label_position=None if include_notebook_panel else "bottom")
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
    alias_offsets = {
        "best_structure": -0.18,
        "best_correctness": -0.06,
        "best_composite": 0.06,
        "latest": 0.18,
        "none": 0.0,
    }
    for row in plot_rows:
        alias_roles = sorted(filter(None, row.get("Alias Roles", "").split(";"))) or ["none"]
        for alias in alias_roles:
            decision_ax.scatter(
                row["Timeline Order"] + alias_offsets.get(alias, 0.0),
                y_map.get(row["Keep / Discard"], 0.0),
                marker=marker_map.get(alias, "o"),
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
    if include_notebook_panel:
        decision_ax.tick_params(axis="x", labelbottom=False)
    else:
        decision_ax.set_xticks(x)
        decision_ax.set_xticklabels(x_tick_labels, rotation=60, ha="right", fontsize=8)
    alias_legend_items = [
        Line2D([0], [0], marker="*", color="w", label="best_composite", markerfacecolor="black", markeredgecolor="black", markersize=12),
        Line2D([0], [0], marker="D", color="w", label="best_correctness", markerfacecolor="black", markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="s", color="w", label="best_structure", markerfacecolor="black", markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="latest", markerfacecolor="black", markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="P", color="w", label="no alias", markerfacecolor="black", markeredgecolor="black", markersize=9),
    ]
    decision_legend_items = [
        Line2D([0], [0], marker="o", color="w", label="keep_primary", markerfacecolor="#2ca02c", markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="keep_secondary", markerfacecolor="#1f77b4", markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="reference", markerfacecolor="#7f7f7f", markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="discard", markerfacecolor="#d62728", markeredgecolor="black", markersize=9),
    ]
    alias_legend = decision_ax.legend(
        handles=alias_legend_items,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        borderaxespad=0.0,
    )
    decision_ax.add_artist(alias_legend)
    decision_ax.legend(
        handles=decision_legend_items,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.52),
        fontsize=8,
        borderaxespad=0.0,
    )

    if include_notebook_panel:
        notebook_ax = axes[13]
        add_segments(notebook_ax, label_position="bottom")
        for notebook_index in range(len(notebook_order)):
            notebook_ax.axhline(notebook_index, color="#d7d7d7", linewidth=0.8, zorder=1)
        notebook_ax.step(x, notebook_y, where="mid", color="#6f6f6f", linewidth=1.8, zorder=2)
        notebook_ax.scatter(
            x,
            notebook_y,
            c=[RUN_FAMILY_COLORS.get(row["Run Family"], "#333333") for row in plot_rows],
            s=58,
            zorder=3,
            edgecolor="black",
            linewidth=0.5,
        )
        notebook_ax.set_ylabel("")
        notebook_ax.set_yticks(range(len(notebook_order)))
        notebook_ax.set_yticklabels([wrap_axis_label(label, width=18) for label in notebook_order], fontsize=7)
        notebook_ax.yaxis.tick_right()
        notebook_ax.yaxis.set_label_position("right")
        notebook_ax.tick_params(axis="y", labelright=True, labelleft=False, pad=8)
        for tick_label in notebook_ax.get_yticklabels():
            tick_label.set_horizontalalignment("left")
            tick_label.set_linespacing(0.9)
        notebook_ax.set_ylim(-0.5, len(notebook_order) - 0.5)
        notebook_ax.invert_yaxis()
        notebook_ax.set_title("Notebook Evolution", fontsize=12, loc="left")
        notebook_ax.grid(False, axis="y")
        notebook_ax.set_xticks(x)
        notebook_ax.set_xticklabels(x_tick_labels, rotation=60, ha="right", fontsize=8)
        dataset_arrow_color = "#8c7a00"
        dataset_node_fill = "#fff3bf"
        dataset_node_edge = "#5c5200"
        for transfer in notebook_dataset_transfers:
            source_row = transfer["source_row"]
            target_row = transfer["target_row"]
            source_x = source_row["Timeline Order"]
            source_y = notebook_to_y[source_row["Notebook / Run Slug"]]
            target_x = target_row["Timeline Order"]
            target_y = notebook_to_y[target_row["Notebook / Run Slug"]]
            node_x = target_x - 0.75
            if node_x <= source_x + 0.45:
                node_x = source_x + (target_x - source_x) * 0.55
            node_y = (source_y + target_y) / 2.0
            curve_mag = 0.18 if (target_x - source_x) > 6 else 0.08
            source_arrow = notebook_ax.annotate(
                "",
                xy=(node_x, node_y),
                xytext=(source_x, source_y),
                arrowprops={
                    "arrowstyle": "->",
                    "linewidth": 1.4,
                    "linestyle": (0, (4, 2)),
                    "color": dataset_arrow_color,
                    "shrinkA": 6,
                    "shrinkB": 6,
                    "connectionstyle": f"arc3,rad={-curve_mag}",
                    "alpha": 0.95,
                },
            )
            source_arrow.set_zorder(4)
            target_arrow = notebook_ax.annotate(
                "",
                xy=(target_x, target_y),
                xytext=(node_x, node_y),
                arrowprops={
                    "arrowstyle": "->",
                    "linewidth": 1.4,
                    "linestyle": (0, (4, 2)),
                    "color": dataset_arrow_color,
                    "shrinkA": 6,
                    "shrinkB": 6,
                    "connectionstyle": f"arc3,rad={curve_mag / 2}",
                    "alpha": 0.95,
                },
            )
            target_arrow.set_zorder(4)
            notebook_ax.scatter(
                [node_x],
                [node_y],
                marker="D",
                s=85,
                facecolor=dataset_node_fill,
                edgecolor=dataset_node_edge,
                linewidth=1.0,
                zorder=5,
            )
            label_x = node_x
            label_y = max(-0.15, min(source_y, target_y) - 0.55)
            node_annotation = notebook_ax.annotate(
                transfer["node_label"],
                xy=(node_x, node_y),
                xytext=(label_x, label_y),
                textcoords="data",
                ha="center",
                va="center",
                fontsize=6.8,
                linespacing=0.95,
                bbox={
                    "boxstyle": "round,pad=0.24",
                    "facecolor": "white",
                    "edgecolor": dataset_node_edge,
                    "linewidth": 0.9,
                    "alpha": 0.97,
                },
            )
            node_annotation.set_zorder(6)

    fig.suptitle("RL Fine-Tuning Evolution Across Smoke, Large-Split, and Dedicated Phase D Runs", fontsize=16)
    fig.text(
        0.5,
        0.985,
        "Eval coverage per subset is tiny (2 for smoke runs, 4 for larger runs), so many metrics are strongly quantized.",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    save_plot_dual(fig, stem)
    plt.close(fig)


def build_resource_rows(milestones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    default_cfg = asdict(build_default_run_config("phase_a"))
    default_trainer = dict(default_cfg["trainer_defaults"])
    default_model = dict(default_cfg["model"])

    def row_for_milestone(milestone_id: str) -> dict[str, Any] | None:
        for row in milestones:
            if row["Milestone ID"] == milestone_id:
                return row
        return None

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
        row = milestone_by_id.get(milestone_id)
        if row is None:
            continue
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


def analysis_for_milestones(rows: list[dict[str, Any]]) -> list[str]:
    lookup = {row["Milestone ID"]: row for row in rows}
    baseline = lookup["baseline_local_snapshot"]
    smoke_a = lookup["smoke_phase_a_best"]
    smoke_b = lookup["smoke_phase_b_best"]
    smoke_c = lookup["smoke_phase_c_best"]
    smoke_d = lookup["smoke_phase_d_best"]
    large_c = lookup["large_phase_c_best_recommended"]
    large_d_latest = lookup["large_phase_d_same_notebook_latest"]
    dedicated_d = lookup["dedicated_phase_d_best"]
    return [
        f"The baseline row starts at exact match {format_float(baseline['Exact Match'])}, parseable rate {format_float(baseline['Parseable Rate'])}, malformed rate {format_float(baseline['Malformed Rate'])}, and truncation rate {format_float(baseline['Truncation Rate'])}.",
        f"Smoke Phase A moved exact match to {format_float(smoke_a['Exact Match'])} but still had weak structure, with parseable rate {format_float(smoke_a['Parseable Rate'])} and truncation rate {format_float(smoke_a['Truncation Rate'])}.",
        f"Smoke Phase B fixed structure on the tiny split: parseable rate reached {format_float(smoke_b['Parseable Rate'])}, malformed and truncation both dropped to {format_float(smoke_b['Malformed Rate'])}, but exact match stayed at {format_float(smoke_b['Exact Match'])}.",
        f"Smoke Phase C kept that strong structure profile but still sat at exact match {format_float(smoke_c['Exact Match'])}, which shows the small split had likely hit a correctness ceiling.",
        f"Smoke Phase D did not improve over Smoke Phase C; exact match stayed at {format_float(smoke_d['Exact Match'])} while structure regressed back to parseable rate {format_float(smoke_d['Parseable Rate'])} and truncation rate {format_float(smoke_d['Truncation Rate'])}.",
        f"Larger-split Phase C is the key jump: exact match rose to {format_float(large_c['Exact Match'])}, composite score to {format_float(large_c['Composite Score'])}, while structure stayed perfect.",
        f"The same-notebook Phase D branch shows why checkpoint-aware selection matters: its latest checkpoint fell to exact match {format_float(large_d_latest['Exact Match'])} even though the branch had reached {format_float(lookup['large_phase_d_same_notebook_best']['Exact Match'])} earlier.",
        f"The dedicated Phase D rerun recovered to exact match {format_float(dedicated_d['Exact Match'])}, matching but not beating the recommended larger-split Phase C checkpoint.",
    ]


def analysis_for_all_checkpoints(rows: list[dict[str, Any]]) -> list[str]:
    artifact_rows = [row for row in rows if row["Run Family"] != "baseline_pre_refactor"]
    large_c_rows = [row for row in rows if row["Run Family"] == "large_split_continue" and row["Phase"] == "phase_c"]
    large_d_rows = [row for row in rows if row["Run Family"] == "large_split_continue" and row["Phase"] == "phase_d"]
    dedicated_d_rows = [row for row in rows if row["Run Family"] == "phase_d_dedicated" and row["Phase"] == "phase_d"]
    alias_rows = [row for row in artifact_rows if row["Alias Roles"]]
    best_vs_latest_different = [
        row["Phase Label"]
        for row in artifact_rows
        if row["Is Best Composite"] and not row["Is Latest"]
    ]
    large_c_exact = [float(row["Exact Match"]) for row in large_c_rows if isinstance(row.get("Exact Match"), (int, float))]
    return [
        f"The table contains {len(rows)} rows in total, including {len(artifact_rows)} artifact-backed checkpoint rows and one baseline reference row.",
        f"{len(alias_rows)} rows carry at least one alias label, which makes it possible to compare best-structure, best-correctness, best-composite, and latest checkpoints directly.",
        f"The larger-split Phase C trajectory is explicitly non-monotonic: exact match ranges from {format_float(min(large_c_exact))} to {format_float(max(large_c_exact))} across checkpoints 60 to 602.",
        f"In the larger-split Phase C run, best composite landed at checkpoint-120 while latest was checkpoint-602; both ended strong, but the mid-run oscillation shows why phase-level alias selection still matters.",
        f"In the larger-split Phase D same-notebook branch, best composite was checkpoint-60 while latest was checkpoint-130, confirming that later checkpoints could regress.",
        f"The dedicated Phase D branch is short but informative: checkpoint-120 was weak at exact match {format_float(dedicated_d_rows[1]['Exact Match']) if len(dedicated_d_rows) > 1 else 'n/a'}, and checkpoint-130 recovered to {format_float(dedicated_d_rows[-1]['Exact Match'])}.",
        f"The smoke rows show the same structure-first pattern: Phase B and Phase C remained at exact match 0.5 while structure stayed perfect, then the larger split moved correctness upward.",
    ]


def analysis_for_resource_runtime(rows: list[dict[str, Any]]) -> list[str]:
    reference = rows[0]
    smoke = rows[1]
    large_c = rows[2]
    same_notebook_d = rows[3]
    dedicated_d = rows[4]
    return [
        f"The reference row shows the high-capacity default profile, while every successful Kaggle run used the much smaller `kaggle_t4` profile instead.",
        f"The stable Kaggle T4 profile cut sequence length from {reference['max_seq_length']} to {smoke['max_seq_length']}, image size from {reference['image_size']} to {smoke['image_size']}, generations from {reference['num_generations']} to {smoke['num_generations']}, and LoRA rank from {reference['LoRA Rank']} to {smoke['LoRA Rank']}.",
        f"The smoke Phase A row shows that this lower-memory profile was sufficient to validate the full RL loop on `testmini`.",
        f"The large Phase C row shows the key resource-constrained result: the train split increased from `{smoke['Train Split']}` to `{large_c['Train Split']}` while the core T4 profile stayed the same, which supports the interpretation that split size mainly increased runtime rather than per-step VRAM.",
        f"The same-notebook and dedicated Phase D rows show that the same T4-safe profile could support specialization runs as well, although the dedicated rerun was needed to preserve earlier outputs and recover the better final score.",
    ]


def analysis_for_knob_tradeoffs(rows: list[dict[str, Any]]) -> list[str]:
    lookup = {row["Knob"]: row for row in rows}
    seq_default = float(lookup["max_seq_length"]["Default"])
    seq_t4 = float(lookup["max_seq_length"]["Kaggle T4"])
    prompt_default = float(lookup["max_prompt_length"]["Default"])
    prompt_t4 = float(lookup["max_prompt_length"]["Kaggle T4"])
    completion_default = float(lookup["max_completion_length"]["Default"])
    completion_t4 = float(lookup["max_completion_length"]["Kaggle T4"])
    image_default = float(lookup["image_size"]["Default"])
    image_t4 = float(lookup["image_size"]["Kaggle T4"])
    area_reduction = 1.0 - (image_t4 * image_t4) / (image_default * image_default)
    return [
        f"The biggest headline reduction was sequence length: {int(seq_default)} to {int(seq_t4)}, which is about a {format_float((1 - seq_t4 / seq_default) * 100)}% cut.",
        f"Prompt budget dropped from {int(prompt_default)} to {int(prompt_t4)} and completion budget from {int(completion_default)} to {int(completion_t4)}, which traded long-form reasoning space for lower generation-time memory use.",
        f"Image size fell from {int(image_default)} to {int(image_t4)}; in area terms that is roughly a {format_float(area_reduction * 100)}% reduction in pixel workload for the vision stack.",
        f"LoRA rank and number of generations were both halved, which directly reduced trainable state and multi-sample generation cost.",
        f"The table makes the tradeoff explicit: almost every memory-saving change came with a loss in context budget, visual detail, or exploration diversity.",
    ]


def analysis_for_timeline(rows: list[dict[str, Any]]) -> list[str]:
    labels = [row["Event Label"] for row in rows]
    return [
        f"The timeline contains {len(rows)} milestone events: {' -> '.join(labels)}.",
        "It shows a clear order of improvement: baseline weakness, smoke-run validation, smoke-run structure stabilization, then larger-split correctness improvement.",
        "The smoke Phase D milestone marks the point where the small split stopped helping, which is why the next successful move was to scale the data rather than keep iterating on the same tiny split.",
        "Large Phase C is the first milestone where correctness clearly improved beyond the smoke plateau, and both later Phase D branches should be read as specialization branches rather than primary replacements for that checkpoint.",
    ]


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


def plot_controller_rule_heatmap(audit_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not audit_rows:
        return

    plot_rows = sorted(audit_rows, key=lambda item: int(item["Timeline Order"]))
    row_labels = [row["X Label"] for row in plot_rows]
    rule_columns = [
        ("Phase Reset", "reset"),
        ("Parseable Guard Fired", "parse"),
        ("Format Guard Fired", "format"),
        ("Finish Guard Fired", "finish"),
        ("Stable Structure", "stable"),
        ("Stable Window Ready", "window"),
        ("Correctness Plateau", "plateau"),
        ("Correctness Rule Fired", "correct"),
    ]
    delta_columns = [
        ("Parseability Delta", "d_parse"),
        ("Formatting Delta", "d_fmt"),
        ("Finished Delta", "d_fin"),
        ("Correctness Delta", "d_corr"),
    ]

    rule_matrix = np.array(
        [[1.0 if bool(row.get(column)) else 0.0 for column, _ in rule_columns] for row in plot_rows],
        dtype=float,
    )
    delta_matrix = np.array(
        [[float(row.get(column) or 0.0) for column, _ in delta_columns] for row in plot_rows],
        dtype=float,
    )

    fig, (rule_ax, delta_ax) = plt.subplots(
        1,
        2,
        figsize=(14, max(6, len(plot_rows) * 0.32)),
        gridspec_kw={"width_ratios": [len(rule_columns), len(delta_columns)]},
    )
    rule_im = rule_ax.imshow(rule_matrix, aspect="auto", cmap="Greys", vmin=0.0, vmax=1.0)
    delta_limit = max(0.5, float(np.nanmax(np.abs(delta_matrix))) if delta_matrix.size else 0.5)
    delta_im = delta_ax.imshow(delta_matrix, aspect="auto", cmap="RdYlGn", vmin=-delta_limit, vmax=delta_limit)

    rule_ax.set_yticks(range(len(plot_rows)))
    rule_ax.set_yticklabels(row_labels, fontsize=8)
    delta_ax.set_yticks(range(len(plot_rows)))
    delta_ax.set_yticklabels([])
    rule_ax.set_xticks(range(len(rule_columns)))
    rule_ax.set_xticklabels([label for _, label in rule_columns], rotation=45, ha="right")
    delta_ax.set_xticks(range(len(delta_columns)))
    delta_ax.set_xticklabels([label for _, label in delta_columns], rotation=45, ha="right")
    rule_ax.set_title("Rule Triggers", loc="left", fontsize=12)
    delta_ax.set_title("Weight Deltas", loc="left", fontsize=12)

    for row_index, row in enumerate(plot_rows):
        for column_index, (column_name, _) in enumerate(rule_columns):
            if bool(row.get(column_name)):
                rule_ax.text(column_index, row_index, "x", ha="center", va="center", color="#ff7f0e", fontsize=8)
        for column_index, (column_name, _) in enumerate(delta_columns):
            value = float(row.get(column_name) or 0.0)
            if abs(value) > 1e-9:
                delta_ax.text(column_index, row_index, f"{value:+.2f}", ha="center", va="center", color="black", fontsize=7)

    fig.colorbar(rule_im, ax=rule_ax, fraction=0.046, pad=0.03)
    fig.colorbar(delta_im, ax=delta_ax, fraction=0.046, pad=0.03)
    fig.suptitle("Controller Rule Audit Heatmap", fontsize=14)
    save_plot_dual(fig, "controller_rule_heatmap")
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


def draw_curriculum_map_ax(
    ax: Any,
    milestone_rows: list[dict[str, Any]],
    *,
    title: str = "Curriculum Map",
    title_size: int = 14,
    show_note: bool = True,
) -> None:
    rows = key_curriculum_rows(milestone_rows)
    lookup = {row["Milestone ID"]: row for row in rows}
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    positions = {
        "baseline_local_snapshot": (0.08, 0.58),
        "smoke_phase_a_best": (0.22, 0.58),
        "smoke_phase_b_best": (0.36, 0.58),
        "smoke_phase_c_best": (0.50, 0.58),
        "smoke_phase_d_best": (0.50, 0.24),
        "large_phase_c_best_recommended": (0.68, 0.58),
        "large_phase_d_same_notebook_best": (0.86, 0.76),
        "dedicated_phase_d_best": (0.86, 0.38),
    }
    edges = [
        ("baseline_local_snapshot", "smoke_phase_a_best", "staged RL starts"),
        ("smoke_phase_a_best", "smoke_phase_b_best", "structure improves"),
        ("smoke_phase_b_best", "smoke_phase_c_best", "precision mix"),
        ("smoke_phase_c_best", "smoke_phase_d_best", "small-split Stage 3 branch"),
        ("smoke_phase_c_best", "large_phase_c_best_recommended", "scale up + warm-start ckpt-119"),
        ("large_phase_c_best_recommended", "large_phase_d_same_notebook_best", "same notebook Phase D"),
        ("large_phase_c_best_recommended", "dedicated_phase_d_best", "new notebook + warm-start ckpt-120"),
    ]

    for source_id, target_id, label in edges:
        source_x, source_y = positions[source_id]
        target_x, target_y = positions[target_id]
        ax.annotate(
            "",
            xy=(target_x - 0.055, target_y),
            xytext=(source_x + 0.055, source_y),
            arrowprops={"arrowstyle": "->", "linewidth": 1.8, "color": "#666666"},
        )
        mid_x = (source_x + target_x) / 2
        mid_y = (source_y + target_y) / 2 + (0.04 if target_y >= source_y else -0.05)
        ax.text(mid_x, mid_y, label, ha="center", va="center", fontsize=8.5, color="#555555")

    for row in rows:
        x_pos, y_pos = positions[row["Milestone ID"]]
        is_primary = row["Milestone ID"] == "large_phase_c_best_recommended"
        train_eval = f"{SPLIT_DISPLAY.get(row['Train Split'], row['Train Split'])} -> {SPLIT_DISPLAY.get(row['Eval Split'], row['Eval Split'])}"
        stage_text = compact_stage_mix_text(row["Stage Mix / Curriculum"])
        if row["Phase"] == "baseline":
            stage_text = "pre-staged baseline"
        node_text = "\n".join(
            [
                row["Phase Label"],
                train_eval,
                stage_text,
                f"exact {format_float(row['Exact Match'])} | comp {format_float(row['Composite Score'])}",
            ]
        )
        ax.text(
            x_pos,
            y_pos,
            node_text,
            ha="center",
            va="center",
            fontsize=8.8,
            color="#111111",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": RUN_FAMILY_COLORS.get(row["Run Family"], "#dddddd"),
                "edgecolor": "#222222",
                "alpha": 0.14 if not is_primary else 0.22,
                "linewidth": 2.2 if is_primary else 1.2,
            },
        )
        if is_primary:
            ax.text(x_pos, y_pos + 0.15, "recommended final", ha="center", va="bottom", fontsize=8.5, color="#b42318")

    ax.set_title(title, loc="left", fontsize=title_size)
    if show_note:
        ax.text(
            0.0,
            -0.08,
            "Main causal story: smoke phases fixed structure on testmini, then larger-split Phase C on test delivered the main correctness gain.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#555555",
        )


def plot_curriculum_map(milestone_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(18, 7))
    draw_curriculum_map_ax(ax, milestone_rows)
    save_plot_dual(fig, "curriculum_map")
    plt.close(fig)


def draw_phase_stage_heatmap_ax(
    ax: Any,
    milestone_rows: list[dict[str, Any]],
    *,
    title: str = "Phase-by-Stage Heatmap",
    title_size: int = 14,
    show_note: bool = True,
) -> None:
    import numpy as np

    rows = phase_curriculum_rows(milestone_rows)
    matrix = np.array([stage_mix_vector(row["Stage Mix / Curriculum"]) for row in rows], dtype=float)
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels([STAGE_DISPLAY[stage_name] for stage_name in STAGE_ORDER], fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"{row['Phase Label']} | {SPLIT_DISPLAY.get(row['Train Split'], row['Train Split'])}" for row in rows],
        fontsize=9,
    )
    ax.set_title(title, loc="left", fontsize=title_size)

    for row_index, row in enumerate(rows):
        for col_index, stage_name in enumerate(STAGE_ORDER):
            value = matrix[row_index, col_index]
            label = f"{value:.0%}" if value > 0 else "—"
            text_color = "#111111" if value < 0.65 else "white"
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=9, color=text_color, fontweight="bold")
        ax.text(
            1.02,
            row_index,
            f"eval={SPLIT_DISPLAY.get(row['Eval Split'], row['Eval Split'])}",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#444444",
            transform=ax.get_yaxis_transform(),
        )

    ax.set_xticks([index - 0.5 for index in range(1, len(STAGE_ORDER))], minor=True)
    ax.set_yticks([index - 0.5 for index in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_note:
        ax.text(
            0.0,
            -0.16,
            "Cell value = phase sampling weight for that stage. Baseline is omitted because it had no staged curriculum.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#555555",
        )


def plot_phase_stage_heatmap(milestone_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    draw_phase_stage_heatmap_ax(ax, milestone_rows)
    save_plot_dual(fig, "phase_stage_heatmap")
    plt.close(fig)


def draw_split_transition_ladder_ax(
    ax: Any,
    timeline_rows: list[dict[str, Any]],
    *,
    title: str = "Split Transition Ladder",
    title_size: int = 14,
    show_note: bool = True,
) -> None:
    rows = list(timeline_rows)
    x_values = [row["Timeline Order"] for row in rows]
    split_to_y = {"unknown_pre_refactor": 0, "testmini": 1, "test": 2}
    train_y = [split_to_y.get(row["Train Split"], 0) for row in rows]
    eval_y = [split_to_y.get(row["Eval Split"], 0) for row in rows]

    ax.axvspan(0.5, 4.5, color="#eef3f9", alpha=0.6, zorder=0)
    ax.axvspan(4.5, 6.5, color="#fdebec", alpha=0.55, zorder=0)
    ax.axvspan(6.5, 7.5, color="#edf7f5", alpha=0.65, zorder=0)

    ax.plot(x_values, train_y, color="#1f77b4", linewidth=3, marker="o", markersize=8, label="train split")
    ax.plot(
        x_values,
        eval_y,
        color="#ff7f0e",
        linewidth=2.2,
        linestyle="--",
        marker="o",
        markersize=7,
        markerfacecolor="white",
        label="eval split",
    )

    for index, row in enumerate(rows):
        y_pos = train_y[index]
        dy = 0.24 if index % 2 == 0 else -0.30
        ax.text(
            row["Timeline Order"],
            y_pos + dy,
            row["Event Label"],
            ha="center",
            va="bottom" if dy > 0 else "top",
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
        )
        if row["Warm Start"]:
            ax.text(
                row["Timeline Order"],
                -0.18,
                "warm-start" if "/kaggle/input/" in str(row["Warm Start"]) else "resume selector",
                ha="center",
                va="top",
                fontsize=7.8,
                color="#555555",
            )

    ax.set_xticks(x_values)
    ax.set_xticklabels([row["Event Label"] for row in rows], rotation=20, ha="right", fontsize=9)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["unknown", "testmini", "test"])
    ax.set_ylim(-0.35, 2.35)
    ax.set_xlim(min(x_values) - 0.4, max(x_values) + 0.4)
    ax.set_title(title, loc="left", fontsize=title_size)
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)

    if show_note:
        ax.text(
            0.0,
            -0.24,
            "The key ladder move is train split testmini -> test at Large Phase C, while checkpoint evaluation stayed on testmini for comparability.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#555555",
        )


def plot_split_transition_ladder(timeline_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 5.8))
    draw_split_transition_ladder_ax(ax, timeline_rows)
    save_plot_dual(fig, "split_transition_ladder")
    plt.close(fig)


def draw_milestone_performance_heatmap_ax(
    ax: Any,
    milestone_rows: list[dict[str, Any]],
    *,
    title: str = "Milestone Performance Table / Heatmap",
    title_size: int = 14,
    show_note: bool = True,
) -> None:
    import numpy as np

    rows = key_curriculum_rows(milestone_rows)
    metrics = [
        ("Exact Match", False, "Exact"),
        ("Tolerance Accuracy", False, "Tolerance"),
        ("Parseable Rate", False, "Parseable"),
        ("Reasoning Tag Compliance", False, "Reasoning"),
        ("Solution Tag Compliance", False, "Solution"),
        ("Malformed Rate", True, "Malformed\n(lower better)"),
        ("Truncation Rate", True, "Truncation\n(lower better)"),
        ("Composite Score", False, "Composite"),
    ]

    color_matrix: list[list[float]] = []
    raw_matrix: list[list[float]] = []
    for row in rows:
        color_row = []
        raw_row = []
        for key, invert, _label in metrics:
            raw_value = float(row[key])
            raw_row.append(raw_value)
            color_row.append(1.0 - raw_value if invert else raw_value)
        raw_matrix.append(raw_row)
        color_matrix.append(color_row)

    matrix = np.array(color_matrix, dtype=float)
    raw_values = np.array(raw_matrix, dtype=float)
    ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([label for _key, _invert, label in metrics], rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row["Phase Label"] for row in rows], fontsize=9)
    ax.set_title(title, loc="left", fontsize=title_size)

    for row_index, row in enumerate(rows):
        for col_index, (_key, _invert, _label) in enumerate(metrics):
            raw_value = raw_values[row_index, col_index]
            text_color = "#111111" if matrix[row_index, col_index] < 0.64 else "white"
            ax.text(
                col_index,
                row_index,
                f"{raw_value:.0%}",
                ha="center",
                va="center",
                fontsize=8.8,
                color=text_color,
                fontweight="bold" if row["Milestone ID"] == "large_phase_c_best_recommended" else "normal",
            )

    ax.set_xticks([index - 0.5 for index in range(1, len(metrics))], minor=True)
    ax.set_yticks([index - 0.5 for index in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_note:
        ax.text(
            0.0,
            -0.16,
            "Cell text shows raw values. For color only, malformed and truncation are inverted so greener still means better.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#555555",
        )


def plot_milestone_performance_heatmap(milestone_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6.8))
    draw_milestone_performance_heatmap_ax(ax, milestone_rows)
    save_plot_dual(fig, "milestone_performance_heatmap")
    plt.close(fig)


def _layout_alluvial_column(
    order: list[str],
    weights: dict[str, float],
    scale: float,
    *,
    top: float = 0.94,
    bottom: float = 0.08,
    gap: float = 0.025,
) -> dict[str, tuple[float, float]]:
    used_height = sum(weights[name] for name in order) * scale + max(0, len(order) - 1) * gap
    y_cursor = top - (top - bottom - used_height) / 2
    positions: dict[str, tuple[float, float]] = {}
    for name in order:
        height = weights[name] * scale
        positions[name] = (y_cursor - height, y_cursor)
        y_cursor = y_cursor - height - gap
    return positions


def _draw_alluvial_band(
    ax: Any,
    x0: float,
    x1: float,
    y0_top: float,
    y0_bottom: float,
    y1_top: float,
    y1_bottom: float,
    *,
    color: str,
    alpha: float = 0.55,
) -> None:
    import numpy as np

    t = np.linspace(0.0, 1.0, 60)
    ease = 3 * t * t - 2 * t * t * t
    xs = x0 + (x1 - x0) * t
    top_curve = y0_top + (y1_top - y0_top) * ease
    bottom_curve = y0_bottom + (y1_bottom - y0_bottom) * ease
    polygon_x = list(xs) + list(xs[::-1])
    polygon_y = list(top_curve) + list(bottom_curve[::-1])
    ax.fill(polygon_x, polygon_y, color=color, alpha=alpha, linewidth=0.0)


def draw_curriculum_alluvial_ax(
    ax: Any,
    milestone_rows: list[dict[str, Any]],
    *,
    title: str = "Curriculum Alluvial Diagram",
    title_size: int = 14,
    show_note: bool = True,
) -> None:
    from matplotlib.patches import Rectangle

    rows = phase_curriculum_rows(milestone_rows)
    split_order = ["testmini", "test"]
    phase_order = [row["Milestone ID"] for row in rows]
    stage_order = list(STAGE_ORDER)

    split_weights = {split_name: sum(1.0 for row in rows if row["Train Split"] == split_name) for split_name in split_order}
    phase_weights = {row["Milestone ID"]: 1.0 for row in rows}
    stage_weights = {stage_name: 0.0 for stage_name in stage_order}
    for row in rows:
        mix = parse_stage_mix_text(row["Stage Mix / Curriculum"])
        for stage_name in stage_order:
            stage_weights[stage_name] += mix.get(stage_name, 0.0)

    total_weight = float(len(rows))
    gap = 0.025
    available = 0.94 - 0.08
    scale = min(
        (available - max(0, len(order) - 1) * gap) / total_weight
        for order in (split_order, phase_order, stage_order)
    )

    split_positions = _layout_alluvial_column(split_order, split_weights, scale, gap=gap)
    phase_positions = _layout_alluvial_column(phase_order, phase_weights, scale, gap=gap)
    stage_positions = _layout_alluvial_column(stage_order, stage_weights, scale, gap=gap)

    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    split_x, phase_x, stage_x = 0.12, 0.48, 0.84
    node_width = 0.10

    split_cursor = {name: split_positions[name][1] for name in split_order}
    phase_left_cursor = {name: phase_positions[name][1] for name in phase_order}
    phase_right_cursor = {name: phase_positions[name][1] for name in phase_order}
    stage_cursor = {name: stage_positions[name][1] for name in stage_order}

    for row in rows:
        split_name = row["Train Split"]
        phase_name = row["Milestone ID"]
        weight = 1.0
        split_top = split_cursor[split_name]
        split_bottom = split_top - weight * scale
        split_cursor[split_name] = split_bottom
        phase_top = phase_left_cursor[phase_name]
        phase_bottom = phase_top - weight * scale
        phase_left_cursor[phase_name] = phase_bottom
        _draw_alluvial_band(
            ax,
            split_x + node_width / 2,
            phase_x - node_width / 2,
            split_top,
            split_bottom,
            phase_top,
            phase_bottom,
            color=RUN_FAMILY_COLORS.get(row["Run Family"], "#999999"),
            alpha=0.45,
        )

    for row in rows:
        phase_name = row["Milestone ID"]
        mix = parse_stage_mix_text(row["Stage Mix / Curriculum"])
        for stage_name in stage_order:
            weight = mix.get(stage_name, 0.0)
            if weight <= 0:
                continue
            phase_top = phase_right_cursor[phase_name]
            phase_bottom = phase_top - weight * scale
            phase_right_cursor[phase_name] = phase_bottom
            stage_top = stage_cursor[stage_name]
            stage_bottom = stage_top - weight * scale
            stage_cursor[stage_name] = stage_bottom
            _draw_alluvial_band(
                ax,
                phase_x + node_width / 2,
                stage_x - node_width / 2,
                phase_top,
                phase_bottom,
                stage_top,
                stage_bottom,
                color=STAGE_COLORS[stage_name],
                alpha=0.58,
            )

    for split_name in split_order:
        y0, y1 = split_positions[split_name]
        ax.add_patch(
            Rectangle(
                (split_x - node_width / 2, y0),
                node_width,
                y1 - y0,
                facecolor="#f5f5f5",
                edgecolor="#333333",
                linewidth=1.0,
            )
        )
        ax.text(split_x, (y0 + y1) / 2, split_name, ha="center", va="center", fontsize=9, fontweight="bold")

    for row in rows:
        y0, y1 = phase_positions[row["Milestone ID"]]
        ax.add_patch(
            Rectangle(
                (phase_x - node_width / 2, y0),
                node_width,
                y1 - y0,
                facecolor=RUN_FAMILY_COLORS.get(row["Run Family"], "#dddddd"),
                edgecolor="#333333",
                linewidth=1.0,
                alpha=0.18,
            )
        )
        label = row["Phase Label"].replace("Phase ", "P")
        ax.text(phase_x, (y0 + y1) / 2, textwrap.fill(label, width=12), ha="center", va="center", fontsize=8.3)

    for stage_name in stage_order:
        y0, y1 = stage_positions[stage_name]
        share = stage_weights[stage_name] / total_weight if total_weight else 0.0
        ax.add_patch(
            Rectangle(
                (stage_x - node_width / 2, y0),
                node_width,
                y1 - y0,
                facecolor=STAGE_COLORS[stage_name],
                edgecolor="#333333",
                linewidth=1.0,
                alpha=0.20,
            )
        )
        ax.text(
            stage_x,
            (y0 + y1) / 2,
            f"{STAGE_SHORT[stage_name]}\n{share:.0%}",
            ha="center",
            va="center",
            fontsize=8.4,
            fontweight="bold",
        )

    ax.text(split_x, 0.985, "Train Split", ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(phase_x, 0.985, "Phase Instance", ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(stage_x, 0.985, "Curriculum Stage", ha="center", va="top", fontsize=10, fontweight="bold")
    ax.set_title(title, loc="left", fontsize=title_size)

    if show_note:
        ax.text(
            0.0,
            -0.08,
            "Baseline is omitted here. The diagram shows how staged RL phases draw from train splits and fan out into Stage 1/2/3 curriculum weights.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#555555",
        )


def plot_curriculum_alluvial(milestone_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 8))
    draw_curriculum_alluvial_ax(ax, milestone_rows)
    save_plot_dual(fig, "curriculum_alluvial")
    plt.close(fig)


def plot_curriculum_overview(milestone_rows: list[dict[str, Any]], timeline_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20, 17))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.15], hspace=0.28, wspace=0.16)
    ax_map = fig.add_subplot(grid[0, :])
    ax_heatmap = fig.add_subplot(grid[1, 0])
    ax_ladder = fig.add_subplot(grid[1, 1])
    ax_perf = fig.add_subplot(grid[2, :])

    draw_curriculum_map_ax(ax_map, milestone_rows, title="Curriculum Map", title_size=15, show_note=False)
    draw_phase_stage_heatmap_ax(ax_heatmap, milestone_rows, title="Phase-by-Stage Heatmap", title_size=13, show_note=False)
    draw_split_transition_ladder_ax(ax_ladder, timeline_rows, title="Split Transition Ladder", title_size=13, show_note=False)
    draw_milestone_performance_heatmap_ax(
        ax_perf,
        milestone_rows,
        title="Milestone Performance Table / Heatmap",
        title_size=14,
        show_note=False,
    )

    fig.suptitle("Entire RL Curriculum Overview", fontsize=19, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.012,
        "Recommended final checkpoint remains Large Phase C checkpoint-120: structure stayed perfect after the split scale-up and correctness improved from 0.50 to 0.75.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#555555",
    )
    save_plot_dual(fig, "entire_curriculum_overview")
    plt.close(fig)


def plot_base_vs_final_improvement(all_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = find_row(all_rows, run_family="baseline_pre_refactor", phase="baseline", global_step=0)
    final = find_row(all_rows, run_family="large_split_continue", phase="phase_c", global_step=120)
    if not baseline or not final:
        raise ValueError("Missing baseline or final checkpoint row for base-vs-final comparison plot.")

    lower_better = {"Malformed Rate", "Truncation Rate", "Repetition Rate", "Average Completion Tokens"}
    highlighted_metrics = {
        "Exact Match",
        "Reasoning Tag Compliance",
        "Solution Tag Compliance",
    }
    metric_groups = [
        (
            "Core Accuracy",
            [
                "Exact Match",
                "Tolerance Accuracy",
            ],
        ),
        (
            "Structure / Completion",
            [
                "Parseable Rate",
                "Reasoning Tag Compliance",
                "Solution Tag Compliance",
                "Well-Formed Rate",
                "Malformed Rate",
                "Completion Success Rate",
                "Truncation Rate",
            ],
        ),
        (
            "Generation",
            [
                "Average Completion Tokens",
                "Repetition Rate",
                "Sample Diversity",
            ],
        ),
        (
            "Scores / Reward",
            [
                "Average Total Reward",
                "Structure Score",
                "Correctness Score",
                "Composite Score",
            ],
        ),
    ]

    total_rows = sum(len(metrics) + 2 for _, metrics in metric_groups)
    fig_height = max(12, 0.56 * total_rows + 1.6)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    fig.subplots_adjust(left=0.035, right=0.965, top=0.955, bottom=0.06)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, total_rows + 0.35)

    line_color = "#e4e4e4"
    text_color = "#111111"
    muted_color = "#555555"
    pos_color = "#1a7f37"
    neg_color = "#b42318"
    neutral_color = "#666666"
    highlight_fill = "#f3f7ff"

    x_eval = 0.02
    x_base = 0.48
    x_trained = 0.67
    x_improvement = 0.86

    def format_value(metric: str, value: float) -> str:
        if metric == "Average Completion Tokens":
            return f"{value:.1f}"
        if metric == "Average Total Reward":
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{value * 100:.1f}%"

    def format_improvement(metric: str, base_value: float, final_value: float) -> tuple[str, str]:
        if math.isclose(base_value, 0.0):
            return "—", neutral_color
        if metric in lower_better:
            improvement = (base_value - final_value) / base_value * 100.0
        else:
            improvement = (final_value - base_value) / base_value * 100.0
        color = pos_color if improvement > 0 else neg_color if improvement < 0 else neutral_color
        sign = "+" if improvement > 0 else ""
        return f"{sign}{improvement:.1f}%", color

    y = total_rows + 0.10
    for section_title, metrics in metric_groups:
        y -= 0.65
        ax.text(x_eval, y, section_title, fontsize=16, ha="left", va="center", color=text_color)
        y -= 0.55
        ax.hlines(y, 0.0, 1.0, colors=line_color, linewidth=1.0)
        y -= 0.48
        ax.text(x_eval, y, "Eval", fontsize=12, ha="left", va="center", color=text_color, fontweight="bold")
        ax.text(x_base, y, "Base", fontsize=12, ha="right", va="center", color=text_color, fontweight="bold")
        ax.text(x_trained, y, "Trained", fontsize=12, ha="right", va="center", color=text_color, fontweight="bold")
        ax.text(x_improvement, y, "% Improvement", fontsize=12, ha="right", va="center", color=text_color, fontweight="bold")
        y -= 0.42
        ax.hlines(y, 0.0, 1.0, colors=line_color, linewidth=1.0)

        for metric in metrics:
            row_top = y
            y -= 0.56
            base_value = float(baseline[metric])
            final_value = float(final[metric])
            improvement_text, improvement_color = format_improvement(metric, base_value, final_value)
            label = f"{metric} (lower better)" if metric in lower_better else metric
            is_highlighted = metric in highlighted_metrics
            row_fontweight = "bold" if is_highlighted else "normal"

            if is_highlighted:
                ax.axhspan(y - 0.43, row_top, xmin=0.0, xmax=1.0, facecolor=highlight_fill, edgecolor="none", zorder=0)

            ax.text(x_eval, y, label, fontsize=11, ha="left", va="center", color=text_color, fontweight=row_fontweight)
            ax.text(x_base, y, format_value(metric, base_value), fontsize=11, ha="right", va="center", color=text_color, fontweight=row_fontweight)
            ax.text(x_trained, y, format_value(metric, final_value), fontsize=11, ha="right", va="center", color=text_color, fontweight=row_fontweight)
            ax.text(x_improvement, y, improvement_text, fontsize=11, ha="right", va="center", color=improvement_color, fontweight="bold")

            y -= 0.43
            ax.hlines(y, 0.0, 1.0, colors=line_color, linewidth=0.9)

    fig.suptitle("Base vs Trained Improvement", fontsize=24, fontweight="bold", y=0.992)
    fig.text(
        0.5,
        0.016,
        "Pre-Refactor Baseline (`base`) vs Large Phase C checkpoint-120 (`lgc:c:120`) from master_table_all_checkpoints.",
        ha="center",
        va="bottom",
        fontsize=10,
        color=muted_color,
    )
    fig.savefig(PLOTS_DIR / "base_vs_final_improvement.png", dpi=220, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "base_vs_final_improvement.svg", bbox_inches="tight")
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
        "- `tables/master_table_milestones.md`",
        "- `tables/master_table_all_checkpoints.csv`",
        "- `tables/master_table_all_checkpoints.md`",
        "- `tables/resource_runtime_tuning.csv`",
        "- `tables/resource_runtime_tuning.md`",
        "- `tables/resource_knob_tradeoffs.csv`",
        "- `tables/resource_knob_tradeoffs.md`",
        "- `tables/runtime_stabilization_timeline.csv`",
        "- `tables/runtime_stabilization_timeline.md`",
        "",
        "Generated plots:",
        "- `plots/evolution_panels.png`",
        "- `plots/evolution_panels_notebook.png`",
        "- `plots/base_vs_final_improvement.png`",
        "- `plots/curriculum_map.png`",
        "- `plots/phase_stage_heatmap.png`",
        "- `plots/split_transition_ladder.png`",
        "- `plots/milestone_performance_heatmap.png`",
        "- `plots/curriculum_alluvial.png`",
        "- `plots/entire_curriculum_overview.png`",
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


def write_table_docs(
    milestone_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
    knob_rows: list[dict[str, Any]],
    timeline_rows: list[dict[str, Any]],
    controller_audit_rows: list[dict[str, Any]],
) -> None:
    master_glossary = render_column_glossary(MASTER_COLUMN_GROUPS, MASTER_COLUMN_DESCRIPTIONS)
    resource_glossary = render_column_glossary(
        [("Resource Runtime Tuning Columns", list(resource_rows[0].keys()))], RESOURCE_RUNTIME_COLUMN_DESCRIPTIONS
    )
    knob_glossary = render_column_glossary(
        [("Resource Knob Tradeoff Columns", list(knob_rows[0].keys()))], RESOURCE_KNOB_COLUMN_DESCRIPTIONS
    )
    timeline_glossary = render_column_glossary(
        [("Runtime Stabilization Timeline Columns", list(timeline_rows[0].keys()))], TIMELINE_COLUMN_DESCRIPTIONS
    )
    controller_glossary = render_column_glossary(
        [("Controller Audit Columns", CONTROLLER_AUDIT_COLUMNS)], CONTROLLER_AUDIT_COLUMN_DESCRIPTIONS
    )

    write_markdown_doc(
        markdown_path=TABLES_DIR / "master_table_milestones.md",
        title="Master Table A: Milestones Only",
        csv_name="tables/master_table_milestones.csv",
        purpose="Compact report-ready summary of the baseline, major phase milestones, important regressions, and final checkpoint recommendations.",
        glossary_markdown=master_glossary,
        analysis_lines=analysis_for_milestones(milestone_rows),
    )
    write_markdown_doc(
        markdown_path=TABLES_DIR / "master_table_all_checkpoints.md",
        title="Master Table B: Every Checkpoint Row",
        csv_name="tables/master_table_all_checkpoints.csv",
        purpose="Checkpoint-level audit trail covering every locally available evaluated checkpoint across smoke, larger-split, and dedicated Phase D runs.",
        glossary_markdown=master_glossary,
        analysis_lines=analysis_for_all_checkpoints(all_rows),
    )
    write_markdown_doc(
        markdown_path=TABLES_DIR / "resource_runtime_tuning.md",
        title="Table C: Resource-Constrained Runtime Tuning",
        csv_name="tables/resource_runtime_tuning.csv",
        purpose="Configuration-and-outcome view showing how the stable Kaggle T4 profile differs from the reference default profile and what each successful run achieved under the constrained profile.",
        glossary_markdown=resource_glossary,
        analysis_lines=analysis_for_resource_runtime(resource_rows),
    )
    write_markdown_doc(
        markdown_path=TABLES_DIR / "resource_knob_tradeoffs.md",
        title="Resource Knob Tradeoffs",
        csv_name="tables/resource_knob_tradeoffs.csv",
        purpose="Per-knob explanation of how the Kaggle T4 profile reduced memory pressure and what tradeoffs were accepted in exchange.",
        glossary_markdown=knob_glossary,
        analysis_lines=analysis_for_knob_tradeoffs(knob_rows),
    )
    write_markdown_doc(
        markdown_path=TABLES_DIR / "runtime_stabilization_timeline.md",
        title="Runtime Stabilization Timeline",
        csv_name="tables/runtime_stabilization_timeline.csv",
        purpose="Chronological milestone view showing how the project moved from the weak baseline to smoke validation, larger-split correctness gains, and dedicated Phase D specialization.",
        glossary_markdown=timeline_glossary,
        analysis_lines=analysis_for_timeline(timeline_rows),
    )
    write_markdown_doc(
        markdown_path=TABLES_DIR / "controller_rule_audit.md",
        title="Controller Rule Audit",
        csv_name="tables/controller_rule_audit.csv",
        purpose="Checkpoint-by-checkpoint audit of which reward-controller rules fired, what evidence triggered them, and which reward weights changed.",
        glossary_markdown=controller_glossary,
        analysis_lines=analysis_for_controller_audit(controller_audit_rows),
    )


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
    if len(all_rows) <= 1:
        fallback_rows = load_csv_rows(TABLES_DIR / "master_table_all_checkpoints.csv")
        if fallback_rows:
            all_rows = sanitize_rows(fallback_rows)
    milestone_rows = sanitize_rows(build_milestone_rows(all_rows))
    resource_rows = build_resource_rows(milestone_rows)
    knob_rows = build_knob_tradeoff_rows()
    timeline_rows = build_runtime_timeline_rows(milestone_rows)
    controller_audit_rows = sanitize_rows(build_controller_audit_rows(all_rows))

    write_csv(TABLES_DIR / "master_table_all_checkpoints.csv", all_rows, MASTER_TABLE_COLUMNS)
    write_csv(TABLES_DIR / "master_table_milestones.csv", milestone_rows, MASTER_TABLE_COLUMNS)
    write_csv(TABLES_DIR / "resource_runtime_tuning.csv", resource_rows, list(resource_rows[0].keys()))
    write_csv(TABLES_DIR / "resource_knob_tradeoffs.csv", knob_rows, list(knob_rows[0].keys()))
    write_csv(TABLES_DIR / "runtime_stabilization_timeline.csv", timeline_rows, list(timeline_rows[0].keys()))
    write_csv(TABLES_DIR / "controller_rule_audit.csv", controller_audit_rows, CONTROLLER_AUDIT_COLUMNS)
    write_table_docs(milestone_rows, all_rows, resource_rows, knob_rows, timeline_rows, controller_audit_rows)

    plot_main_evolution(all_rows)
    plot_main_evolution(all_rows, stem="evolution_panels_notebook", include_notebook_panel=True)
    plot_base_vs_final_improvement(all_rows)
    plot_curriculum_map(milestone_rows)
    plot_phase_stage_heatmap(milestone_rows)
    plot_split_transition_ladder(timeline_rows)
    plot_milestone_performance_heatmap(milestone_rows)
    plot_curriculum_alluvial(milestone_rows)
    plot_curriculum_overview(milestone_rows, timeline_rows)
    plot_runtime_timeline(timeline_rows)
    plot_frontier(all_rows)
    plot_heatmap(all_rows)
    plot_controller_rule_heatmap(controller_audit_rows)
    plot_lineage()

    write_readme(milestone_rows, all_rows, resource_rows)
    write_summary(milestone_rows)
    write_data_sources()


if __name__ == "__main__":
    main()
