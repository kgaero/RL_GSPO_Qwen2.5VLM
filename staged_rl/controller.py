"""Metric-gated reward scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from .config import RewardGateConfig


@dataclass
class RewardControllerState:
    """Persisted reward-controller state."""

    history: list[dict[str, Any]] = field(default_factory=list)
    reward_weights: dict[str, float] = field(default_factory=dict)


class RewardController:
    """Update reward weights from recent checkpoint metrics."""

    def __init__(
        self,
        gate_config: RewardGateConfig,
        component_bounds: Mapping[str, tuple[float, float]],
        initial_weights: Mapping[str, float],
    ) -> None:
        self.gate_config = gate_config
        self.component_bounds = dict(component_bounds)
        self.state = RewardControllerState(
            history=[],
            reward_weights={name: float(weight) for name, weight in initial_weights.items()},
        )

    @classmethod
    def from_state(
        cls,
        gate_config: RewardGateConfig,
        component_bounds: Mapping[str, tuple[float, float]],
        initial_weights: Mapping[str, float],
        state_dict: Mapping[str, Any] | None,
    ) -> "RewardController":
        controller = cls(gate_config, component_bounds, initial_weights)
        if state_dict:
            controller.state = RewardControllerState(
                history=list(state_dict.get("history", [])),
                reward_weights={key: float(value) for key, value in state_dict.get("reward_weights", {}).items()},
            )
            for name, weight in initial_weights.items():
                controller.state.reward_weights.setdefault(name, float(weight))
        return controller

    def to_dict(self) -> dict[str, Any]:
        """Serialize controller state."""

        return {
            "history": list(self.state.history),
            "reward_weights": dict(self.state.reward_weights),
        }

    def current_weights(self) -> dict[str, float]:
        """Return the current reward weights."""

        return dict(self.state.reward_weights)

    def update_from_metrics(self, metrics: Mapping[str, float], max_completion_length: int) -> dict[str, float]:
        """Apply gating rules after a checkpoint evaluation."""

        history_entry = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
        self.state.history.append(history_entry)
        weights = dict(self.state.reward_weights)
        cfg = self.gate_config

        parseable = metrics.get("parseable_answer_rate", 0.0)
        solution_ok = metrics.get("solution_tag_compliance", 0.0)
        reasoning_ok = metrics.get("reasoning_tag_compliance", 0.0)
        malformed = metrics.get("malformed_answer_rate", 1.0)
        truncation = metrics.get("truncation_rate", 1.0)
        avg_tokens = metrics.get("average_completion_tokens", float(max_completion_length))
        exact = metrics.get("normalized_exact_match", 0.0)

        if parseable < cfg.parseable_floor_threshold and "parseable_reward" in weights:
            weights["parseable_reward"] = max(weights["parseable_reward"], cfg.parseable_guard_weight)

        if (
            solution_ok < cfg.solution_tag_floor_threshold
            or reasoning_ok < cfg.reasoning_tag_floor_threshold
            or malformed > cfg.malformed_ceiling_threshold
        ) and "format_reward" in weights:
            weights["format_reward"] = max(weights["format_reward"], cfg.formatting_guard_weight)

        if (
            truncation > cfg.truncation_ceiling_threshold
            or avg_tokens > cfg.average_token_fraction_threshold * max_completion_length
        ) and "finished_reward" in weights:
            weights["finished_reward"] += cfg.finish_step

        if (
            parseable >= cfg.parseable_stable_threshold
            and solution_ok >= cfg.solution_tag_stable_threshold
            and reasoning_ok >= cfg.reasoning_tag_stable_threshold
            and malformed <= cfg.malformed_stable_threshold
            and truncation <= cfg.truncation_stable_threshold
        ):
            recent = self.state.history[-cfg.stable_window :]
            if len(recent) >= cfg.stable_window:
                previous_exact = recent[0].get("normalized_exact_match", exact)
                if exact - previous_exact < cfg.exact_match_plateau_delta and "correctness_reward" in weights:
                    weights["correctness_reward"] += cfg.correctness_step

        weights["format_reward"] = max(weights.get("format_reward", 0.0), cfg.format_floor)
        weights["parseable_reward"] = max(weights.get("parseable_reward", 0.0), cfg.parseable_floor)
        weights["finished_reward"] = max(weights.get("finished_reward", 0.0), cfg.finish_floor)

        for name, weight in list(weights.items()):
            lower, upper = self.component_bounds[name]
            weights[name] = min(max(weight, lower), upper)

        self.state.reward_weights = weights
        return dict(weights)
