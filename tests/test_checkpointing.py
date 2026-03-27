"""Tests for checkpoint score and alias resolution."""

import tempfile
import unittest
from pathlib import Path

from staged_rl.checkpointing import CheckpointRegistry, build_resume_plan, compute_checkpoint_scores
from staged_rl.config import CheckpointScoreConfig


class CheckpointingTests(unittest.TestCase):
    def test_compute_checkpoint_scores(self):
        scores = compute_checkpoint_scores(
            {
                "normalized_exact_match": 0.5,
                "tolerance_accuracy": 0.6,
                "parseable_answer_rate": 0.9,
                "solution_tag_compliance": 0.95,
                "reasoning_tag_compliance": 0.94,
                "malformed_answer_rate": 0.05,
                "truncation_rate": 0.1,
            },
            CheckpointScoreConfig(),
        )
        self.assertIn("structure_score", scores)
        self.assertIn("correctness_score", scores)
        self.assertIn("composite_score", scores)

    def test_registry_and_resume_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "phase_a"
            registry = CheckpointRegistry(run_dir)
            registry.register(
                {
                    "checkpoint_path": str(run_dir / "checkpoint-10"),
                    "global_step": 10,
                    "phase_name": "phase_a",
                    "metrics": {"normalized_exact_match": 0.3},
                    "scores": {
                        "structure_score": 0.4,
                        "correctness_score": 0.3,
                        "composite_score": 0.35,
                    },
                }
            )
            registry.register(
                {
                    "checkpoint_path": str(run_dir / "checkpoint-20"),
                    "global_step": 20,
                    "phase_name": "phase_a",
                    "metrics": {"normalized_exact_match": 0.5},
                    "scores": {
                        "structure_score": 0.45,
                        "correctness_score": 0.5,
                        "composite_score": 0.48,
                    },
                }
            )
            (run_dir / "checkpoint-20").mkdir(parents=True, exist_ok=True)

            plan = build_resume_plan(
                selector="best_composite",
                current_phase="phase_b",
                current_phase_dir=Path(tmpdir) / "phase_b",
                search_dirs=[run_dir],
                default_model_name="base-model",
            )
            self.assertEqual(plan.model_load_path, "base-model")
            self.assertIsNone(plan.trainer_resume_path)
            self.assertEqual(plan.adapter_warm_start_path, str(run_dir / "checkpoint-20"))

            latest_plan = build_resume_plan(
                selector="latest",
                current_phase="phase_a",
                current_phase_dir=run_dir,
                search_dirs=[run_dir],
                default_model_name="base-model",
            )
            self.assertEqual(latest_plan.trainer_resume_path, str(run_dir / "checkpoint-20"))
            self.assertIsNone(latest_plan.adapter_warm_start_path)

            warm_start_plan = build_resume_plan(
                selector=str(run_dir / "checkpoint-20"),
                current_phase="phase_a",
                current_phase_dir=run_dir,
                search_dirs=[run_dir],
                default_model_name="base-model",
                force_warm_start=True,
            )
            self.assertEqual(warm_start_plan.model_load_path, "base-model")
            self.assertIsNone(warm_start_plan.trainer_resume_path)
            self.assertEqual(warm_start_plan.adapter_warm_start_path, str(run_dir / "checkpoint-20"))


if __name__ == "__main__":
    unittest.main()
