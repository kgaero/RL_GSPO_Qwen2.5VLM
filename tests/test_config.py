"""Tests for named hardware profiles."""

import unittest

from staged_rl.config import apply_hardware_profile, build_default_run_config


class HardwareProfileTests(unittest.TestCase):
    def test_kaggle_t4_profile_applies_expected_overrides(self):
        run_config = build_default_run_config()
        apply_hardware_profile(run_config, "kaggle_t4")

        self.assertEqual(run_config.hardware_profile_name, "kaggle_t4")
        self.assertEqual(run_config.model.max_seq_length, 4096)
        self.assertEqual(run_config.model.gpu_memory_utilization, 0.55)
        self.assertEqual(run_config.model.lora_rank, 8)
        self.assertEqual(run_config.trainer_defaults.gradient_accumulation_steps, 4)
        self.assertEqual(run_config.trainer_defaults.num_generations, 2)
        self.assertEqual(run_config.trainer_defaults.max_completion_length, 160)
        self.assertEqual(run_config.eval.num_samples_per_prompt, 2)
        self.assertEqual(run_config.eval.max_eval_examples_per_subset, 8)
        self.assertEqual(run_config.phases["phase_d"].trainer_overrides["max_completion_length"], 224)


if __name__ == "__main__":
    unittest.main()
