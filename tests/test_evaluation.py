"""Tests for metric aggregation."""

import unittest

from staged_rl.evaluation import aggregate_subset_metrics, determine_failure_mode


class EvaluationTests(unittest.TestCase):
    def test_determine_failure_mode(self):
        record = {
            "truncation": True,
            "solution_tag_compliance": False,
            "reasoning_tag_compliance": False,
            "malformed_answer": True,
            "parseable_answer": False,
            "normalized_exact_match": False,
        }
        self.assertEqual(determine_failure_mode(record), "truncation")

    def test_aggregate_subset_metrics(self):
        per_prompt_records = [
            {
                "samples": [
                    {
                        "normalized_exact_match": True,
                        "tolerance_match": True,
                        "parseable_answer": True,
                        "solution_tag_compliance": True,
                        "reasoning_tag_compliance": True,
                        "malformed_answer": False,
                        "truncation": False,
                        "completion_tokens": 10,
                        "repetition_rate": 0.0,
                        "total_reward": 1.0,
                        "reward_component/format_reward": 1.0,
                    }
                ],
                "best_of_k_accuracy": True,
                "best_of_k_tolerance_accuracy": True,
                "sampled_answer_diversity": 0.5,
            }
        ]
        all_sample_records = per_prompt_records[0]["samples"]
        metrics = aggregate_subset_metrics(per_prompt_records, all_sample_records)
        self.assertEqual(metrics["normalized_exact_match"], 1.0)
        self.assertEqual(metrics["best_of_k_accuracy"], 1.0)
        self.assertIn("reward_component/format_reward_mean", metrics)


if __name__ == "__main__":
    unittest.main()
