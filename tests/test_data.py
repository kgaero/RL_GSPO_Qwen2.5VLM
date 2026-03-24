"""Tests for stage filters and dataset analysis."""

import unittest

from staged_rl.config import build_default_stage_specs
from staged_rl.data import analyze_dataset_records, match_filter_spec, normalize_context_family


SAMPLE_NUMERIC = {
    "pid": "1",
    "question_type": "free_form",
    "answer_type": "integer",
    "language": "english",
    "context": "synthetic scene",
    "context_family": "synthetic scene",
    "source": "clevr-math",
    "task": "math word problem",
    "category": "math-targeted-vqa",
    "grade": "elementary school",
    "skills": ["arithmetic reasoning"],
    "unit": "",
    "precision": None,
    "answer_mode": "numeric_free_form",
    "answer": "4",
    "question": "How many objects are left?",
}

SAMPLE_MULTI = {
    "pid": "2",
    "question_type": "multi_choice",
    "answer_type": "text",
    "language": "english",
    "context": "geometry diagram",
    "context_family": "geometry diagram",
    "source": "geometry3k",
    "task": "geometry problem solving",
    "category": "math-targeted-vqa",
    "grade": "high school",
    "skills": ["geometry reasoning", "algebraic reasoning"],
    "unit": "",
    "precision": None,
    "answer_mode": "multi_choice",
    "answer": "97",
    "question": "Find m angle H",
}


class DataTests(unittest.TestCase):
    def test_normalize_context_family(self):
        self.assertEqual(normalize_context_family("bar chart"), "chart")
        self.assertEqual(normalize_context_family("line plot"), "plot")
        self.assertEqual(normalize_context_family("geometry diagram"), "geometry diagram")

    def test_stage_filters(self):
        stages = build_default_stage_specs()
        self.assertTrue(match_filter_spec(SAMPLE_NUMERIC, stages["stage1_easy_numeric"].filter_spec))
        self.assertFalse(match_filter_spec(SAMPLE_MULTI, stages["stage1_easy_numeric"].filter_spec))
        self.assertTrue(match_filter_spec(SAMPLE_MULTI, stages["stage4_multi_choice"].filter_spec))

    def test_dataset_analysis(self):
        stages = build_default_stage_specs()
        analysis = analyze_dataset_records([SAMPLE_NUMERIC, SAMPLE_MULTI], stages)
        self.assertEqual(analysis["total_rows"], 2)
        self.assertEqual(analysis["field_counts"]["question_type"]["free_form"], 1)
        self.assertEqual(analysis["stage_summaries"]["stage1_easy_numeric"]["count"], 1)
        self.assertEqual(analysis["stage_summaries"]["stage4_multi_choice"]["count"], 1)


if __name__ == "__main__":
    unittest.main()

