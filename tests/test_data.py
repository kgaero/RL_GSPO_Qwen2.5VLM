"""Tests for stage filters and dataset analysis."""

import unittest
from unittest import mock

from staged_rl.config import build_default_stage_specs
from staged_rl.data import (
    _apply_runtime_image_transform,
    analyze_dataset_records,
    match_filter_spec,
    normalize_context_family,
    normalize_image_payload,
)


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
    class _FakeImage:
        def __init__(self, size=(16, 16), mode="L"):
            self.size = size
            self.mode = mode

        def resize(self, size):
            return DataTests._FakeImage(size=size, mode=self.mode)

        def convert(self, mode):
            return DataTests._FakeImage(size=self.size, mode=mode)

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

    def test_normalize_image_payload_decodes_dataset_dict(self):
        fake_image = self._FakeImage()
        with mock.patch("staged_rl.data._load_image_payload", return_value=fake_image):
            normalized = normalize_image_payload({"path": "/tmp/example.png", "bytes": None}, image_size=64)

        self.assertIsInstance(normalized, self._FakeImage)
        self.assertEqual(normalized.size, (64, 64))
        self.assertEqual(normalized.mode, "RGB")

    def test_normalize_image_payload_prefers_existing_image_object(self):
        fake_image = self._FakeImage()
        normalized = normalize_image_payload(fake_image, image_size=32)

        self.assertIsInstance(normalized, self._FakeImage)
        self.assertEqual(normalized.size, (32, 32))
        self.assertEqual(normalized.mode, "RGB")

    def test_runtime_image_transform_normalizes_interleaved_dict_payloads(self):
        class _FakeDataset:
            def __init__(self, image):
                self.image = image

            def with_transform(self, transform):
                return transform({"image": self.image})

        fake_image = self._FakeImage()
        with mock.patch("staged_rl.data.normalize_image_payload", return_value=fake_image) as patched:
            transformed = _apply_runtime_image_transform(_FakeDataset({"path": "/tmp/example.png"}), image_size=48)

        patched.assert_called_once_with({"path": "/tmp/example.png"}, image_size=48)
        self.assertIs(transformed["image"], fake_image)


if __name__ == "__main__":
    unittest.main()
