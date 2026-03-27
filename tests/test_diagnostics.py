"""Tests for diagnostics helpers."""

import tempfile
import unittest
from pathlib import Path

from staged_rl.diagnostics import write_fatal_error


class DiagnosticsTests(unittest.TestCase):
    def test_write_fatal_error_persists_traceback_and_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fatal_error.txt"
            try:
                raise RuntimeError("boom")
            except RuntimeError as exc:
                write_fatal_error(output_path, exc, {"phase": "phase_a", "resume": None})

            text = output_path.read_text(encoding="utf-8")
            self.assertIn("exception_type: RuntimeError", text)
            self.assertIn("exception_message: boom", text)
            self.assertIn('"phase": "phase_a"', text)
            self.assertIn("traceback:", text)
            self.assertIn("RuntimeError: boom", text)


if __name__ == "__main__":
    unittest.main()
