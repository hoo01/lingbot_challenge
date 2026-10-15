from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "plan_h0_timing.py"
SPEC = importlib.util.spec_from_file_location("plan_h0_timing", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
timing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timing)


class TimingBudgetTests(unittest.TestCase):
    def records(self) -> list[dict]:
        result = []
        for step in range(1, 101):
            record = {"step": step, "step_seconds": 3.0 if 21 <= step <= 99 else 99.0}
            if step == 1:
                record["setup_seconds"] = 120.0
            if step % 10 == 0:
                record["validation_seconds"] = 10.0
            result.append(record)
        return result

    def test_frozen_window_and_scaled_overheads_produce_complete_budget(self) -> None:
        report = timing.build_budget(
            self.records(),
            checkpoint_records=[{"step": 1, "checkpoint_seconds": 20.0}],
            validation_scale=4.0,
        )
        self.assertEqual(report["stable_window"]["samples"], 79)
        self.assertEqual(report["step_seconds"]["median"], 3.0)
        self.assertEqual(report["validation_seconds"], 40.0)
        self.assertEqual(report["checkpoint_seconds"], 20.0)
        self.assertTrue(report["timing_gate_complete"])
        target_15000 = next(row for row in report["targets"] if row["target_step"] == 15000)
        expected_seconds = 120.0 + 15000 * 3.0 + 60 * 40.0 + 30 * 20.0
        self.assertAlmostEqual(target_15000["budget_hours"], 1.2 * expected_seconds / 3600)

    def test_missing_checkpoint_timing_is_provisional_and_cannot_pass(self) -> None:
        report = timing.build_budget(self.records(), validation_scale=4.0)
        self.assertFalse(report["timing_gate_complete"])
        self.assertIn("provisional", report["status"])
        self.assertFalse(any(row["fits_rental_window"] for row in report["targets"]))

    def test_invalid_or_missing_stable_window_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "window 21..99 is incomplete"):
            timing.build_budget([{"step": 20, "step_seconds": 3.0}])


if __name__ == "__main__":
    unittest.main()
