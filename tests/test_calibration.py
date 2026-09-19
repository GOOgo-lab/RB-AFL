from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


@unittest.skipUnless(importlib.util.find_spec("torch"), "Calibration module imports the optional model stack")
class CalibrationTests(unittest.TestCase):
    def test_target_far_then_minimum_frr(self) -> None:
        from rbafl.calibration import calibrate_threshold

        scores = pd.DataFrame(
            {
                "score_type": ["genuine"] * 4 + ["impostor"] * 5,
                "nc": [0.95, 0.90, 0.85, 0.80, 0.10, 0.20, 0.30, 0.40, 0.60],
            }
        )
        with tempfile.TemporaryDirectory() as temp:
            summary = calibrate_threshold(
                scores,
                Path(temp),
                target_far=0.0,
                threshold_step=0.01,
            )
        self.assertEqual(summary["selection_policy"], "target_far_then_minimum_frr")
        self.assertEqual(summary["false_accept_count"], 0)
        self.assertAlmostEqual(summary["selected_threshold"], 0.61)


if __name__ == "__main__":
    unittest.main()

