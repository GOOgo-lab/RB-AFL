from __future__ import annotations

import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is an optional heavy CI dependency")
class BalancedQuantizationTests(unittest.TestCase):
    def test_exact_hamming_weight_with_ties(self) -> None:
        from rbafl.model import feature_to_bits

        feature = np.zeros(256, dtype=np.float32)
        bits = feature_to_bits(feature, 256, "balanced_topk")
        self.assertEqual(int(bits.sum()), 128)
        self.assertEqual(int((bits == 0).sum()), 128)


if __name__ == "__main__":
    unittest.main()

