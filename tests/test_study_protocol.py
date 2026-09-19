from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rbafl.protocol import (
    OBJECT_DELETE_MASK_REPEATS,
    OBJECT_DELETE_RATIOS,
    SCALE_FACTORS,
    attack_plan,
)
from rbafl.study import (
    EXPECTED_TEST_PAIRS_PER_SEED,
    EXPECTED_TRAINING_SEEDS,
    collapse_directed_uniqueness,
    create_exact_split_from_manifest,
    load_config,
    one_sided_binomial_upper,
    validate_split_file,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_public_config_is_valid(self) -> None:
        config = load_config(REPOSITORY_ROOT / "configs" / "protocol_50_20_41.json")
        self.assertEqual(config["study"]["split_counts"], {"train": 50, "calibration": 20, "test": 41})
        self.assertEqual(config["study"]["training_seeds"], list(EXPECTED_TRAINING_SEEDS))

    def test_attack_axes_have_valid_baselines_and_strong_deletion(self) -> None:
        self.assertIn(1.0, SCALE_FACTORS)
        self.assertNotIn(0.0, SCALE_FACTORS)
        self.assertGreaterEqual(max(OBJECT_DELETE_RATIOS), 0.50)

        deletion_cases = [case for case in attack_plan() if case.attack == "object_delete"]
        clean = [case for case in deletion_cases if case.strength == 0.0]
        attacked = [case for case in deletion_cases if case.strength > 0.0]
        self.assertEqual(len(clean), 1)
        for strength in OBJECT_DELETE_RATIOS[1:]:
            self.assertEqual(
                sum(case.strength == strength for case in attacked),
                OBJECT_DELETE_MASK_REPEATS,
            )
        self.assertEqual(len({case.case_id for case in deletion_cases}), len(deletion_cases))


class SplitTests(unittest.TestCase):
    def _manifest(self, root: Path, order: np.ndarray | None = None) -> Path:
        identities = [f"identity_{index:03d}" for index in range(111)]
        if order is not None:
            identities = [identities[int(index)] for index in order]
        rows = [
            {
                "identity": identity,
                "sample_type": "base",
                "source_path": f"data/{identity}.geojson",
                "tensor_path": f"tensors/{identity}/base.npy",
            }
            for identity in identities
        ]
        path = root / "manifest.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_exact_disjoint_deterministic_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root)
            first = root / "first.csv"
            create_exact_split_from_manifest(manifest, first, split_seed=20260909)
            report = validate_split_file(first)
            self.assertEqual(report["counts"], {"train": 50, "calibration": 20, "test": 41})

            shuffled_manifest = self._manifest(root, np.random.default_rng(5).permutation(111))
            second = root / "second.csv"
            create_exact_split_from_manifest(shuffled_manifest, second, split_seed=20260909)
            a = pd.read_csv(first)[["identity", "split", "split_order", "split_seed"]]
            b = pd.read_csv(second)[["identity", "split", "split_order", "split_seed"]]
            pd.testing.assert_frame_equal(a, b)

    def test_incompatible_existing_split_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root)
            split = root / "split.csv"
            create_exact_split_from_manifest(manifest, split, split_seed=20260909)
            audit = json.loads(split.with_suffix(".audit.json").read_text(encoding="utf-8"))
            audit["protocol_fingerprint"] = "wrong"
            split.with_suffix(".audit.json").write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                create_exact_split_from_manifest(manifest, split, split_seed=20260909)


class UniquenessTests(unittest.TestCase):
    def test_unordered_pair_count_is_8200(self) -> None:
        identities = [f"id_{index:02d}" for index in range(41)]
        rows = []
        for seed in EXPECTED_TRAINING_SEEDS:
            for i, left in enumerate(identities):
                for j, right in enumerate(identities):
                    if i == j:
                        continue
                    nc = 0.2 + ((i + j + seed) % 20) / 100.0
                    rows.append(
                        {
                            "training_seed": seed,
                            "registered_identity": left,
                            "tested_identity": right,
                            "nc": nc,
                        }
                    )
        result = collapse_directed_uniqueness(pd.DataFrame(rows), threshold=0.75)
        self.assertEqual(EXPECTED_TEST_PAIRS_PER_SEED, math.comb(41, 2))
        self.assertEqual(len(result), 8200)
        self.assertTrue(result.groupby("training_seed")["pair_id"].nunique().eq(820).all())
        self.assertFalse(result.duplicated(["training_seed", "pair_id"]).any())

    def test_zero_event_one_sided_upper_bound(self) -> None:
        upper = one_sided_binomial_upper(0, 8200)
        self.assertAlmostEqual(upper, 1.0 - 0.05 ** (1.0 / 8200), places=14)


if __name__ == "__main__":
    unittest.main()
