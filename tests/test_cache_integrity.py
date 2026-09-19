from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


@unittest.skipUnless(importlib.util.find_spec("torch"), "Fast evaluator imports optional PyTorch")
class AttackCacheIntegrityTests(unittest.TestCase):
    def test_complete_cache_is_bound_to_source_and_ordered_plan(self) -> None:
        from rbafl.evaluation_fast import (
            CACHE_VERSION,
            _attack_plan_sha256,
            _cache_identity_worker,
            _completed_cache_status,
            _source_signature,
        )
        from rbafl.protocol import attack_plan

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.geojson"
            donor = root / "donor.geojson"
            source.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
            donor.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
            cache = root / "cache"
            cache.mkdir()
            plan = attack_plan()
            grid_size = 2
            np.save(cache / "attacks.npy", np.zeros((len(plan), 4, grid_size, grid_size), dtype=np.float32))
            np.save(cache / "clean.npy", np.zeros((4, grid_size, grid_size), dtype=np.float32))
            pd.DataFrame(
                [
                    {
                        "case_index": index,
                        "case_id": case.case_id,
                        "attack": case.attack,
                        "strength": case.strength,
                        "repeat_index": case.repeat_index,
                    }
                    for index, case in enumerate(plan)
                ]
            ).to_csv(cache / "attack_metadata.csv", index=False)
            payload = {
                "identity": "synthetic",
                "source_path": str(source),
                "cache_dir": str(cache),
                "external_path": str(donor),
                "grid_size": grid_size,
                "density_sigma": 3.0,
                "fixed_attack_seed": 20260910,
                "attack_plan_sha256": _attack_plan_sha256(),
                "source_signature": _source_signature(source),
                "external_signature": _source_signature(donor),
            }
            (cache / "complete.json").write_text(
                json.dumps(
                    {
                        "cache_version": CACHE_VERSION,
                        "fixed_attack_seed": payload["fixed_attack_seed"],
                        "density_sigma": payload["density_sigma"],
                        "attack_plan_sha256": payload["attack_plan_sha256"],
                        "source_signature": payload["source_signature"],
                        "external_signature": payload["external_signature"],
                    }
                ),
                encoding="utf-8",
            )
            valid, reason = _completed_cache_status(payload)
            self.assertTrue(valid, reason)
            self.assertEqual(_cache_identity_worker(payload)["status"], "reused")

            source.write_text('{"type":"FeatureCollection","features":[{}]}', encoding="utf-8")
            changed = dict(payload, source_signature=_source_signature(source))
            valid, reason = _completed_cache_status(changed)
            self.assertFalse(valid)
            self.assertEqual(reason, "source_signature_mismatch")


if __name__ == "__main__":
    unittest.main()
