from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


@unittest.skipUnless(importlib.util.find_spec("torch"), "Dataset class imports optional PyTorch")
class InternalHoldoutTests(unittest.TestCase):
    def test_triplet_pools_do_not_cross_internal_partition(self) -> None:
        from rbafl.data import PreparedTripletDataset

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for identity in ("a", "b"):
                identity_dir = root / "tensors" / identity
                identity_dir.mkdir(parents=True)
                for sample_type in ("base", "aug_001", "aug_002", "aug_003"):
                    tensor = identity_dir / f"{sample_type}.npy"
                    np.save(tensor, np.zeros((4, 2, 2), dtype=np.float32))
                    rows.append(
                        {
                            "identity": identity,
                            "source_path": f"{identity}.geojson",
                            "sample_type": sample_type,
                            "tensor_path": str(tensor),
                        }
                    )
            pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
            train = PreparedTripletDataset(root, ("occ",), "train", validation_per_identity=1)
            validation = PreparedTripletDataset(root, ("occ",), "val", validation_per_identity=1)
            for class_id in train.all_by_class:
                self.assertTrue(
                    set(train.all_by_class[class_id]).isdisjoint(validation.all_by_class[class_id])
                )


if __name__ == "__main__":
    unittest.main()
