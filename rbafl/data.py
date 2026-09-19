from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .protocol import CHANNEL_INDEX, protocol_dict
from .vector import (
    build_four_channels,
    discover_vector_files,
    random_training_attack,
    read_vector,
    stable_seed,
)


def prepare_dataset(
    source_root: str | Path,
    output_root: str | Path,
    grid_size: int = 256,
    density_sigma: float = 3.0,
    augmentations_per_identity: int = 8,
    seed: int = 20260730,
) -> pd.DataFrame:
    output = Path(output_root).expanduser().resolve()
    source_root_resolved = Path(source_root).expanduser().resolve()
    tensor_root = output / "tensors"
    tensor_root.mkdir(parents=True, exist_ok=True)
    samples = discover_vector_files(source_root)
    if len(samples) < 2:
        raise RuntimeError("At least two vector identities are required for uniqueness/triplet training")

    rows: List[Dict[str, object]] = []
    for sample in samples:
        source_relpath = sample.path.resolve().relative_to(source_root_resolved).as_posix()
        identity_dir = tensor_root / sample.identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        source_sha256 = vector_source_sha256(sample.path)
        t_read = time.perf_counter()
        gdf = read_vector(sample.path)
        read_s = time.perf_counter() - t_read

        t_build = time.perf_counter()
        tensor, meta = build_four_channels(gdf, grid_size=grid_size, density_sigma=density_sigma)
        build_s = time.perf_counter() - t_build
        base_path = identity_dir / "base.npy"
        np.save(base_path, tensor, allow_pickle=False)
        rows.append(
            {
                "identity": sample.identity,
                "source_path": source_relpath,
                "source_sha256": source_sha256,
                "sample_type": "base",
                "tensor_path": base_path.relative_to(output).as_posix(),
                "tensor_sha256": file_sha256(base_path),
                "augmentation_seed": "",
                "read_time_s": read_s,
                "channel_build_time_s": build_s,
                "tensor_bytes": int(tensor.nbytes),
                **meta,
            }
        )

        for index in range(augmentations_per_identity):
            aug_seed = stable_seed(seed, sample.identity, index)
            attacked, attack_meta = random_training_attack(gdf, aug_seed)
            t_aug = time.perf_counter()
            aug_tensor, aug_meta = build_four_channels(
                attacked,
                grid_size=grid_size,
                density_sigma=density_sigma,
            )
            aug_build_s = time.perf_counter() - t_aug
            aug_path = identity_dir / f"aug_{index + 1:03d}.npy"
            np.save(aug_path, aug_tensor, allow_pickle=False)
            rows.append(
                {
                    "identity": sample.identity,
                    "source_path": source_relpath,
                    "source_sha256": source_sha256,
                    "sample_type": f"aug_{index + 1:03d}",
                    "tensor_path": aug_path.relative_to(output).as_posix(),
                    "tensor_sha256": file_sha256(aug_path),
                    "augmentation_seed": aug_seed,
                    "read_time_s": 0.0,
                    "channel_build_time_s": aug_build_s,
                    "tensor_bytes": int(aug_tensor.nbytes),
                    **attack_meta,
                    **{f"channel_{k}": v for k, v in aug_meta.items()},
                }
            )

    manifest = pd.DataFrame(rows)
    manifest_path = output / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    info = {
        "manifest_schema_version": 3,
        "source_root_name": source_root_resolved.name,
        "grid_size": int(grid_size),
        "density_sigma": float(density_sigma),
        "augmentations_per_identity": int(augmentations_per_identity),
        "seed": int(seed),
        "identity_count": int(manifest["identity"].nunique()),
        "sample_count": int(len(manifest)),
        "protocol": protocol_dict(),
    }
    (output / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def resolve_prepared_tensor_path(
    prepared_root: str | Path,
    identity: str,
    sample_type: str,
    manifest_tensor_path: str | Path,
) -> Path:
    """Resolve a cached .npy path after moving a prepared cache across OSes.

    Older manifests may store stale workstation-specific absolute paths. In that
    case, reconstruct the canonical portable path
    from the *current* prepared root. The original manifest.csv is never changed.
    """
    root = Path(prepared_root).expanduser().resolve()
    candidate = Path(str(manifest_tensor_path))
    if candidate.is_file():
        return candidate.resolve()
    fallback = root / "tensors" / str(identity) / f"{sample_type}.npy"
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(
        "Prepared tensor not found. "
        f"manifest path={candidate}; portable fallback={fallback}"
    )


def prepared_manifest(prepared_root: str | Path) -> pd.DataFrame:
    root = Path(prepared_root).expanduser().resolve()
    path = root / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared manifest not found: {path}")
    df = pd.read_csv(path)
    required = {"identity", "source_path", "sample_type", "tensor_path"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Prepared manifest is missing columns: {', '.join(missing)}")
    df["identity"] = df["identity"].astype(str)
    df["sample_type"] = df["sample_type"].astype(str)

    resolved_paths: List[str] = []
    remapped = 0
    unresolved = 0
    for row in df[["identity", "sample_type", "tensor_path"]].to_dict("records"):
        candidate = Path(str(row["tensor_path"]))
        if candidate.is_file():
            resolved_paths.append(str(candidate.resolve()))
            continue
        fallback = root / "tensors" / str(row["identity"]) / f"{row['sample_type']}.npy"
        if fallback.is_file():
            resolved_paths.append(str(fallback.resolve()))
            remapped += 1
        else:
            resolved_paths.append(str(row["tensor_path"]))
            unresolved += 1
    df["tensor_path"] = resolved_paths
    df.attrs["tensor_path_remapped_count"] = int(remapped)
    df.attrs["tensor_path_unresolved_count"] = int(unresolved)
    return df


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector_source_sha256(path: str | Path) -> str:
    """Hash a vector source and, for Shapefiles, its defining sidecars.

    The digest is independent of the workstation path, so renamed byte-identical
    datasets can be rejected before an identity split is frozen.
    """

    source = Path(path).expanduser().resolve()
    files = [source]
    if source.suffix.lower() == ".shp":
        files.extend(
            candidate
            for extension in (".shx", ".dbf", ".prj", ".cpg")
            if (candidate := source.with_suffix(extension)).is_file()
        )
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.suffix.lower().encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("ascii"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def create_identity_split_exact(
    prepared_root: str | Path,
    output_csv: str | Path,
    *,
    split_seed: int = 20260909,
    train_count: int = 50,
    calibration_count: int = 20,
    test_count: int = 41,
    reuse_existing: bool = True,
) -> pd.DataFrame:
    """Create the frozen 50/20/41 identity-level study split.

    ``calibration`` is reserved exclusively for decision-threshold selection;
    ``test`` is reserved for final robustness, uniqueness, ablation, and timing
    results. This function rejects any identity count other than the exact
    requested total and refuses to reuse a
    split generated under a different protocol.
    """

    counts = {
        "train": int(train_count),
        "calibration": int(calibration_count),
        "test": int(test_count),
    }
    if any(value < 2 for value in counts.values()):
        raise ValueError("Every study split must contain at least two identities")

    manifest_path = Path(prepared_root).expanduser().resolve() / "manifest.csv"
    manifest = prepared_manifest(prepared_root)
    base = manifest[manifest["sample_type"] == "base"].copy()
    if base["identity"].duplicated().any():
        duplicated = sorted(base.loc[base["identity"].duplicated(), "identity"].unique())
        raise RuntimeError(
            "Each identity must have exactly one base sample; duplicates: "
            f"{duplicated[:10]}"
        )
    identities = sorted(base["identity"].astype(str).tolist())
    expected_total = int(sum(counts.values()))
    if len(identities) != expected_total:
        raise RuntimeError(
            f"The frozen protocol requires exactly {expected_total} identities "
            f"({train_count}/{calibration_count}/{test_count}); found {len(identities)}."
        )

    protocol_payload = {
        "schema_version": 1,
        "split_seed": int(split_seed),
        "counts": counts,
        "prepared_manifest_sha256": file_sha256(manifest_path),
        "identity_sha256": hashlib.sha256(
            "\n".join(identities).encode("utf-8")
        ).hexdigest(),
    }
    protocol_fingerprint = hashlib.sha256(
        json.dumps(protocol_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    output = Path(output_csv).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = output.with_suffix(".audit.json")
    if output.is_file() and reuse_existing:
        if not audit_path.is_file():
            raise RuntimeError(
                f"Refusing to reuse {output}: its protocol audit file is missing"
            )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("protocol_fingerprint") != protocol_fingerprint:
            raise RuntimeError(
                "Existing identity split was generated from a different seed, count "
                "protocol, identity list, or prepared manifest. Move it aside and rerun."
            )
        existing = pd.read_csv(output)
        _validate_exact_identity_split(existing, identities, counts)
        return existing.sort_values(["split_order", "identity"]).reset_index(drop=True)

    rng = np.random.default_rng(int(split_seed))
    shuffled = np.asarray(identities, dtype=object)[rng.permutation(len(identities))].tolist()
    boundaries = (
        ("train", shuffled[:train_count]),
        (
            "calibration",
            shuffled[train_count : train_count + calibration_count],
        ),
        ("test", shuffled[train_count + calibration_count :]),
    )
    rows: List[Dict[str, object]] = []
    split_order = 0
    for split_name, names in boundaries:
        for identity in names:
            rows.append(
                {
                    "identity": str(identity),
                    "split": split_name,
                    "split_order": int(split_order),
                    "split_seed": int(split_seed),
                    "protocol_fingerprint": protocol_fingerprint,
                }
            )
            split_order += 1

    result = pd.DataFrame(rows)
    _validate_exact_identity_split(result, identities, counts)
    result.to_csv(output, index=False, encoding="utf-8")
    audit = {
        **protocol_payload,
        "protocol_fingerprint": protocol_fingerprint,
        "identity_split_sha256": file_sha256(output),
        "split_labels": {
            "train": "encoder fitting only",
            "calibration": "decision-threshold selection only",
            "test": "final evaluation only",
        },
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _validate_exact_identity_split(
    split_df: pd.DataFrame,
    prepared_identities: Sequence[str],
    expected_counts: Dict[str, int],
) -> None:
    _validate_identity_split(split_df, prepared_identities)
    work = split_df.copy()
    work["split"] = work["split"].astype(str).str.lower()
    actual_counts = {str(k): int(v) for k, v in work["split"].value_counts().items()}
    if actual_counts != expected_counts:
        raise ValueError(
            f"Identity split counts do not match the frozen protocol: "
            f"expected={expected_counts}, actual={actual_counts}"
        )
    split_sets = {
        name: set(work.loc[work["split"] == name, "identity"].astype(str))
        for name in expected_counts
    }
    names = list(split_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(f"Study split leakage between {left} and {right}: {sorted(overlap)[:10]}")


def _validate_identity_split(split_df: pd.DataFrame, prepared_identities: Sequence[str]) -> None:
    required = {"identity", "split"}
    missing = required.difference(split_df.columns)
    if missing:
        raise ValueError(f"Identity split is missing columns: {', '.join(sorted(missing))}")
    work = split_df.copy()
    work["identity"] = work["identity"].astype(str)
    work["split"] = work["split"].astype(str).str.lower()
    valid_names = {"train", "calibration", "test"}
    unknown = sorted(set(work["split"]) - valid_names)
    if unknown:
        raise ValueError(f"Unknown split labels: {unknown}")
    if work["identity"].duplicated().any():
        raise ValueError("An identity occurs more than once in identity_split.csv")
    expected = set(map(str, prepared_identities))
    actual = set(work["identity"])
    if expected != actual:
        missing_ids = sorted(expected - actual)
        extra_ids = sorted(actual - expected)
        raise ValueError(
            f"Identity split does not match prepared dataset. missing={missing_ids[:10]}, extra={extra_ids[:10]}"
        )
    counts = work["split"].value_counts().to_dict()
    if counts.get("train", 0) < 2:
        raise ValueError("Training split must contain at least two identities")
    if counts.get("calibration", 0) < 2:
        raise ValueError("Calibration split must contain at least two identities for impostor-score calibration")
    if counts.get("test", 0) < 2:
        raise ValueError("Test split must contain at least two identities for uniqueness evaluation")


def load_identity_split(path: str | Path) -> pd.DataFrame:
    split_path = Path(path).expanduser().resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f"Identity split not found: {split_path}")
    df = pd.read_csv(split_path)
    required = {"identity", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Identity split is missing columns: {', '.join(sorted(missing))}")
    df["identity"] = df["identity"].astype(str)
    df["split"] = df["split"].astype(str).str.lower()
    return df


def identities_for_split(path: str | Path, split_name: str) -> List[str]:
    split_name = split_name.strip().lower()
    if split_name not in {"train", "calibration", "test"}:
        raise ValueError("split_name must be train, calibration, or test")
    df = load_identity_split(path)
    names = df.loc[df["split"] == split_name, "identity"].astype(str).tolist()
    if not names:
        raise RuntimeError(f"Identity split contains no identities for '{split_name}'")
    return names


def subset_manifest_by_identities(
    prepared_root: str | Path,
    identities: Iterable[str],
) -> pd.DataFrame:
    ids = set(map(str, identities))
    df = prepared_manifest(prepared_root)
    result = df[df["identity"].isin(ids)].copy()
    found = set(result["identity"].unique().tolist())
    missing = sorted(ids - found)
    if missing:
        raise KeyError(f"Prepared dataset is missing identities: {missing[:10]}")
    return result.reset_index(drop=True)


def _normalize_channels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[0] != 4:
        raise ValueError(f"Expected a four-channel (4,H,W) tensor, received {x.shape}")
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


class PreparedTripletDataset(Dataset):
    """Triplet dataset restricted to a caller-provided set of study identities.

    ``split='train'`` and ``split='val'`` are *internal optimisation splits* made
    only within the training identities.  They are deliberately separate from
    the study-level held-out ``calibration`` identities used for threshold
    calibration.
    """

    def __init__(
        self,
        prepared_root: str | Path,
        channel_names: Sequence[str],
        split: str,
        validation_per_identity: int = 1,
        seed: int = 20260730,
        allowed_identities: Optional[Sequence[str]] = None,
        tensor_cache: Optional[Dict[str, np.ndarray]] = None,
        preload_to_ram: bool = False,
    ) -> None:
        self.root = Path(prepared_root).expanduser().resolve()
        self.df = prepared_manifest(self.root)
        if allowed_identities is not None:
            allowed = set(map(str, allowed_identities))
            self.df = self.df[self.df["identity"].isin(allowed)].copy()
            missing = sorted(allowed - set(self.df["identity"].unique().tolist()))
            if missing:
                raise KeyError(f"Prepared dataset is missing allowed training identities: {missing[:10]}")
        self.channel_indices = tuple(CHANNEL_INDEX[x] for x in channel_names)
        self.split = split
        self.seed = int(seed)
        self.class_names = sorted(self.df["identity"].astype(str).unique().tolist())
        if len(self.class_names) < 2:
            raise RuntimeError("At least two training identities are required")
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        self.all_by_class: Dict[int, List[str]] = {}
        self.paths: List[Tuple[str, int]] = []
        self.tensor_cache = tensor_cache if tensor_cache is not None else {}

        for identity in self.class_names:
            cid = self.class_to_id[identity]
            sub = self.df[self.df["identity"].astype(str) == identity].copy()
            base = sub[sub["sample_type"] == "base"]["tensor_path"].astype(str).tolist()
            aug = sorted(sub[sub["sample_type"] != "base"]["tensor_path"].astype(str).tolist())
            if not base:
                raise RuntimeError(f"Identity {identity} has no base tensor")
            n_val = min(max(1, validation_per_identity), len(aug)) if aug else 0
            if split == "train":
                chosen = base + (aug[:-n_val] if n_val else aug)
            elif split == "val":
                chosen = aug[-n_val:] if n_val else base
            else:
                raise ValueError("split must be 'train' or 'val'")
            # Positives and negatives must come from the same internal partition
            # as the anchor. Otherwise a held-out augmentation can leak back into
            # training through triplet sampling even though it is not an anchor.
            self.all_by_class[cid] = list(chosen)
            self.paths.extend((path, cid) for path in chosen)

        if preload_to_ram:
            # The complete study training tensor set is typically < 1 GB.  On a
            # 134-GB machine it is much faster to load each .npy once per GPU
            # process than to reopen anchor/positive/negative files for 40 epochs.
            required = sorted({p for paths in self.all_by_class.values() for p in paths})
            for path in required:
                if path not in self.tensor_cache:
                    self.tensor_cache[path] = _normalize_channels(np.load(path, allow_pickle=False))

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, path: str) -> torch.Tensor:
        x4 = self.tensor_cache.get(path)
        if x4 is None:
            x4 = _normalize_channels(np.load(path, allow_pickle=False))
        selected = np.ascontiguousarray(x4[list(self.channel_indices)])
        return torch.from_numpy(selected)

    def __getitem__(self, index: int):
        anchor_path, cid = self.paths[index]
        rng = random.Random(stable_seed(self.seed, self.split, index))
        positive_candidates = [p for p in self.all_by_class[cid] if p != anchor_path]
        positive_path = rng.choice(positive_candidates or [anchor_path])
        negative_cid = rng.choice([x for x in self.all_by_class if x != cid])
        negative_path = rng.choice(self.all_by_class[negative_cid])
        return {
            "anchor": self._load(anchor_path),
            "positive": self._load(positive_path),
            "negative": self._load(negative_path),
            "class_id": torch.tensor(cid, dtype=torch.long),
        }


def base_manifest(
    prepared_root: str | Path,
    identity_split_path: str | Path | None = None,
    split_name: str | None = None,
) -> pd.DataFrame:
    df = prepared_manifest(prepared_root)
    base = df[df["sample_type"] == "base"].copy()
    if identity_split_path is not None or split_name is not None:
        if identity_split_path is None or split_name is None:
            raise ValueError("identity_split_path and split_name must be provided together")
        allowed = set(identities_for_split(identity_split_path, split_name))
        base = base[base["identity"].isin(allowed)].copy()
        if len(base) != len(allowed):
            found = set(base["identity"].astype(str).tolist())
            raise RuntimeError(
                f"Base manifest does not fully cover split '{split_name}'; missing={sorted(allowed - found)[:10]}"
            )
    return base.sort_values("identity").reset_index(drop=True)
