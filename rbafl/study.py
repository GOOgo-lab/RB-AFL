"""Reproducible 50/20/41 study protocol and result aggregation.

This module deliberately keeps path/configuration logic out of the algorithmic
modules.  Heavy GIS and PyTorch imports occur only inside stages that need them,
so split/statistics checks remain usable in lightweight CI environments.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


EXPECTED_SPLITS = {"train": 50, "calibration": 20, "test": 41}
EXPECTED_IDENTITY_COUNT = 111
EXPECTED_TEST_PAIRS_PER_SEED = math.comb(41, 2)
EXPECTED_TRAINING_SEEDS = tuple(range(20260730, 20260740))


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate a public-study JSON configuration."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    from .protocol import OBJECT_DELETE_MASK_REPEATS, OBJECT_DELETE_RATIOS, SCALE_FACTORS

    required = {
        "study",
        "paths",
        "data",
        "training",
        "calibration",
        "evaluation",
        "uniqueness",
        "efficiency",
        "watermark",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Configuration is missing sections: {missing}")

    study = config["study"]
    counts = {str(k): int(v) for k, v in study.get("split_counts", {}).items()}
    if counts != EXPECTED_SPLITS:
        raise ValueError(f"split_counts must be exactly {EXPECTED_SPLITS}; got {counts}")
    if int(study.get("expected_identities", -1)) != EXPECTED_IDENTITY_COUNT:
        raise ValueError("expected_identities must be 111")
    seeds = tuple(int(x) for x in study.get("training_seeds", []))
    if seeds != EXPECTED_TRAINING_SEEDS:
        raise ValueError(
            "training_seeds must be the ten frozen values 20260730..20260739"
        )
    if int(study.get("split_seed", -1)) in seeds:
        raise ValueError("split_seed must be independent of all training seeds")

    watermark = config["watermark"]
    bit_length = int(watermark.get("bit_length", 0))
    required_ones = int(watermark.get("required_one_bits", -1))
    required_zeros = int(watermark.get("required_zero_bits", -1))
    if bit_length != 256 or required_ones != 45 or required_zeros != 211:
        raise ValueError(
            "The frozen eligible-watermark protocol requires 256 bits with exactly 45 one bits and 211 zero bits (manuscript Figure 4)"
        )
    if str(watermark.get("quantization", "")) not in {"median", "mean", "zero"}:
        raise ValueError("Choose an elementwise threshold: median, mean or zero")
    if int(config["training"].get("embedding_dim", 0)) != bit_length:
        raise ValueError("Embedding dimension must equal watermark length")
    if int(config["data"].get("grid_size", 0)) != 256:
        raise ValueError("Manuscript tensor size is 256 x 256")

    scale_factors = [float(x) for x in config["evaluation"].get("uniform_scale_factors", [])]
    if not scale_factors or 0.0 in scale_factors or 1.0 not in scale_factors:
        raise ValueError("uniform scale factors must include clean factor 1.0 and must not include 0")
    if tuple(scale_factors) != tuple(float(x) for x in SCALE_FACTORS):
        raise ValueError("Configured uniform scale factors do not match rbafl.protocol")
    deletion = [float(x) for x in config["evaluation"].get("object_delete_ratios", [])]
    if max(deletion, default=0.0) < 0.50:
        raise ValueError("object deletion protocol must extend to at least 50%")
    if tuple(deletion) != tuple(float(x) for x in OBJECT_DELETE_RATIOS):
        raise ValueError("Configured object-deletion ratios do not match rbafl.protocol")
    if int(config["evaluation"].get("object_delete_mask_repeats", -1)) != OBJECT_DELETE_MASK_REPEATS:
        raise ValueError(
            "Configured object-deletion mask repeats do not match rbafl.protocol"
        )

    target_far = float(config["calibration"].get("target_far", -1))
    if not 0.0 <= target_far <= 1.0:
        raise ValueError("calibration.target_far must be in [0,1]")
    if str(config["calibration"].get("split")) != "calibration":
        raise ValueError("Threshold selection may use only the calibration split")
    if str(config["calibration"].get("experiment")) != "E5":
        raise ValueError("Threshold calibration must use the proposed E5 model")
    if (config["calibration"].get("policy") != "predeclared_fixed"
            or config["calibration"].get("fixed_threshold") != 0.75):
        raise ValueError("Manuscript protocol predeclares T_NC=0.75")
    if str(config["evaluation"].get("split")) != "test":
        raise ValueError("Final evaluation may use only the test split")
    uniqueness = config["uniqueness"]
    if (
        str(uniqueness.get("split")) != "test"
        or str(uniqueness.get("experiment")) != "E5"
        or str(uniqueness.get("pair_mode")) != "unordered"
        or str(uniqueness.get("directed_pair_reduction")) != "max"
        or int(uniqueness.get("expected_pairs_per_seed", -1)) != EXPECTED_TEST_PAIRS_PER_SEED
        or int(uniqueness.get("expected_total_pair_scores", -1)) != 8200
    ):
        raise ValueError("Uniqueness protocol must be E5/test with 820 unordered pairs per seed and 8,200 total")
    efficiency = config["efficiency"]
    if str(efficiency.get("split")) != "test" or str(efficiency.get("experiment")) != "E5":
        raise ValueError("Efficiency benchmark must use E5 on the test split")
    if int(efficiency.get("warmup_repeats", -1)) < 0 or int(efficiency.get("measured_repeats", 0)) < 1:
        raise ValueError("Efficiency repeats must include at least one measured repetition")


def config_paths(config: Mapping[str, Any]) -> dict[str, Path | None]:
    config_path = Path(str(config["_config_path"])).resolve()
    base = config_path.parent

    def resolve(value: object, *, optional: bool = False) -> Path | None:
        text = str(value or "").strip()
        if not text:
            return None if optional else base
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        return candidate.resolve()

    paths = config["paths"]
    run_root = resolve(paths["run_root"])
    assert run_root is not None
    return {
        "source_root": resolve(paths["source_root"]),
        "watermark": resolve(paths["watermark"]),
        "external_vector": resolve(paths.get("external_vector", ""), optional=True),
        "run_root": run_root,
        "prepared": run_root / "prepared",
        "split": run_root / "split" / "identity_split.csv",
        "source_map": run_root / "manifest" / "resolved_source_paths.csv",
        "models": run_root / "models",
        "calibration": run_root / "calibration",
        "threshold": run_root / "calibration" / "threshold_summary.json",
        "attack_cache": run_root / "attack_cache",
        "evaluation": run_root / "evaluation",
        "uniqueness": run_root / "uniqueness",
        "efficiency": run_root / "efficiency",
        "reports": run_root / "reports",
        "figures": run_root / "figures",
    }


def protocol_lock(config: Mapping[str, Any], split_path: str | Path | None = None) -> dict[str, Any]:
    from .protocol import protocol_dict

    payload: dict[str, Any] = {
        "schema_version": 1,
        "expected_identities": EXPECTED_IDENTITY_COUNT,
        "split_counts": EXPECTED_SPLITS,
        "split_seed": int(config["study"]["split_seed"]),
        "training_seeds": list(EXPECTED_TRAINING_SEEDS),
        "bit_length": int(config["watermark"]["bit_length"]),
        "required_one_bits": int(config["watermark"]["required_one_bits"]),
        "quantization": str(config["watermark"]["quantization"]),
        "threshold_policy": str(config["calibration"]["policy"]),
        "target_far": float(config["calibration"]["target_far"]),
        "fixed_attack_seed": int(config["evaluation"]["fixed_attack_seed"]),
        "attack_protocol_sha256": _json_sha256(protocol_dict()["attacks"]),
        "config_sha256": str(config["_config_sha256"]),
    }
    if split_path is not None and Path(split_path).is_file():
        payload["identity_split_sha256"] = hashlib.sha256(Path(split_path).read_bytes()).hexdigest()
    payload["protocol_fingerprint"] = _json_sha256(payload)
    return payload


def write_protocol_lock(config: Mapping[str, Any]) -> Path:
    paths = config_paths(config)
    run_root = paths["run_root"]
    split_path = paths["split"]
    assert isinstance(run_root, Path) and isinstance(split_path, Path)
    run_root.mkdir(parents=True, exist_ok=True)
    output = run_root / "protocol_lock.json"
    payload = protocol_lock(config, split_path)
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("protocol_fingerprint") != payload.get("protocol_fingerprint"):
            raise RuntimeError(
                "Run directory contains artifacts from a different protocol/config. "
                "Use a new run_root rather than mixing results."
            )
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def prepare_stage(config: Mapping[str, Any]) -> Path:
    from .data import (
        file_sha256,
        prepare_dataset,
        resolve_prepared_tensor_path,
        vector_source_sha256,
    )

    paths = config_paths(config)
    source_root = paths["source_root"]
    prepared = paths["prepared"]
    assert isinstance(source_root, Path) and isinstance(prepared, Path)
    manifest_path = prepared / "manifest.csv"
    info_path = prepared / "dataset_info.json"
    if manifest_path.is_file() and info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        expected = {
            "manifest_schema_version": 3,
            "source_root_name": source_root.name,
            "grid_size": int(config["data"]["grid_size"]),
            "density_sigma": float(config["data"]["density_sigma"]),
            "augmentations_per_identity": int(config["data"]["augmentations_per_identity"]),
            "seed": int(config["data"]["preparation_seed"]),
            "identity_count": EXPECTED_IDENTITY_COUNT,
        }
        actual = {key: info.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                "Existing prepared cache does not match the frozen data protocol. "
                "Use a new run_root rather than overwriting it."
            )
        manifest = pd.read_csv(manifest_path).fillna("")
        columns = set(manifest.columns)
        if not {"source_sha256", "tensor_sha256"}.issubset(columns):
            raise RuntimeError(
                "Existing prepared manifest predates content hashing. Use a new run_root."
            )
        base = manifest[manifest["sample_type"].astype(str) == "base"]
        for row in base.to_dict("records"):
            source = Path(str(row["source_path"]))
            if not source.is_absolute():
                source = source_root / source
            if not source.is_file() or vector_source_sha256(source) != str(row["source_sha256"]):
                raise RuntimeError(
                    f"Raw vector content changed after preparation: {row['identity']}"
                )
        for row in manifest.to_dict("records"):
            tensor = resolve_prepared_tensor_path(
                prepared,
                str(row["identity"]),
                str(row["sample_type"]),
                str(row["tensor_path"]),
            )
            if file_sha256(tensor) != str(row["tensor_sha256"]):
                raise RuntimeError(
                    f"Prepared tensor content changed after preparation: {row['identity']} / {row['sample_type']}"
                )
        return manifest_path
    manifest = prepare_dataset(
        source_root,
        prepared,
        grid_size=int(config["data"]["grid_size"]),
        density_sigma=float(config["data"]["density_sigma"]),
        augmentations_per_identity=int(config["data"]["augmentations_per_identity"]),
        seed=int(config["data"]["preparation_seed"]),
    )
    base = manifest[manifest["sample_type"].astype(str) == "base"]
    if len(base) != EXPECTED_IDENTITY_COUNT or base["identity"].nunique() != EXPECTED_IDENTITY_COUNT:
        raise RuntimeError(
            f"Frozen protocol requires exactly 111 independent identities; found {len(base)}"
        )
    if base["source_sha256"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate source bundles detected before splitting")
    return manifest_path


def create_exact_split_from_manifest(
    manifest_path: str | Path,
    output_csv: str | Path,
    *,
    split_seed: int,
    reuse_existing: bool = True,
) -> pd.DataFrame:
    """Pure-Pandas exact splitter used by both CI and the public pipeline."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = pd.read_csv(manifest_path)
    required = {"identity", "sample_type"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Prepared manifest is missing {sorted(required - set(manifest.columns))}")
    base = manifest[manifest["sample_type"].astype(str) == "base"].copy()
    base["identity"] = base["identity"].astype(str)
    if base["identity"].duplicated().any():
        raise ValueError("Every identity must have exactly one base row")
    if "source_path" in base.columns:
        normalized_sources = base["source_path"].astype(str).str.strip()
        duplicated_sources = normalized_sources[normalized_sources.duplicated(keep=False)]
        if not duplicated_sources.empty:
            raise ValueError(
                "The same source path is assigned to multiple identities; resolve duplicate data before splitting"
            )
    if "source_sha256" in base.columns:
        hashes = base["source_sha256"].fillna("").astype(str).str.strip()
        nonempty = hashes[hashes != ""]
        if nonempty.duplicated(keep=False).any():
            raise ValueError(
                "The same source SHA-256 occurs under multiple identities; resolve duplicate data before splitting"
            )
    identities = sorted(base["identity"].tolist())
    if len(identities) != EXPECTED_IDENTITY_COUNT:
        raise RuntimeError(
            f"Frozen protocol requires exactly 111 identities; found {len(identities)}"
        )
    protocol = {
        "schema_version": 1,
        "split_seed": int(split_seed),
        "split_counts": EXPECTED_SPLITS,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "identity_sha256": hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest(),
    }
    fingerprint = _json_sha256(protocol)
    output = Path(output_csv).expanduser().resolve()
    audit_path = output.with_suffix(".audit.json")
    if output.is_file() and reuse_existing:
        if not audit_path.is_file():
            raise RuntimeError("Existing split has no audit file and cannot be safely reused")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError("Existing split does not match the frozen manifest/seed/count protocol")
        validate_split_file(output)
        return pd.read_csv(output)

    rng = np.random.default_rng(int(split_seed))
    shuffled = np.asarray(identities, dtype=object)[rng.permutation(len(identities))].tolist()
    boundaries = (
        ("train", shuffled[:50]),
        ("calibration", shuffled[50:70]),
        ("test", shuffled[70:]),
    )
    rows: list[dict[str, Any]] = []
    order = 0
    for split_name, names in boundaries:
        for identity in names:
            rows.append(
                {
                    "identity": str(identity),
                    "split": split_name,
                    "split_order": int(order),
                    "split_seed": int(split_seed),
                    "protocol_fingerprint": fingerprint,
                }
            )
            order += 1
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8")
    report = validate_split_file(output)
    audit = {
        **protocol,
        "protocol_fingerprint": fingerprint,
        "identity_split_sha256": report["sha256"],
        "split_labels": {
            "train": "encoder fitting only",
            "calibration": "threshold selection only",
            "test": "final evaluation only",
        },
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def split_stage(config: Mapping[str, Any], *, reuse_existing: bool = True) -> Path:
    paths = config_paths(config)
    prepared = paths["prepared"]
    split_path = paths["split"]
    assert isinstance(prepared, Path) and isinstance(split_path, Path)
    create_exact_split_from_manifest(
        prepared / "manifest.csv",
        split_path,
        split_seed=int(config["study"]["split_seed"]),
        reuse_existing=reuse_existing,
    )
    write_source_map(config)
    write_protocol_lock(config)
    return split_path


def validate_split_file(path: str | Path) -> dict[str, Any]:
    split = pd.read_csv(path)
    required = {"identity", "split"}
    if not required.issubset(split.columns):
        raise ValueError(f"identity_split.csv must contain {sorted(required)}")
    split["identity"] = split["identity"].astype(str)
    split["split"] = split["split"].astype(str).str.lower()
    if len(split) != EXPECTED_IDENTITY_COUNT or split["identity"].nunique() != EXPECTED_IDENTITY_COUNT:
        raise ValueError("The split must contain exactly 111 unique identities")
    counts = {str(k): int(v) for k, v in split["split"].value_counts().items()}
    if counts != EXPECTED_SPLITS:
        raise ValueError(f"Expected split counts {EXPECTED_SPLITS}; got {counts}")
    sets = {
        name: set(split.loc[split["split"] == name, "identity"])
        for name in EXPECTED_SPLITS
    }
    if any(sets[a] & sets[b] for a in sets for b in sets if a < b):
        raise ValueError("Identity leakage detected between study splits")
    return {
        "identity_count": EXPECTED_IDENTITY_COUNT,
        "counts": counts,
        "disjoint": True,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def write_source_map(config: Mapping[str, Any]) -> Path:
    paths = config_paths(config)
    prepared = paths["prepared"]
    source_root = paths["source_root"]
    output = paths["source_map"]
    assert isinstance(prepared, Path) and isinstance(source_root, Path) and isinstance(output, Path)
    manifest = pd.read_csv(prepared / "manifest.csv")
    base = manifest[manifest["sample_type"].astype(str) == "base"].copy()
    rows = []
    for row in base.to_dict("records"):
        source_value = Path(str(row["source_path"])).expanduser()
        source = (source_value if source_value.is_absolute() else source_root / source_value).resolve()
        rows.append(
            {
                "identity": str(row["identity"]),
                "original_source_path": str(row["source_path"]),
                "resolved_source_path": str(source),
                "status": "resolved" if source.is_file() else "missing",
            }
        )
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8")
    return output


def _training_config(config: Mapping[str, Any], seed: int):
    from .model import TrainingConfig

    item = config["training"]
    return TrainingConfig(
        epochs=int(item["epochs"]),
        batch_size=int(item["batch_size"]),
        learning_rate=float(item["learning_rate"]),
        weight_decay=float(item["weight_decay"]),
        embedding_dim=int(item["embedding_dim"]),
        triplet_margin=float(item["triplet_margin"]),
        validation_per_identity=int(item["internal_validation_per_identity"]),
        num_workers=int(item["num_workers"]),
        seed=int(seed),
        device=str(item["device"]),
        preload_tensors_to_ram=bool(item["preload_tensors_to_ram"]),
        pin_memory=bool(item["pin_memory"]),
        persistent_workers=bool(item["persistent_workers"]),
        prefetch_factor=int(item["prefetch_factor"]),
    )


def validate_checkpoint_provenance(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    expected_seed: int,
    expected_exp_id: str,
) -> dict[str, Any]:
    """Reject a copied or stale checkpoint before calibration/test access."""

    from dataclasses import asdict

    import torch

    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "rbafl_geometry_encoder_v1.0.0":
        raise RuntimeError(f"Unsupported public checkpoint format: {path}")
    if str(checkpoint.get("experiment", {}).get("exp_id", "")) != expected_exp_id:
        raise RuntimeError(f"Checkpoint experiment mismatch for {path}")
    if checkpoint.get("training_config") != asdict(_training_config(config, expected_seed)):
        raise RuntimeError(f"Checkpoint training configuration/seed mismatch for {path}")

    paths = config_paths(config)
    split_path = paths["split"]
    prepared = paths["prepared"]
    assert isinstance(split_path, Path) and isinstance(prepared, Path)
    split_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()
    prepared_sha256 = hashlib.sha256((prepared / "manifest.csv").read_bytes()).hexdigest()
    study_split = checkpoint.get("study_split", {})
    if str(study_split.get("identity_split_sha256", "")) != split_sha256:
        raise RuntimeError(f"Checkpoint identity split mismatch for {path}")
    if str(study_split.get("prepared_manifest_sha256", "")) != prepared_sha256:
        raise RuntimeError(f"Checkpoint prepared-data manifest mismatch for {path}")
    if str(study_split.get("study_train_split", "")) != "train":
        raise RuntimeError(f"Checkpoint was not trained on the train split: {path}")
    if int(study_split.get("training_identity_count", -1)) != 50:
        raise RuntimeError(f"Checkpoint does not contain exactly 50 training identities: {path}")
    split = pd.read_csv(split_path)
    expected_classes = sorted(
        split.loc[split["split"].astype(str) == "train", "identity"].astype(str).tolist()
    )
    if sorted(str(value) for value in checkpoint.get("class_names", [])) != expected_classes:
        raise RuntimeError(f"Checkpoint class identities do not match the frozen train split: {path}")
    return checkpoint


def train_stage(
    config: Mapping[str, Any],
    *,
    seeds: Sequence[int] | None = None,
    experiments: Sequence[str] | None = None,
) -> list[Path]:
    from .model import train_all_ablations

    prepare_stage(config)
    paths = config_paths(config)
    prepared = paths["prepared"]
    split_path = paths["split"]
    models = paths["models"]
    assert isinstance(prepared, Path) and isinstance(split_path, Path) and isinstance(models, Path)
    validate_split_file(split_path)
    write_protocol_lock(config)
    chosen_seeds = [int(x) for x in (seeds or EXPECTED_TRAINING_SEEDS)]
    if not set(chosen_seeds).issubset(EXPECTED_TRAINING_SEEDS):
        raise ValueError("Unknown training seed")
    outputs: list[Path] = []
    for seed in chosen_seeds:
        outputs.extend(
            train_all_ablations(
                prepared,
                models / f"seed_{seed}",
                _training_config(config, seed),
                selected=experiments,
                identity_split_path=split_path,
                study_train_split="train",
                reuse_existing=True,
            )
        )
    return outputs


def _e5_checkpoints(config: Mapping[str, Any]) -> dict[int, Path]:
    paths = config_paths(config)
    models = paths["models"]
    assert isinstance(models, Path)
    result = {
        seed: models / f"seed_{seed}" / "E5_proposed" / "best.pt"
        for seed in EXPECTED_TRAINING_SEEDS
    }
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Threshold calibration/efficiency requires all ten E5 checkpoints; "
            f"first missing: {missing[0]}"
        )
    for seed, path in result.items():
        validate_checkpoint_provenance(
            config,
            path,
            expected_seed=seed,
            expected_exp_id="E5",
        )
    return result


def _validate_eligible_watermark(config: Mapping[str, Any]) -> np.ndarray:
    from .watermark import watermark_image_to_bits

    paths = config_paths(config)
    watermark = paths["watermark"]
    assert isinstance(watermark, Path)
    bits, _, _ = watermark_image_to_bits(watermark, int(config["watermark"]["bit_length"]))
    ones = int(bits.sum())
    required = int(config["watermark"]["required_one_bits"])
    if ones != required:
        raise ValueError(
            f"Ineligible copyright watermark: expected exactly {required} one bits, got {ones}. "
            "Use the original 45-one / 211-zero manuscript watermark; do not rebalance it."
        )
    return bits


def calibrate_stage(config: Mapping[str, Any]) -> Path:
    from .data import file_sha256
    from .calibration import calibrate_threshold, collect_calibration_scores
    from .watermark import bit_sha256

    prepare_stage(config)
    paths = config_paths(config)
    prepared = paths["prepared"]
    split_path = paths["split"]
    watermark_path = paths["watermark"]
    output = paths["calibration"]
    assert all(
        isinstance(x, Path)
        for x in (prepared, split_path, watermark_path, output)
    )
    split_report = validate_split_file(split_path)
    write_protocol_lock(config)
    watermark_bits = _validate_eligible_watermark(config)
    checkpoints = _e5_checkpoints(config)
    output.mkdir(parents=True, exist_ok=True)
    scores = collect_calibration_scores(
        prepared,
        split_path,
        checkpoints,
        watermark_path,
        output,
        bit_length=int(config["watermark"]["bit_length"]),
        threshold_mode=str(config["watermark"]["quantization"]),
        device=str(config["training"]["device"]),
        random_watermark_patterns=int(config["calibration"]["random_watermark_patterns"]),
        watermark_pattern_seed=int(config["calibration"]["watermark_pattern_seed"]),
    )
    summary = calibrate_threshold(
        scores,
        output,
        target_far=float(config["calibration"]["target_far"]),
        threshold_step=float(config["calibration"]["threshold_step"]),
        fixed_threshold=float(config["calibration"]["fixed_threshold"]),
    )
    summary.update(
        {
            "split_name": "calibration",
            "calibration_identity_count": 20,
            "identity_split_sha256": split_report["sha256"],
            "watermark_sha256": bit_sha256(watermark_bits),
            "checkpoint_sha256_by_seed": {
                str(seed): file_sha256(path) for seed, path in checkpoints.items()
            },
            "test_data_used_for_threshold_selection": False,
            "quantization": str(config["watermark"]["quantization"]),
            "protocol_fingerprint": protocol_lock(config, split_path)["protocol_fingerprint"],
            "far_95_one_sided_upper": one_sided_binomial_upper(
                int(summary["false_accept_count"]), int(summary["impostor_score_count"])
            ),
            "frr_95_one_sided_upper": one_sided_binomial_upper(
                int(summary["false_reject_count"]), int(summary["genuine_score_count"])
            ),
            "dependence_note": (
                "Score-level binomial intervals do not model dependence among samples "
                "sharing an identity or training seed; retain raw rows for clustered analysis."
            ),
        }
    )
    threshold_path = output / "threshold_summary.json"
    threshold_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = hashlib.sha256(threshold_path.read_bytes()).hexdigest()
    (output / "threshold_lock.sha256").write_text(f"{digest}  threshold_summary.json\n", encoding="utf-8")
    return threshold_path


def load_frozen_threshold(config: Mapping[str, Any]) -> float:
    from .data import file_sha256
    from .watermark import bit_sha256

    path = config_paths(config)["threshold"]
    assert isinstance(path, Path)
    if not path.is_file():
        raise FileNotFoundError("Run the calibration stage before accessing test data")
    lock_path = path.parent / "threshold_lock.sha256"
    if not lock_path.is_file():
        raise RuntimeError("Frozen threshold checksum is missing")
    expected_digest = lock_path.read_text(encoding="utf-8").strip().split()[0]
    actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_digest != actual_digest:
        raise RuntimeError("Frozen threshold checksum verification failed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split_name") != "calibration" or payload.get("test_data_used_for_threshold_selection") is not False:
        raise RuntimeError("Threshold artifact is not a calibration-only frozen threshold")
    if payload.get("selection_policy") != config["calibration"]["policy"]:
        raise RuntimeError("Frozen threshold selection policy does not match the protocol")
    if not math.isclose(
        float(payload.get("target_far", float("nan"))),
        float(config["calibration"]["target_far"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("Frozen threshold FAR target does not match the protocol")

    paths = config_paths(config)
    split_path = paths["split"]
    assert isinstance(split_path, Path)
    if payload.get("identity_split_sha256") != hashlib.sha256(split_path.read_bytes()).hexdigest():
        raise RuntimeError("Frozen threshold was calibrated on a different identity split")
    watermark_bits = _validate_eligible_watermark(config)
    if payload.get("watermark_sha256") != bit_sha256(watermark_bits):
        raise RuntimeError("Frozen threshold was calibrated with a different watermark")
    checkpoints = _e5_checkpoints(config)
    current_checkpoint_hashes = {
        str(seed): file_sha256(checkpoint) for seed, checkpoint in checkpoints.items()
    }
    if payload.get("checkpoint_sha256_by_seed") != current_checkpoint_hashes:
        raise RuntimeError("Frozen threshold was calibrated with different E5 checkpoints")
    if payload.get("protocol_fingerprint") != protocol_lock(config, split_path)["protocol_fingerprint"]:
        raise RuntimeError("Threshold artifact uses a different protocol")
    value = float(payload["selected_threshold"])
    if value != float(config["calibration"]["fixed_threshold"]):
        raise ValueError("Threshold differs from the predeclared manuscript value")
    if not 0.0 <= value <= 1.0:
        raise ValueError("Frozen threshold is outside [0,1]")
    return value


def cache_attacks_stage(config: Mapping[str, Any]) -> Path:
    from .evaluation_fast import build_fixed_test_attack_cache

    paths = config_paths(config)
    prepared = paths["prepared"]
    split_path = paths["split"]
    source_map = paths["source_map"]
    attack_cache = paths["attack_cache"]
    external = paths["external_vector"]
    assert all(isinstance(x, Path) for x in (prepared, split_path, source_map, attack_cache))
    validate_split_file(split_path)
    write_protocol_lock(config)
    if external is not None and not external.is_file():
        raise FileNotFoundError(
            "The full attack protocol includes merge/object-add attacks; set paths.external_vector "
            "to a licensed donor vector dataset."
        )
    evaluation = config["evaluation"]
    build_fixed_test_attack_cache(
        prepared,
        attack_cache,
        identity_split_path=split_path,
        source_path_map=source_map,
        study_split_name="test",
        external_vector_path=external,
        grid_size=int(config["data"]["grid_size"]),
        density_sigma=float(config["data"]["density_sigma"]),
        fixed_attack_seed=int(evaluation["fixed_attack_seed"]),
        workers=int(evaluation["attack_cache_workers"]),
        shards_per_identity=int(evaluation["attack_cache_shards_per_identity"]),
        max_shards_per_identity=int(evaluation["attack_cache_max_shards_per_identity"]),
        longtail_cases_per_shard=int(evaluation["attack_cache_longtail_cases_per_shard"]),
        source_cache_workers=int(evaluation["source_cache_workers"]),
        import_legacy_cache=False,
        noise_index_workers=int(evaluation["noise_index_workers"]),
        min_available_memory_gb=float(evaluation["min_available_memory_gb"]),
        max_heavy_inflight=int(evaluation["max_heavy_inflight"]),
    )
    return attack_cache


def evaluate_stage(config: Mapping[str, Any], *, seeds: Sequence[int] | None = None) -> list[Path]:
    from .evaluation_fast import evaluate_all_cached

    paths = config_paths(config)
    prepared = paths["prepared"]
    split_path = paths["split"]
    source_map = paths["source_map"]
    models = paths["models"]
    attack_cache = paths["attack_cache"]
    evaluation_root = paths["evaluation"]
    watermark = paths["watermark"]
    assert all(isinstance(x, Path) for x in (prepared, split_path, source_map, models, attack_cache, evaluation_root, watermark))
    validate_split_file(split_path)
    write_protocol_lock(config)
    threshold = load_frozen_threshold(config)
    chosen = [int(x) for x in (seeds or EXPECTED_TRAINING_SEEDS)]
    outputs: list[Path] = []
    for seed in chosen:
        seed_models = models / f"seed_{seed}"
        checkpoint_files = sorted(seed_models.glob("E*_*/best.pt"))
        if len(checkpoint_files) != 7:
            raise FileNotFoundError(
                f"Expected seven E1-E7 checkpoints for seed {seed}; found {len(checkpoint_files)}"
            )
        for path in checkpoint_files:
            expected_exp_id = path.parent.name.split("_", 1)[0]
            validate_checkpoint_provenance(
                config,
                path,
                expected_seed=seed,
                expected_exp_id=expected_exp_id,
            )
        seed_output = evaluation_root / f"seed_{seed}"
        fingerprint_payload = {
            "schema_version": 1,
            "training_seed": int(seed),
            "config_sha256": str(config["_config_sha256"]),
            "identity_split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "threshold_sha256": hashlib.sha256(paths["threshold"].read_bytes()).hexdigest(),
            "watermark_sha256": hashlib.sha256(watermark.read_bytes()).hexdigest(),
            "attack_cache_manifest_sha256": hashlib.sha256(
                (attack_cache / "cache_manifest.json").read_bytes()
            ).hexdigest(),
            "checkpoints": {
                path.parent.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in checkpoint_files
            },
        }
        fingerprint_payload["fingerprint"] = _json_sha256(fingerprint_payload)
        fingerprint_path = seed_output / "evaluation_fingerprint.json"
        if fingerprint_path.is_file():
            existing = json.loads(fingerprint_path.read_text(encoding="utf-8"))
            if existing.get("fingerprint") != fingerprint_payload["fingerprint"]:
                raise RuntimeError(
                    f"Refusing to resume seed {seed}: checkpoint, split, watermark, threshold, "
                    "attack cache, or config changed. Use a new run_root."
                )
        else:
            seed_output.mkdir(parents=True, exist_ok=True)
            fingerprint_path.write_text(
                json.dumps(fingerprint_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        result = evaluate_all_cached(
            prepared,
            seed_models,
            watermark,
            seed_output,
            attack_cache,
            selected=None,
            grid_size=int(config["data"]["grid_size"]),
            density_sigma=float(config["data"]["density_sigma"]),
            bit_length=int(config["watermark"]["bit_length"]),
            threshold_mode=str(config["watermark"]["quantization"]),
            nc_threshold=threshold,
            seed=seed,
            device=str(config["training"]["device"]),
            timing_repeats=int(config["evaluation"]["timing_repeats"]),
            identity_split_path=split_path,
            study_split_name="test",
            source_path_map=source_map,
            eval_batch_size=int(config["evaluation"]["eval_batch_size"]),
            fixed_attack_seed=int(config["evaluation"]["fixed_attack_seed"]),
            resume=True,
        )
        outputs.extend(Path(path) for path in result.values())
    return outputs


def one_sided_binomial_upper(successes: int, total: int, confidence: float = 0.95) -> float:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    if successes == total:
        return 1.0
    if successes == 0:
        return float(1.0 - (1.0 - confidence) ** (1.0 / total))
    return float(stats.beta.ppf(confidence, successes + 1, total - successes))


def collapse_directed_uniqueness(
    rows: pd.DataFrame,
    *,
    threshold: float,
    expected_seeds: Sequence[int] = EXPECTED_TRAINING_SEEDS,
    symmetry_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Collapse A->B/B->A diagnostics into conservative unordered pair scores."""

    required = {"training_seed", "registered_identity", "tested_identity", "nc"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Uniqueness rows are missing columns: {sorted(missing)}")
    work = rows.copy()
    work["training_seed"] = work["training_seed"].astype(int)
    work["registered_identity"] = work["registered_identity"].astype(str)
    work["tested_identity"] = work["tested_identity"].astype(str)
    work = work[work["registered_identity"] != work["tested_identity"]].copy()
    if work["nc"].isna().any() or ((work["nc"].astype(float) < 0) | (work["nc"].astype(float) > 1)).any():
        raise ValueError("NC values must be finite and in [0,1]")
    work["identity_a"] = work.apply(
        lambda row: min(row["registered_identity"], row["tested_identity"]), axis=1
    )
    work["identity_b"] = work.apply(
        lambda row: max(row["registered_identity"], row["tested_identity"]), axis=1
    )
    work["pair_id"] = work["identity_a"] + "::" + work["identity_b"]

    output_rows: list[dict[str, Any]] = []
    for (seed, pair_id), group in work.groupby(["training_seed", "pair_id"], sort=True):
        if len(group) != 2:
            raise RuntimeError(f"Expected both directions for seed={seed}, pair={pair_id}; got {len(group)}")
        a = str(group.iloc[0]["identity_a"])
        b = str(group.iloc[0]["identity_b"])
        forward = group[
            (group["registered_identity"] == a) & (group["tested_identity"] == b)
        ]
        reverse = group[
            (group["registered_identity"] == b) & (group["tested_identity"] == a)
        ]
        if len(forward) != 1 or len(reverse) != 1:
            raise RuntimeError(f"Malformed directed pair for seed={seed}, pair={pair_id}")
        nc_ab = float(forward.iloc[0]["nc"])
        nc_ba = float(reverse.iloc[0]["nc"])
        delta = abs(nc_ab - nc_ba)
        if delta > symmetry_tolerance:
            raise RuntimeError(
                f"Uniqueness symmetry failed for seed={seed}, pair={pair_id}: "
                f"|{nc_ab}-{nc_ba}|={delta}"
            )
        pair_nc = max(nc_ab, nc_ba)
        output_rows.append(
            {
                "training_seed": int(seed),
                "identity_a": a,
                "identity_b": b,
                "pair_id": pair_id,
                "nc_a_to_b": nc_ab,
                "nc_b_to_a": nc_ba,
                "direction_delta": delta,
                "nc": pair_nc,
                "pair_reduction": "max",
                "threshold": float(threshold),
                "accepted": bool(pair_nc >= threshold),
            }
        )

    result = pd.DataFrame(output_rows)
    expected_seed_set = {int(x) for x in expected_seeds}
    if set(result["training_seed"].unique()) != expected_seed_set:
        raise RuntimeError("Uniqueness rows do not cover the frozen ten training seeds")
    per_seed = result.groupby("training_seed")["pair_id"].nunique()
    if not per_seed.eq(EXPECTED_TEST_PAIRS_PER_SEED).all():
        raise RuntimeError(f"Expected {EXPECTED_TEST_PAIRS_PER_SEED} unordered pairs per seed; got {per_seed.to_dict()}")
    if len(result) != len(expected_seed_set) * EXPECTED_TEST_PAIRS_PER_SEED:
        raise RuntimeError("Expected exactly 8,200 unordered pair scores")
    return result.sort_values(["training_seed", "pair_id"]).reset_index(drop=True)


def uniqueness_stage(config: Mapping[str, Any]) -> Path:
    paths = config_paths(config)
    evaluation_root = paths["evaluation"]
    output = paths["uniqueness"]
    assert isinstance(evaluation_root, Path) and isinstance(output, Path)
    write_protocol_lock(config)
    threshold = load_frozen_threshold(config)
    frames: list[pd.DataFrame] = []
    for seed in EXPECTED_TRAINING_SEEDS:
        source = evaluation_root / f"seed_{seed}" / "uniqueness_rows_all.csv"
        if not source.is_file():
            raise FileNotFoundError(f"Missing test uniqueness output: {source}")
        frame = pd.read_csv(source)
        if "exp_id" not in frame.columns:
            raise ValueError(f"Missing exp_id in {source}")
        frame = frame[frame["exp_id"].astype(str) == "E5"].copy()
        frame["training_seed"] = int(seed)
        frames.append(frame)
    directed = pd.concat(frames, ignore_index=True)
    # Section 5 describes 410 clean self matches versus 3,800 calibration
    # impostors. Export those cohorts separately from augmented calibration FRR.
    output.mkdir(parents=True, exist_ok=True)
    self_rows = directed[directed["registered_identity"].astype(str) == directed["tested_identity"].astype(str)].copy()
    if len(self_rows) != 410 or self_rows.duplicated(["training_seed", "registered_identity"]).any():
        raise RuntimeError("Discussion diagnostic requires exactly 410 unique clean self matches")
    self_rows.to_csv(output / "discussion_clean_self_scores.csv", index=False, encoding="utf-8")
    calibration = pd.read_csv(paths["calibration"] / "calibration_scores.csv")
    impostors = calibration.loc[calibration["score_type"] == "impostor", "nc"].astype(float).to_numpy()
    if impostors.size != 3800:
        raise RuntimeError("Discussion diagnostic requires 3,800 calibration base impostors")
    genuine = self_rows["nc"].astype(float).to_numpy()
    curve = pd.DataFrame([{"threshold": float(t), "far": float(np.mean(impostors >= t)),
                           "frr": float(np.mean(genuine < t))} for t in np.linspace(0, 1, 1001)])
    curve.to_csv(output / "discussion_far_frr_curve.csv", index=False, encoding="utf-8")
    discussion = {"clean_self_count": 410, "calibration_impostor_count": 3800,
                  "predeclared_threshold": threshold,
                  "clean_self_frr": float(np.mean(genuine < threshold)),
                  "calibration_base_far": float(np.mean(impostors >= threshold)),
                  "clean_self_nc_min": float(genuine.min()),
                  "note": "Clean deterministic self recovery should give NC=1. This is not attacked-sample robustness and does not select a threshold."}
    (output / "discussion_diagnostic.json").write_text(json.dumps(discussion, indent=2), encoding="utf-8")
    pairs = collapse_directed_uniqueness(directed, threshold=threshold)
    output.mkdir(parents=True, exist_ok=True)
    pairs_path = output / "uniqueness_unordered_pairs.csv"
    pairs.to_csv(pairs_path, index=False, encoding="utf-8")

    by_seed = (
        pairs.groupby("training_seed", as_index=False)
        .agg(
            pair_count=("nc", "count"),
            nc_mean=("nc", "mean"),
            nc_median=("nc", "median"),
            nc_max=("nc", "max"),
            exceed_count=("accepted", "sum"),
        )
    )
    by_seed["empirical_far"] = by_seed["exceed_count"] / by_seed["pair_count"]
    by_seed["far_95_one_sided_upper"] = [
        one_sided_binomial_upper(int(k), int(n))
        for k, n in zip(by_seed["exceed_count"], by_seed["pair_count"])
    ]
    by_seed.to_csv(output / "uniqueness_by_seed.csv", index=False, encoding="utf-8")

    exceed = int(pairs["accepted"].sum())
    total = int(len(pairs))
    conservative_pair = pairs.groupby("pair_id", as_index=False).agg(
        nc=("nc", "max"), accepted=("accepted", "max")
    )
    pair_exceed = int(conservative_pair["accepted"].sum())
    summary = {
        "identity_count": 41,
        "seed_count": 10,
        "pair_mode": "unordered",
        "directed_pair_reduction": "maximum of the two directions",
        "pairs_per_seed": EXPECTED_TEST_PAIRS_PER_SEED,
        "pair_score_count": total,
        "threshold": threshold,
        "nc_mean": float(pairs["nc"].mean()),
        "nc_median": float(pairs["nc"].median()),
        "nc_p95": float(pairs["nc"].quantile(0.95)),
        "nc_p99": float(pairs["nc"].quantile(0.99)),
        "maximum_nc": float(pairs["nc"].max()),
        "exceed_count": exceed,
        "empirical_far": exceed / total,
        "far_95_one_sided_upper": one_sided_binomial_upper(exceed, total),
        "conservative_identity_pair_units": int(len(conservative_pair)),
        "conservative_pair_exceed_count": pair_exceed,
        "conservative_pair_far": pair_exceed / len(conservative_pair),
        "conservative_pair_far_95_one_sided_upper": one_sided_binomial_upper(
            pair_exceed, len(conservative_pair)
        ),
    }
    (output / "uniqueness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return pairs_path


def _mean_ci95(values: Sequence[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    n = int(array.size)
    if n == 0:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else 0.0
    return mean, sd, mean - half, mean + half


def aggregate_ablation_stage(config: Mapping[str, Any]) -> Path:
    paths = config_paths(config)
    evaluation_root = paths["evaluation"]
    reports = paths["reports"]
    assert isinstance(evaluation_root, Path) and isinstance(reports, Path)
    write_protocol_lock(config)
    frames: list[pd.DataFrame] = []
    for seed in EXPECTED_TRAINING_SEEDS:
        source = evaluation_root / f"seed_{seed}" / "ablation_results.csv"
        if not source.is_file():
            raise FileNotFoundError(f"Missing ablation output: {source}")
        frame = pd.read_csv(source)
        frame["training_seed"] = int(seed)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    if set(raw["exp_id"].astype(str)) != {f"E{i}" for i in range(1, 8)}:
        raise RuntimeError("Ablation output must cover E1-E7")
    counts = raw.groupby("exp_id")["training_seed"].nunique()
    if not counts.eq(10).all() or len(raw) != 70:
        raise RuntimeError(f"Expected E1-E7 x 10 seeds = 70 rows; got {len(raw)}")
    reports.mkdir(parents=True, exist_ok=True)
    raw_path = reports / "ablation_raw.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8")

    metrics = [
        name
        for name in (
            "robust_nc_mean",
            "robust_pass_rate",
            "offdiagonal_nc_mean",
            "offdiagonal_nc_max",
            "false_accept_rate",
            "zero_watermark_time_including_read_mean_s",
            "feature_extract_time_median_mean_s",
        )
        if name in raw.columns
    ]
    summary_rows: list[dict[str, Any]] = []
    for exp_id, group in raw.groupby("exp_id", sort=True):
        for metric in metrics:
            values = group[metric].astype(float).to_numpy()
            mean, sd, low, high = _mean_ci95(values)
            summary_rows.append(
                {
                    "exp_id": str(exp_id),
                    "metric": metric,
                    "seed_count": int(values.size),
                    "mean": mean,
                    "sd": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        reports / "ablation_summary.csv", index=False, encoding="utf-8"
    )

    paired_rows: list[dict[str, Any]] = []
    for baseline in ("E4", "E6", "E7"):
        for metric in metrics:
            left = raw[raw["exp_id"] == "E5"][["training_seed", metric]].rename(columns={metric: "e5"})
            right = raw[raw["exp_id"] == baseline][["training_seed", metric]].rename(columns={metric: "baseline"})
            paired = left.merge(right, on="training_seed", validate="one_to_one")
            diff = paired["e5"].astype(float) - paired["baseline"].astype(float)
            mean, sd, low, high = _mean_ci95(diff)
            try:
                wilcoxon_p = float(stats.wilcoxon(diff).pvalue)
            except ValueError:
                wilcoxon_p = 1.0
            paired_rows.append(
                {
                    "comparison": f"E5-{baseline}",
                    "metric": metric,
                    "paired_seed_count": int(len(diff)),
                    "mean_difference": mean,
                    "difference_sd": sd,
                    "difference_ci95_low": low,
                    "difference_ci95_high": high,
                    "paired_t_p": float(stats.ttest_rel(paired["e5"], paired["baseline"]).pvalue),
                    "wilcoxon_p": wilcoxon_p,
                    "paired_effect_dz": float(mean / sd) if sd > 0 else 0.0,
                }
            )
    paired_df = pd.DataFrame(paired_rows)
    if not paired_df.empty:
        order = paired_df["wilcoxon_p"].sort_values().index
        m = len(order)
        adjusted = pd.Series(index=paired_df.index, dtype=float)
        running = 0.0
        for rank, idx in enumerate(order):
            value = min(1.0, float(paired_df.at[idx, "wilcoxon_p"]) * (m - rank))
            running = max(running, value)
            adjusted.at[idx] = running
        paired_df["wilcoxon_p_holm"] = adjusted
    paired_df.to_csv(reports / "paired_statistics.csv", index=False, encoding="utf-8")
    return raw_path


def environment_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "perf_counter": "time.perf_counter_ns",
        "config_sha256": str(config["_config_sha256"]),
        "packages": {},
    }
    for name in ("numpy", "pandas", "scipy", "geopandas", "shapely", "rasterio", "torch", "PIL"):
        try:
            module = __import__(name)
            payload["packages"][name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            payload["packages"][name] = "not installed"
    try:
        import torch

        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["torch_cuda_build"] = str(torch.version.cuda)
        payload["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception:
        payload["cuda_available"] = False
        payload["torch_cuda_build"] = ""
        payload["gpu_name"] = ""
    payload["timing_protocol"] = dict(config["efficiency"])
    return payload


def _sync_device(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def benchmark_efficiency_stage(
    config: Mapping[str, Any], *, seeds: Sequence[int] | None = None
) -> Path:
    """Benchmark E5 registration stages with explicit warm-ups and repeats."""

    from .data import base_manifest
    from .model import extract_embedding, feature_to_bits, load_encoder
    from .vector import build_four_channels, read_vector
    from .watermark import create_registry_record, watermark_image_to_bits

    paths = config_paths(config)
    prepared = paths["prepared"]
    source_root = paths["source_root"]
    split_path = paths["split"]
    watermark_path = paths["watermark"]
    output = paths["efficiency"]
    assert all(
        isinstance(x, Path)
        for x in (prepared, split_path, watermark_path, output)
    )
    validate_split_file(split_path)
    write_protocol_lock(config)
    _validate_eligible_watermark(config)
    base = base_manifest(prepared, split_path, "test")
    if len(base) != 41:
        raise RuntimeError(f"Efficiency benchmark requires 41 test identities; got {len(base)}")
    watermark_bits, _, _ = watermark_image_to_bits(
        watermark_path, int(config["watermark"]["bit_length"])
    )
    checkpoints = _e5_checkpoints(config)
    chosen = [int(x) for x in (seeds or EXPECTED_TRAINING_SEEDS)]
    warmups = int(config["efficiency"]["warmup_repeats"])
    repeats = int(config["efficiency"]["measured_repeats"])
    device_request = str(config["efficiency"]["device"])
    rows: list[dict[str, Any]] = []

    for seed in chosen:
        checkpoint_path = checkpoints[seed]
        encoder, checkpoint, resolved = load_encoder(checkpoint_path, device_request)
        experiment = checkpoint["experiment"]
        channel_indices = tuple(int(x) for x in experiment["channel_indices"])
        for item in base.to_dict("records"):
            identity = str(item["identity"])
            source_value = Path(str(item["source_path"])).expanduser()
            source_path = str(
                (source_value if source_value.is_absolute() else source_root / source_value).resolve()
            )
            for repeat_index in range(-warmups, repeats):
                warmup = repeat_index < 0
                total_start = time.perf_counter_ns()

                start = time.perf_counter_ns()
                gdf = read_vector(source_path)
                vector_read_s = (time.perf_counter_ns() - start) / 1e9

                start = time.perf_counter_ns()
                tensor, channel_meta = build_four_channels(
                    gdf,
                    grid_size=int(config["data"]["grid_size"]),
                    density_sigma=float(config["data"]["density_sigma"]),
                )
                channel_s = (time.perf_counter_ns() - start) / 1e9

                _sync_device(resolved)
                start = time.perf_counter_ns()
                embedding = extract_embedding(encoder, tensor, channel_indices, resolved)
                _sync_device(resolved)
                feature_s = (time.perf_counter_ns() - start) / 1e9

                start = time.perf_counter_ns()
                bits = feature_to_bits(
                    embedding,
                    int(config["watermark"]["bit_length"]),
                    str(config["watermark"]["quantization"]),
                )
                quantization_s = (time.perf_counter_ns() - start) / 1e9

                start = time.perf_counter_ns()
                create_registry_record(
                    identity=identity,
                    source_path=source_path,
                    checkpoint_path=checkpoint_path,
                    watermark_bits=watermark_bits,
                    feature_bits=bits,
                    experiment=experiment,
                    channel_config={
                        "grid_size": int(config["data"]["grid_size"]),
                        "density_sigma": float(config["data"]["density_sigma"]),
                        "channel_meta": channel_meta,
                    },
                    threshold_mode=str(config["watermark"]["quantization"]),
                )
                xor_record_s = (time.perf_counter_ns() - start) / 1e9
                total_s = (time.perf_counter_ns() - total_start) / 1e9

                stage_sum = vector_read_s + channel_s + feature_s + quantization_s + xor_record_s
                rows.append(
                    {
                        "training_seed": int(seed),
                        "identity": identity,
                        "repeat": int(repeat_index),
                        "warmup": bool(warmup),
                        "device": str(resolved),
                        "feature_count": int(len(gdf)),
                        "source_bytes": int(Path(source_path).stat().st_size),
                        "vector_read_time_s": vector_read_s,
                        "channel_construction_time_s": channel_s,
                        "feature_extraction_time_s": feature_s,
                        "quantization_time_s": quantization_s,
                        "xor_record_time_s": xor_record_s,
                        "stage_sum_time_s": stage_sum,
                        "end_to_end_time_s": total_s,
                        "unattributed_overhead_time_s": total_s - stage_sum,
                    }
                )

    raw = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "efficiency_raw.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8")
    measured = raw[~raw["warmup"]].copy()
    expected = len(chosen) * 41 * repeats
    if len(measured) != expected:
        raise RuntimeError(f"Expected {expected} measured timing rows; got {len(measured)}")
    timing_columns = [
        "vector_read_time_s",
        "channel_construction_time_s",
        "feature_extraction_time_s",
        "quantization_time_s",
        "xor_record_time_s",
        "end_to_end_time_s",
    ]
    identity_seed = measured.groupby(["training_seed", "identity"], as_index=False)[timing_columns].median()
    identity_seed.to_csv(output / "efficiency_identity_seed_medians.csv", index=False, encoding="utf-8")
    summary_rows: list[dict[str, Any]] = []
    for metric in timing_columns:
        values = identity_seed[metric].astype(float).to_numpy()
        mean, sd, low, high = _mean_ci95(values)
        summary_rows.append(
            {
                "metric": metric,
                "identity_seed_units": int(len(values)),
                "raw_measurements": int(len(measured)),
                "mean": mean,
                "sd": sd,
                "ci95_low": low,
                "ci95_high": high,
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        output / "efficiency_summary.csv", index=False, encoding="utf-8"
    )
    metadata = environment_metadata(config)
    metadata.update(
        {
            "test_identity_count": 41,
            "training_seed_count": len(chosen),
            "measured_identity_seed_units": int(len(identity_seed)),
            "warmup_repeats": warmups,
            "measured_repeats": repeats,
            "aggregation_unit": "median per identity and training seed",
        }
    )
    (output / "environment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return raw_path


def figures_stage(config: Mapping[str, Any]) -> Path:
    """Render threshold, uniqueness, robustness, and runtime figures from outputs."""

    from .figures import make_all_figures

    paths = config_paths(config)
    load_frozen_threshold(config)
    required = (
        paths["calibration"] / "calibration_scores.csv",
        paths["uniqueness"] / "uniqueness_unordered_pairs.csv",
        paths["efficiency"] / "efficiency_raw.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run all evaluation stages before figures; first missing: {missing[0]}")
    return make_all_figures(paths)


def run_manifest(config: Mapping[str, Any]) -> Path:
    paths = config_paths(config)
    run_root = paths["run_root"]
    assert isinstance(run_root, Path)
    artifacts: list[dict[str, Any]] = []
    for path in sorted(p for p in run_root.rglob("*") if p.is_file()):
        if path.name == "run_manifest.json":
            continue
        artifacts.append(
            {
                "path": path.relative_to(run_root).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    payload = {
        "schema_version": 1,
        "protocol": protocol_lock(config, paths["split"]),
        "artifacts": artifacts,
    }
    output = run_root / "run_manifest.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
