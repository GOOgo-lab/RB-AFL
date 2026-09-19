from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import base_manifest
from .model import (
    checkpoint_index,
    encoder_parameter_bytes,
    extract_embedding,
    feature_to_bits,
    load_encoder,
)
from .protocol import NC_THRESHOLD, attack_plan, protocol_dict
from .vector import apply_attack, build_four_channels, read_vector, stable_seed
from .watermark import (
    ber_score,
    bit_sha256,
    create_registry_record,
    file_sha256,
    load_registry,
    nc_score,
    recover_watermark,
    save_bits_image,
    save_registry,
    verify_recovered,
    watermark_image_to_bits,
)


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _timed_call(fn, device: str = "cpu"):
    _sync(device)
    start = time.perf_counter()
    value = fn()
    _sync(device)
    return value, time.perf_counter() - start


def _median_encode_time(
    encoder,
    tensor: np.ndarray,
    channel_indices: Sequence[int],
    device: str,
    repeats: int,
) -> float:
    for _ in range(2):
        extract_embedding(encoder, tensor, channel_indices, device)
    times: List[float] = []
    for _ in range(max(1, repeats)):
        _, elapsed = _timed_call(
            lambda: extract_embedding(encoder, tensor, channel_indices, device), device
        )
        times.append(elapsed)
    return float(statistics.median(times))


def _source_dataset_bytes(path: str | Path) -> int:
    source = Path(path)
    if source.suffix.lower() == ".shp":
        return int(sum(p.stat().st_size for p in source.parent.glob(source.stem + ".*")))
    return int(source.stat().st_size)


def _checkpoint_rows(models_root: str | Path, selected: Optional[Sequence[str]]) -> pd.DataFrame:
    index = checkpoint_index(models_root)
    if selected:
        keys = {x.lower() for x in selected}
        index = index[
            index.apply(
                lambda r: str(r["exp_id"]).lower() in keys
                or str(r["exp_name"]).lower() in keys,
                axis=1,
            )
        ]
    if index.empty:
        raise RuntimeError("No trained experiment matched --selected")
    return index.sort_values("exp_id").reset_index(drop=True)


def register_one(
    vector_path: str | Path,
    watermark_path: str | Path,
    checkpoint_path: str | Path,
    registry_path: str | Path,
    identity: Optional[str] = None,
    grid_size: int = 256,
    density_sigma: float = 3.0,
    bit_length: int = 256,
    threshold_mode: str = "median",
    device: str = "auto",
    timing_repeats: int = 10,
) -> Dict[str, object]:
    encoder, checkpoint, resolved = load_encoder(checkpoint_path, device)
    experiment = checkpoint["experiment"]
    if str(experiment.get("exp_id")) != "E5":
        raise ValueError("Public registration requires an E5 proposed-method checkpoint")
    channel_indices = tuple(int(x) for x in experiment["channel_indices"])
    watermark_bits, width, height = watermark_image_to_bits(watermark_path, bit_length)
    if bit_length != 256 or int(watermark_bits.sum()) != 45:
        raise ValueError("Eligible watermark must contain exactly 45 one bits and 211 zero bits")
    if threshold_mode not in {"median", "mean", "zero"}:
        raise ValueError("Use elementwise median, mean or zero quantization")

    gdf, read_s = _timed_call(lambda: read_vector(vector_path))
    tensor, channel_s = _timed_call(
        lambda: build_four_channels(gdf, grid_size=grid_size, density_sigma=density_sigma)
    )
    channel_tensor, channel_meta = tensor
    embedding, encode_once_s = _timed_call(
        lambda: extract_embedding(encoder, channel_tensor, channel_indices, resolved), resolved
    )
    feature_bits, quantize_s = _timed_call(
        lambda: feature_to_bits(embedding, bit_length, threshold_mode)
    )
    encode_median_s = _median_encode_time(
        encoder, channel_tensor, channel_indices, resolved, timing_repeats
    )
    zero_start = time.perf_counter()
    # create_registry_record performs the actual W XOR B operation.
    tensor_bytes = int(channel_tensor[list(channel_indices)].nbytes)
    space = {
        "selected_channel_tensor_bytes": tensor_bytes,
        "encoder_parameter_bytes": encoder_parameter_bytes(encoder),
        "embedding_bytes": int(embedding.nbytes),
        "feature_bits_unpacked_bytes": int(feature_bits.nbytes),
        "registered_zero_watermark_payload_bytes": int(math.ceil(bit_length / 8)),
        "source_dataset_bytes_not_counted_as_extra": _source_dataset_bytes(vector_path),
    }
    timing = {
        "vector_read_time_s": read_s,
        "channel_build_time_s": channel_s,
        "feature_extract_time_single_s": encode_once_s,
        "feature_extract_time_median_s": encode_median_s,
        "feature_quantization_time_s": quantize_s,
    }
    record = create_registry_record(
        identity=identity or Path(vector_path).stem,
        source_path=vector_path,
        checkpoint_path=checkpoint_path,
        watermark_bits=watermark_bits,
        feature_bits=feature_bits,
        experiment=experiment,
        channel_config={
            "grid_size": grid_size,
            "density_sigma": density_sigma,
            "canonicalization": "centroid + PCA dominant axis + isotropic extent",
            "channel_meta": channel_meta,
            "watermark_width": width,
            "watermark_height": height,
        },
        threshold_mode=threshold_mode,
        timing=timing,
        space=space,
    )
    xor_and_record_s = time.perf_counter() - zero_start
    timing["xor_and_record_build_time_s"] = xor_and_record_s
    timing["zero_watermark_generation_time_s_excluding_read"] = (
        channel_s + encode_median_s + quantize_s + xor_and_record_s
    )
    timing["zero_watermark_generation_time_s_including_read"] = (
        read_s + timing["zero_watermark_generation_time_s_excluding_read"]
    )
    record["timing"] = timing
    record["space"] = space
    registry_file = Path(registry_path).expanduser().resolve()
    existing_records: List[Dict[str, object]] = []
    if registry_file.is_file():
        existing_records = list(load_registry(registry_file).get("records", []))
    if any(
        item.get("record_id") == record.get("record_id")
        or item.get("identity") == record.get("identity")
        for item in existing_records
    ):
        raise ValueError(
            "Registry already contains this record_id or identity; use a distinct identity or a new registry"
        )
    records = [*existing_records, record]
    registry = save_registry(registry_file, records)
    space["registry_json_bytes"] = registry.stat().st_size
    # The JSON already contains the Base64-encoded zero-watermark payload.
    space["additional_storage_bytes_actual"] = space["registry_json_bytes"]
    record["space"] = space
    save_registry(registry_file, records)
    return record


def verify_one(
    suspicious_vector_path: str | Path,
    watermark_path: str | Path,
    checkpoint_path: str | Path,
    registry_path: str | Path,
    identity_or_id: str,
    output_recovered_image: Optional[str | Path] = None,
    nc_threshold: Optional[float] = NC_THRESHOLD,
    device: str = "auto",
) -> Dict[str, object]:
    if nc_threshold is None:
        raise ValueError("nc_threshold is required; load the frozen calibration artifact")
    registry = load_registry(registry_path)
    candidates = [
        x
        for x in registry["records"]
        if x["identity"] == identity_or_id or x["record_id"] == identity_or_id
    ]
    if len(candidates) != 1:
        raise KeyError(f"Expected one registry record for {identity_or_id}; found {len(candidates)}")
    record = candidates[0]
    from .watermark import validate_record_integrity
    validate_record_integrity(record)
    if file_sha256(checkpoint_path) != record["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from the registered checkpoint")
    encoder, checkpoint, resolved = load_encoder(checkpoint_path, device)
    experiment = checkpoint["experiment"]
    if str(experiment.get("exp_id")) != "E5":
        raise ValueError("Public verification requires an E5 proposed-method checkpoint")
    if experiment["exp_id"] != record["experiment"]["exp_id"]:
        raise ValueError("Checkpoint experiment does not match the registry")
    channel_indices = tuple(int(x) for x in experiment["channel_indices"])
    config = record["channel_config"]
    bit_length = int(record["bit_length"])
    watermark_bits, width, height = watermark_image_to_bits(watermark_path, bit_length)
    if bit_length != 256 or int(watermark_bits.sum()) != 45:
        raise ValueError("Eligible watermark must contain exactly 45 one bits and 211 zero bits")
    if str(record.get("threshold_mode")) not in {"median", "mean", "zero"}:
        raise ValueError("Registry record does not use elementwise threshold quantization")
    start = time.perf_counter()
    gdf = read_vector(suspicious_vector_path)
    tensor, _ = build_four_channels(
        gdf,
        grid_size=int(config["grid_size"]),
        density_sigma=float(config["density_sigma"]),
    )
    embedding = extract_embedding(encoder, tensor, channel_indices, resolved)
    feature_bits = feature_to_bits(embedding, bit_length, str(record["threshold_mode"]))
    recovered = recover_watermark(record, feature_bits)
    result = verify_recovered(record, recovered, watermark_bits, nc_threshold)
    result.update(
        {
            "identity": record["identity"],
            "record_id": record["record_id"],
            "verification_time_s": time.perf_counter() - start,
        }
    )
    if output_recovered_image:
        save_bits_image(recovered, width, height, output_recovered_image)
        result["recovered_image"] = str(Path(output_recovered_image).resolve())
    return result


def _uniqueness_matrices(
    identities: Sequence[str],
    records: Sequence[Dict[str, object]],
    clean_bits: Sequence[np.ndarray],
    watermark_bits: np.ndarray,
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[Dict[str, object]]]:
    n = len(identities)
    nc = np.zeros((n, n), dtype=np.float64)
    ber = np.zeros((n, n), dtype=np.float64)
    long_rows: List[Dict[str, object]] = []
    for i, record in enumerate(records):
        for j, bits in enumerate(clean_bits):
            recovered = recover_watermark(record, bits)
            nc[i, j] = nc_score(watermark_bits, recovered)
            ber[i, j] = ber_score(watermark_bits, recovered)
            long_rows.append(
                {
                    "registered_identity": identities[i],
                    "tested_identity": identities[j],
                    "same_identity": i == j,
                    "nc": nc[i, j],
                    "ber": ber[i, j],
                    "accepted": bool(nc[i, j] >= threshold),
                }
            )
    off = ~np.eye(n, dtype=bool)
    off_nc = nc[off]
    off_ber = ber[off]
    summary = {
        "diagonal_nc_mean": float(np.diag(nc).mean()),
        "offdiagonal_nc_mean": float(off_nc.mean()) if off_nc.size else 0.0,
        "offdiagonal_nc_max": float(off_nc.max()) if off_nc.size else 0.0,
        "offdiagonal_ber_mean": float(off_ber.mean()) if off_ber.size else 0.0,
        "false_accept_rate": float(np.mean(off_nc >= threshold)) if off_nc.size else 0.0,
        "threshold": float(threshold),
    }
    return (
        pd.DataFrame(nc, index=identities, columns=identities),
        pd.DataFrame(ber, index=identities, columns=identities),
        summary,
        long_rows,
    )


def _experiment_complete(
    exp_dir: Path,
    identity_count: int,
    attack_count: int,
    threshold: float,
) -> bool:
    required = [
        exp_dir / "summary.json",
        exp_dir / "timing_space.csv",
        exp_dir / "robustness_rows.csv",
        exp_dir / "uniqueness_rows.csv",
        exp_dir / "registry.json",
    ]
    if not all(path.is_file() for path in required):
        return False
    try:
        robust = pd.read_csv(exp_dir / "robustness_rows.csv")
        timing = pd.read_csv(exp_dir / "timing_space.csv")
        unique = pd.read_csv(exp_dir / "uniqueness_rows.csv")
        if len(robust) != identity_count * attack_count:
            return False
        if len(timing) != identity_count or len(unique) != identity_count * identity_count:
            return False
        if "status" not in robust or (robust["status"] == "error").any():
            return False
        if "threshold" in robust and not np.allclose(
            robust["threshold"].dropna().astype(float).to_numpy(), float(threshold), atol=1e-12
        ):
            return False
        summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        if not math.isclose(float(summary.get("threshold", threshold)), float(threshold), abs_tol=1e-12):
            return False
        return True
    except Exception:
        return False


def _load_completed_experiment(exp_dir: Path):
    summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
    timing = pd.read_csv(exp_dir / "timing_space.csv").to_dict("records")
    robust = pd.read_csv(exp_dir / "robustness_rows.csv").to_dict("records")
    unique = pd.read_csv(exp_dir / "uniqueness_rows.csv").to_dict("records")
    return summary, timing, robust, unique


def evaluate_all(
    prepared_root: str | Path,
    models_root: str | Path,
    watermark_path: str | Path,
    output_root: str | Path,
    external_vector_path: Optional[str | Path] = None,
    selected: Optional[Sequence[str]] = None,
    grid_size: int = 256,
    density_sigma: float = 3.0,
    bit_length: int = 256,
    threshold_mode: str = "median",
    nc_threshold: Optional[float] = NC_THRESHOLD,
    seed: int = 20260730,
    device: str = "auto",
    timing_repeats: int = 10,
    identity_split_path: str | Path | None = None,
    study_split_name: str | None = None,
    resume: bool = True,
    source_path_map: str | Path | None = None,
) -> Dict[str, Path]:
    """Evaluate robustness/uniqueness/resources on an optional held-out identity split.

    For held-out study experiments, pass ``identity_split_path`` together with
    ``study_split_name='test'``.  The function is restart-safe at experiment level
    and incrementally saves robustness rows after each identity.
    """

    if nc_threshold is None:
        raise ValueError("nc_threshold is required; load the frozen calibration artifact")
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_payload = protocol_dict()
    if isinstance(protocol_payload.get("decision_rule"), dict):
        protocol_payload["decision_rule"]["threshold"] = float(nc_threshold)
        protocol_payload["decision_rule"]["threshold_source"] = "caller supplied; public pipeline uses held-out calibration"
    protocol_payload["evaluation"] = {
        "seed": int(seed),
        "nc_threshold": float(nc_threshold),
        "identity_split_path": str(Path(identity_split_path).expanduser().resolve())
        if identity_split_path is not None
        else "",
        "study_split_name": study_split_name or "all",
        "source_path_map": str(Path(source_path_map).expanduser().resolve()) if source_path_map is not None else "",
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    base = base_manifest(
        prepared_root,
        identity_split_path=identity_split_path,
        split_name=study_split_name,
    )
    if source_path_map is not None:
        mapping_path = Path(source_path_map).expanduser().resolve()
        if not mapping_path.is_file():
            raise FileNotFoundError(f"Source-path mapping not found: {mapping_path}")
        mapping_df = pd.read_csv(mapping_path).fillna("")
        required_columns = {"identity", "resolved_source_path", "status"}
        missing_columns = required_columns.difference(mapping_df.columns)
        if missing_columns:
            raise ValueError(f"Source-path mapping is missing columns: {sorted(missing_columns)}")
        resolved_map = mapping_df.set_index(mapping_df["identity"].astype(str))["resolved_source_path"].to_dict()
        status_map = mapping_df.set_index(mapping_df["identity"].astype(str))["status"].to_dict()
        base["source_path_original"] = base["source_path"].astype(str)
        for idx in base.index:
            identity = str(base.at[idx, "identity"])
            if str(status_map.get(identity, "")) == "resolved" and str(resolved_map.get(identity, "")).strip():
                base.at[idx, "source_path"] = str(resolved_map[identity])

    identities = base["identity"].astype(str).tolist()
    source_paths = base["source_path"].astype(str).tolist()
    if len(identities) < 2:
        raise RuntimeError("Evaluation requires at least two identities for uniqueness analysis")
    missing_sources = [path for path in source_paths if not Path(path).is_file()]
    if missing_sources:
        raise FileNotFoundError(
            "The prepared manifest points to source vectors that are no longer available. "
            f"First missing path: {missing_sources[0]}"
        )

    watermark_bits, watermark_width, watermark_height = watermark_image_to_bits(
        watermark_path, bit_length
    )
    if np.all(watermark_bits == watermark_bits[0]):
        raise ValueError(
            "Degenerate copyright watermark: all bits are identical. Use a watermark containing both 0 and 1 bits."
        )
    external = read_vector(external_vector_path) if external_vector_path else None
    model_index = _checkpoint_rows(models_root, selected)
    plan = attack_plan()

    all_ablation_rows: List[Dict[str, object]] = []
    all_timing_rows: List[Dict[str, object]] = []
    all_robust_rows: List[Dict[str, object]] = []
    all_unique_rows: List[Dict[str, object]] = []

    for _, model_row in model_index.iterrows():
        exp_id = str(model_row["exp_id"])
        exp_name = str(model_row["exp_name"])
        checkpoint_path = Path(str(model_row["checkpoint"]))
        exp_dir = output / f"{exp_id}_{exp_name}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        if resume and _experiment_complete(exp_dir, len(identities), len(plan), nc_threshold):
            print(f"[EVALUATE] {exp_id} {exp_name}: completed -> reuse", flush=True)
            summary_row, timing_rows_old, robust_rows_old, unique_rows_old = _load_completed_experiment(exp_dir)
            all_ablation_rows.append(summary_row)
            all_timing_rows.extend(timing_rows_old)
            all_robust_rows.extend(robust_rows_old)
            all_unique_rows.extend(unique_rows_old)
            continue

        print(f"[EVALUATE] {exp_id} {exp_name}", flush=True)
        encoder, checkpoint, resolved = load_encoder(checkpoint_path, device)
        experiment = checkpoint["experiment"]
        channel_indices = tuple(int(x) for x in experiment["channel_indices"])
        parameter_bytes = encoder_parameter_bytes(encoder)

        clean_gdfs = []
        clean_bits: List[np.ndarray] = []
        records: List[Dict[str, object]] = []
        timing_rows: List[Dict[str, object]] = []
        for identity, source_path in zip(identities, source_paths):
            gdf, read_s = _timed_call(lambda p=source_path: read_vector(p))
            clean_gdfs.append(gdf)
            tensor_meta, channel_s = _timed_call(
                lambda g=gdf: build_four_channels(
                    g, grid_size=grid_size, density_sigma=density_sigma
                )
            )
            tensor, channel_meta = tensor_meta
            embedding, encode_s = _timed_call(
                lambda: extract_embedding(encoder, tensor, channel_indices, resolved), resolved
            )
            median_encode_s = _median_encode_time(
                encoder, tensor, channel_indices, resolved, timing_repeats
            )
            bits, quantize_s = _timed_call(
                lambda: feature_to_bits(embedding, bit_length, threshold_mode)
            )
            clean_bits.append(bits)
            selected_tensor_bytes = int(tensor[list(channel_indices)].nbytes)
            space = {
                "selected_channel_tensor_bytes": selected_tensor_bytes,
                "encoder_parameter_bytes": parameter_bytes,
                "embedding_bytes": int(embedding.nbytes),
                "registered_zero_watermark_payload_bytes": int(math.ceil(bit_length / 8)),
                "working_memory_bytes_excluding_model": selected_tensor_bytes
                + int(embedding.nbytes)
                + int(bits.nbytes),
            }
            xor_start = time.perf_counter()
            record = create_registry_record(
                identity=identity,
                source_path=source_path,
                checkpoint_path=checkpoint_path,
                watermark_bits=watermark_bits,
                feature_bits=bits,
                experiment=experiment,
                channel_config={
                    "grid_size": grid_size,
                    "density_sigma": density_sigma,
                    "canonicalization": "centroid + PCA dominant axis + isotropic extent",
                    "watermark_width": watermark_width,
                    "watermark_height": watermark_height,
                    "channel_meta": channel_meta,
                },
                threshold_mode=threshold_mode,
                space=space,
            )
            xor_s = time.perf_counter() - xor_start
            record_json_bytes = len(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            space["registry_record_json_bytes"] = record_json_bytes
            space["additional_storage_bytes_actual_per_record"] = record_json_bytes
            timing = {
                "vector_read_time_s": read_s,
                "channel_build_time_s": channel_s,
                "feature_extract_time_single_s": encode_s,
                "feature_extract_time_median_s": median_encode_s,
                "feature_quantization_time_s": quantize_s,
                "xor_and_record_build_time_s": xor_s,
                "zero_watermark_generation_time_s_excluding_read": channel_s
                + median_encode_s
                + quantize_s
                + xor_s,
                "zero_watermark_generation_time_s_including_read": read_s
                + channel_s
                + median_encode_s
                + quantize_s
                + xor_s,
            }
            record["timing"] = timing
            record["space"] = space
            records.append(record)
            timing_rows.append(
                {
                    "evaluation_seed": int(seed),
                    "study_split": study_split_name or "all",
                    "exp_id": exp_id,
                    "exp_name": exp_name,
                    "identity": identity,
                    "channels": ",".join(experiment["channels"]),
                    **timing,
                    **space,
                }
            )

        registry_path = save_registry(exp_dir / "registry.json", records)
        timing_df = pd.DataFrame(timing_rows)
        timing_df.to_csv(exp_dir / "timing_space.csv", index=False, encoding="utf-8-sig")
        all_timing_rows.extend(timing_rows)

        nc_matrix, ber_matrix, unique_summary, unique_rows = _uniqueness_matrices(
            identities, records, clean_bits, watermark_bits, nc_threshold
        )
        nc_matrix.to_csv(exp_dir / "uniqueness_nc_matrix.csv", encoding="utf-8-sig")
        ber_matrix.to_csv(exp_dir / "uniqueness_ber_matrix.csv", encoding="utf-8-sig")
        for row in unique_rows:
            row.update(
                {
                    "evaluation_seed": int(seed),
                    "study_split": study_split_name or "all",
                    "exp_id": exp_id,
                    "exp_name": exp_name,
                }
            )
        pd.DataFrame(unique_rows).to_csv(
            exp_dir / "uniqueness_rows.csv", index=False, encoding="utf-8-sig"
        )
        all_unique_rows.extend(unique_rows)

        # Restart-safe attack evaluation: keep successful/skipped old rows and redo errors/missing rows.
        robust_path_child = exp_dir / "robustness_rows.csv"
        robust_map: Dict[Tuple[str, str], Dict[str, object]] = {}
        if resume and robust_path_child.is_file():
            try:
                old = pd.read_csv(robust_path_child)
                for row in old.to_dict("records"):
                    if str(row.get("status", "")) in {"ok", "skipped"}:
                        key = (str(row.get("identity", "")), str(row.get("case_id", "")))
                        robust_map[key] = row
            except Exception:
                robust_map = {}

        for identity, clean_gdf, record in zip(identities, clean_gdfs, records):
            for case_index, case in enumerate(plan):
                key = (identity, case.case_id)
                if key in robust_map:
                    continue
                attack_seed = stable_seed(
                    seed,
                    exp_id,
                    identity,
                    case.attack,
                    case.strength,
                    case.repeat_index,
                )
                base_row: Dict[str, object] = {
                    "evaluation_seed": int(seed),
                    "study_split": study_split_name or "all",
                    "exp_id": exp_id,
                    "exp_name": exp_name,
                    "identity": identity,
                    "attack": case.attack,
                    "strength": case.strength,
                    "repeat_index": int(case.repeat_index),
                    "case_id": case.case_id,
                    "attack_seed": attack_seed,
                }
                try:
                    attacked_meta, attack_s = _timed_call(
                        lambda: apply_attack(
                            clean_gdf,
                            case.attack,
                            case.strength,
                            attack_seed,
                            external,
                        )
                    )
                    attacked, metadata = attacked_meta
                    tensor, channel_s = _timed_call(
                        lambda: build_four_channels(
                            attacked,
                            grid_size=grid_size,
                            density_sigma=density_sigma,
                        )[0]
                    )
                    embedding, encode_s = _timed_call(
                        lambda: extract_embedding(encoder, tensor, channel_indices, resolved),
                        resolved,
                    )
                    bits, quantize_s = _timed_call(
                        lambda: feature_to_bits(embedding, bit_length, threshold_mode)
                    )
                    recovered, recover_s = _timed_call(lambda: recover_watermark(record, bits))
                    nc = nc_score(watermark_bits, recovered)
                    ber = ber_score(watermark_bits, recovered)
                    base_row.update(
                        {
                            "status": "ok",
                            "nc": nc,
                            "ber": ber,
                            "bit_accuracy": 1.0 - ber,
                            "passed": bool(nc >= nc_threshold),
                            "threshold": nc_threshold,
                            "attack_time_s": attack_s,
                            "channel_build_time_s": channel_s,
                            "feature_extract_time_s": encode_s,
                            "feature_quantization_time_s": quantize_s,
                            "watermark_recovery_time_s": recover_s,
                            "verification_time_s": attack_s
                            + channel_s
                            + encode_s
                            + quantize_s
                            + recover_s,
                            "attack_metadata": json.dumps(
                                metadata, ensure_ascii=False, separators=(",", ":")
                            ),
                            "error": "",
                        }
                    )
                except Exception as exc:
                    missing_external = (
                        case.attack == "merge"
                        and "requires --external-vector" in str(exc)
                    )
                    base_row.update(
                        {
                            "status": "skipped" if missing_external else "error",
                            "nc": np.nan,
                            "ber": np.nan,
                            "bit_accuracy": np.nan,
                            "passed": False,
                            "threshold": nc_threshold,
                            "attack_time_s": np.nan,
                            "channel_build_time_s": np.nan,
                            "feature_extract_time_s": np.nan,
                            "feature_quantization_time_s": np.nan,
                            "watermark_recovery_time_s": np.nan,
                            "verification_time_s": np.nan,
                            "attack_metadata": "{}",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                robust_map[key] = base_row
                print(
                    f"  [{exp_id}] {identity} {case_index + 1:02d}/{len(plan)} "
                    f"{case.case_id} status={base_row['status']}",
                    flush=True,
                )

            # Save after each identity so a long 94-attack run can resume.
            pd.DataFrame(list(robust_map.values())).to_csv(
                robust_path_child, index=False, encoding="utf-8-sig"
            )

        ordered_rows: List[Dict[str, object]] = []
        for identity in identities:
            for case in plan:
                key = (identity, case.case_id)
                if key not in robust_map:
                    raise RuntimeError(f"Missing robustness row after evaluation: {key}")
                ordered_rows.append(robust_map[key])
        robust_df = pd.DataFrame(ordered_rows)
        robust_df.to_csv(robust_path_child, index=False, encoding="utf-8-sig")
        ok = robust_df[robust_df["status"] == "ok"].copy()
        summary = (
            ok.groupby(["attack", "strength"], as_index=False)
            .agg(
                sample_count=("nc", "count"),
                nc_mean=("nc", "mean"),
                nc_std=("nc", "std"),
                nc_min=("nc", "min"),
                ber_mean=("ber", "mean"),
                pass_rate=("passed", "mean"),
                verification_time_mean_s=("verification_time_s", "mean"),
            )
            .fillna({"nc_std": 0.0})
        )
        summary.to_csv(exp_dir / "robustness_summary.csv", index=False, encoding="utf-8-sig")
        all_robust_rows.extend(ordered_rows)

        baseline = (
            ((ok["attack"] == "rotation") & np.isclose(ok["strength"].astype(float), 0.0))
            | ((ok["attack"] == "scale") & np.isclose(ok["strength"].astype(float), 1.0))
            | ((ok["attack"] == "translation") & np.isclose(ok["strength"].astype(float), 0.0))
            | (
                ok["attack"].isin(["object_add", "object_delete", "clip", "merge"])
                & np.isclose(ok["strength"].astype(float), 0.0)
            )
        )
        nonbaseline = ok[~baseline].copy()
        if nonbaseline.empty:
            raise RuntimeError(f"No successful non-baseline robustness rows for {exp_id}")
        robust_units = (
            ok.groupby(["identity", "attack", "strength"], as_index=False)
            .agg(nc=("nc", "mean"), ber=("ber", "mean"), passed=("passed", "mean"))
        )
        time_means = timing_df.mean(numeric_only=True)
        ablation_row: Dict[str, object] = {
            "evaluation_seed": int(seed),
            "study_split": study_split_name or "all",
            "threshold": float(nc_threshold),
            "exp_id": exp_id,
            "exp_name": exp_name,
            "description": experiment["description"],
            "channels": ",".join(experiment["channels"]),
            "num_channels": len(experiment["channels"]),
            "lambda_consistency": experiment["lambda_consistency"],
            "lambda_triplet": experiment["lambda_triplet"],
            "identity_count": len(identities),
            "robust_nc_mean": float(robust_units["nc"].mean()),
            "robust_nc_std": float(robust_units["nc"].std(ddof=1))
            if len(robust_units) > 1
            else 0.0,
            "robust_nc_min": float(robust_units["nc"].min()),
            "robust_ber_mean": float(robust_units["ber"].mean()),
            "robust_pass_rate": float(robust_units["passed"].mean()),
            "robustness_aggregation_unit_count": int(len(robust_units)),
            "robustness_aggregation_unit": "all identity x attack family x strength conditions, including clean; masks averaged first",
            "completed_attack_evaluations": int(len(ok)),
            "failed_attack_evaluations": int((robust_df["status"] == "error").sum()),
            "skipped_attack_evaluations": int((robust_df["status"] == "skipped").sum()),
            **unique_summary,
            "zero_watermark_time_including_read_mean_s": float(
                time_means["zero_watermark_generation_time_s_including_read"]
            ),
            "zero_watermark_time_excluding_read_mean_s": float(
                time_means["zero_watermark_generation_time_s_excluding_read"]
            ),
            "feature_extract_time_median_mean_s": float(
                time_means["feature_extract_time_median_s"]
            ),
            "selected_channel_tensor_bytes_mean": float(
                time_means["selected_channel_tensor_bytes"]
            ),
            "encoder_parameter_bytes": parameter_bytes,
            "registered_zero_watermark_payload_bytes": int(math.ceil(bit_length / 8)),
            "registry_file_bytes": registry_path.stat().st_size,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "watermark_sha256": bit_sha256(watermark_bits),
        }
        all_ablation_rows.append(ablation_row)
        (exp_dir / "summary.json").write_text(
            json.dumps(ablation_row, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    ablation_path = output / "ablation_results.csv"
    pd.DataFrame(all_ablation_rows).sort_values("exp_id").to_csv(
        ablation_path, index=False, encoding="utf-8-sig"
    )
    timing_path = output / "timing_space_all.csv"
    pd.DataFrame(all_timing_rows).to_csv(timing_path, index=False, encoding="utf-8-sig")
    robustness_path = output / "robustness_rows_all.csv"
    pd.DataFrame(all_robust_rows).to_csv(
        robustness_path, index=False, encoding="utf-8-sig"
    )
    uniqueness_path = output / "uniqueness_rows_all.csv"
    pd.DataFrame(all_unique_rows).to_csv(
        uniqueness_path, index=False, encoding="utf-8-sig"
    )
    return {
        "ablation_results": ablation_path,
        "timing_space": timing_path,
        "robustness_rows": robustness_path,
        "uniqueness_rows": uniqueness_path,
    }
