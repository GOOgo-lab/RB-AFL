"""Calibration-only NC score collection and threshold selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from .data import identities_for_split, resolve_prepared_tensor_path, subset_manifest_by_identities
from .model import extract_embedding, feature_to_bits, load_encoder
from .watermark import (
    ber_score,
    bit_distribution_stats,
    nc_score,
    watermark_image_to_bits,
    xor_bits,
)


def collect_calibration_scores(
    prepared_root: str | Path,
    identity_split_path: str | Path,
    seed_to_e5_checkpoint: Mapping[int, str | Path],
    watermark_path: str | Path,
    output_dir: str | Path,
    *,
    bit_length: int = 256,
    threshold_mode: str = "median",
    device: str = "auto",
    random_watermark_patterns: int = 16,
    watermark_pattern_seed: int = 20260911,
) -> pd.DataFrame:
    """Collect E5 genuine and impostor scores from the 20 calibration identities."""

    root = Path(prepared_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    calibration_ids = identities_for_split(identity_split_path, "calibration")
    if len(calibration_ids) != 20:
        raise RuntimeError(f"Threshold calibration requires exactly 20 identities; got {len(calibration_ids)}")
    manifest = subset_manifest_by_identities(root, calibration_ids)
    watermark_bits, _, _ = watermark_image_to_bits(watermark_path, bit_length)
    if not np.any(watermark_bits):
        raise ValueError("Reference watermark must be nonzero")

    patterns: list[tuple[str, np.ndarray]] = [("actual", watermark_bits.copy())]
    rng = np.random.default_rng(int(watermark_pattern_seed))
    for index in range(max(0, int(random_watermark_patterns))):
        order = rng.permutation(bit_length)
        bits = np.zeros(bit_length, dtype=np.uint8)
        bits[order[: int(watermark_bits.sum())]] = 1
        patterns.append((f"matched_weight_random_{index + 1:02d}", bits))

    rows: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []
    zero_stats_rows: list[dict[str, Any]] = []

    def append_scores(
        *,
        seed: int,
        score_type: str,
        registered_identity: str,
        tested_identity: str,
        sample_type: str,
        registered_bits: np.ndarray,
        tested_bits: np.ndarray,
    ) -> None:
        delta = xor_bits(registered_bits, tested_bits)
        for pattern_id, pattern_bits in patterns:
            recovered = xor_bits(pattern_bits, delta)
            row = {
                "training_seed": int(seed),
                "pattern_id": pattern_id,
                "score_type": score_type,
                "registered_identity": registered_identity,
                "tested_identity": tested_identity,
                "sample_type": sample_type,
                "nc": nc_score(pattern_bits, recovered),
                "ber": ber_score(pattern_bits, recovered),
            }
            pattern_rows.append(row)
            if pattern_id == "actual":
                rows.append({key: value for key, value in row.items() if key != "pattern_id"})

    for seed, checkpoint_value in sorted(seed_to_e5_checkpoint.items()):
        checkpoint_path = Path(checkpoint_value).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"E5 checkpoint missing for seed {seed}: {checkpoint_path}")
        encoder, checkpoint, resolved = load_encoder(checkpoint_path, device)
        experiment = checkpoint["experiment"]
        if str(experiment.get("exp_id")) != "E5":
            raise ValueError(f"Threshold calibration requires E5, got {experiment.get('exp_id')}")
        channel_indices = tuple(int(value) for value in experiment["channel_indices"])

        bits_by_identity: dict[str, list[tuple[str, np.ndarray]]] = {}
        for identity in calibration_ids:
            samples: list[tuple[str, np.ndarray]] = []
            subset = manifest[manifest["identity"].astype(str) == identity]
            for sample in subset.sort_values("sample_type").to_dict("records"):
                tensor_path = resolve_prepared_tensor_path(
                    root,
                    str(sample["identity"]),
                    str(sample["sample_type"]),
                    str(sample["tensor_path"]),
                )
                tensor = np.load(tensor_path, allow_pickle=False)
                embedding = extract_embedding(encoder, tensor, channel_indices, resolved)
                feature_bits = feature_to_bits(embedding, bit_length, threshold_mode)
                samples.append((str(sample["sample_type"]), feature_bits))
            bits_by_identity[identity] = samples

        for registered_identity in calibration_ids:
            registered_samples = bits_by_identity[registered_identity]
            base_matches = [bits for sample_type, bits in registered_samples if sample_type == "base"]
            if len(base_matches) != 1:
                raise RuntimeError(
                    f"Calibration identity {registered_identity} must have exactly one base tensor"
                )
            base_bits = base_matches[0]
            zero_stats_rows.append(
                {
                    "training_seed": int(seed),
                    "registered_identity": registered_identity,
                    **{
                        f"zero_watermark_{key}": value
                        for key, value in bit_distribution_stats(
                            xor_bits(watermark_bits, base_bits)
                        ).items()
                    },
                }
            )
            genuine_samples = [
                (sample_type, bits)
                for sample_type, bits in registered_samples
                if sample_type != "base"
            ] or [("base", base_bits)]
            for sample_type, tested_bits in genuine_samples:
                append_scores(
                    seed=int(seed),
                    score_type="genuine",
                    registered_identity=registered_identity,
                    tested_identity=registered_identity,
                    sample_type=sample_type,
                    registered_bits=base_bits,
                    tested_bits=tested_bits,
                )
            for tested_identity in calibration_ids:
                if tested_identity == registered_identity:
                    continue
                for sample_type, tested_bits in bits_by_identity[tested_identity]:
                    if sample_type != "base":
                        continue
                    append_scores(
                        seed=int(seed),
                        score_type="impostor",
                        registered_identity=registered_identity,
                        tested_identity=tested_identity,
                        sample_type=sample_type,
                        registered_bits=base_bits,
                        tested_bits=tested_bits,
                    )

    result = pd.DataFrame(rows)
    result.to_csv(output / "calibration_scores.csv", index=False, encoding="utf-8")
    pattern_frame = pd.DataFrame(pattern_rows)
    pattern_frame.to_csv(
        output / "watermark_pattern_stability_rows.csv", index=False, encoding="utf-8"
    )
    pattern_summary = (
        pattern_frame.groupby(["pattern_id", "score_type"], as_index=False)
        .agg(
            sample_count=("nc", "count"),
            nc_mean=("nc", "mean"),
            nc_std=("nc", "std"),
            nc_min=("nc", "min"),
            nc_max=("nc", "max"),
            ber_mean=("ber", "mean"),
        )
        .fillna({"nc_std": 0.0})
    )
    pattern_summary.to_csv(
        output / "watermark_pattern_stability_summary.csv", index=False, encoding="utf-8"
    )
    zero_frame = pd.DataFrame(zero_stats_rows)
    zero_frame.to_csv(output / "calibration_zero_watermark_stats.csv", index=False, encoding="utf-8")
    watermark_stats = bit_distribution_stats(watermark_bits)
    score_summary = {
        **{f"copyright_{key}": value for key, value in watermark_stats.items()},
        "calibration_identity_count": len(calibration_ids),
        "training_seed_count": len(seed_to_e5_checkpoint),
        "genuine_score_count": int((result["score_type"] == "genuine").sum()),
        "impostor_score_count": int((result["score_type"] == "impostor").sum()),
        "random_watermark_pattern_count": max(0, int(random_watermark_patterns)),
        "watermark_pattern_seed": int(watermark_pattern_seed),
        "zero_watermark_entropy_mean": float(
            zero_frame["zero_watermark_binary_entropy_bits"].astype(float).mean()
        ),
        "zero_watermark_entropy_min": float(
            zero_frame["zero_watermark_binary_entropy_bits"].astype(float).min()
        ),
        "zero_watermark_all_zero_count": int(
            zero_frame["zero_watermark_is_all_zero"].astype(bool).sum()
        ),
        "zero_watermark_all_one_count": int(
            zero_frame["zero_watermark_is_all_one"].astype(bool).sum()
        ),
    }
    (output / "watermark_and_score_summary.json").write_text(
        json.dumps(score_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    values = np.concatenate([genuine, impostor])
    ranks = stats.rankdata(values, method="average")
    n_positive = genuine.size
    n_negative = impostor.size
    u_value = float(ranks[:n_positive].sum()) - n_positive * (n_positive + 1) / 2.0
    return float(u_value / max(1, n_positive * n_negative))


def _two_sided_exact_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(
        stats.beta.ppf(alpha / 2, successes, total - successes + 1)
    )
    upper = 1.0 if successes == total else float(
        stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes)
    )
    return lower, upper


def calibrate_threshold(
    calibration_scores: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    target_far: float = 0.01,
    threshold_step: float = 0.001,
    fixed_threshold: float | None = None,
) -> dict[str, Any]:
    scores = (
        pd.read_csv(calibration_scores)
        if isinstance(calibration_scores, (str, Path))
        else calibration_scores.copy()
    )
    genuine = scores.loc[scores["score_type"] == "genuine", "nc"].astype(float).to_numpy()
    impostor = scores.loc[scores["score_type"] == "impostor", "nc"].astype(float).to_numpy()
    if genuine.size == 0 or impostor.size == 0:
        raise RuntimeError("Threshold calibration requires genuine and impostor scores")
    if not 0.0 <= target_far <= 1.0:
        raise ValueError("target_far must be in [0,1]")
    if not 0.0 < threshold_step <= 0.1:
        raise ValueError("threshold_step must be in (0,0.1]")

    if not np.isfinite(np.concatenate([genuine, impostor])).all():
        raise ValueError("NC scores must be finite")
    if np.any(np.concatenate([genuine, impostor]) < 0) or np.any(np.concatenate([genuine, impostor]) > 1):
        raise ValueError("NC scores must be in [0,1]")
    if fixed_threshold is not None and not 0 <= fixed_threshold <= 1:
        raise ValueError("fixed_threshold must be in [0,1]")
    curve_rows: list[dict[str, float]] = []
    thresholds = np.arange(0.0, 1.0 + threshold_step / 2.0, threshold_step)
    if fixed_threshold is not None:
        thresholds = np.unique(np.append(thresholds, fixed_threshold))
    for raw_threshold in thresholds:
        threshold = float(round(float(raw_threshold), 10))
        far = float(np.mean(impostor >= threshold))
        frr = float(np.mean(genuine < threshold))
        curve_rows.append(
            {
                "threshold": threshold,
                "far": far,
                "frr": frr,
                "tar": 1.0 - frr,
                "balanced_error": 0.5 * (far + frr),
                "far_minus_frr_abs": abs(far - frr),
            }
        )
    curve = pd.DataFrame(curve_rows)
    if fixed_threshold is not None:
        selected = curve.loc[np.isclose(curve["threshold"], fixed_threshold, atol=1e-12, rtol=0)].iloc[0]
        policy = "predeclared_fixed"
    else:
        eligible = curve[curve["far"] <= float(target_far)]
        if eligible.empty:
            raise RuntimeError("No threshold satisfies FAR target")
        selected = eligible.sort_values(["frr", "threshold", "far"]).iloc[0]
        policy = "target_far_then_minimum_frr"
    eer = curve.sort_values(
        ["far_minus_frr_abs", "balanced_error", "threshold"]
    ).iloc[0]
    selected_threshold = float(selected["threshold"])
    false_accepts = int(np.sum(impostor >= selected_threshold))
    false_rejects = int(np.sum(genuine < selected_threshold))
    far_interval = _two_sided_exact_interval(false_accepts, int(impostor.size))
    frr_interval = _two_sided_exact_interval(false_rejects, int(genuine.size))
    summary: dict[str, Any] = {
        "selection_policy": policy,
        "target_far": float(target_far),
        "threshold_step": float(threshold_step),
        "selected_threshold": selected_threshold,
        "calibration_far": float(selected["far"]),
        "calibration_frr": float(selected["frr"]),
        "calibration_tar": float(selected["tar"]),
        "calibration_auc": _auc(genuine, impostor),
        "eer_threshold": float(eer["threshold"]),
        "eer_far": float(eer["far"]),
        "eer_frr": float(eer["frr"]),
        "genuine_score_count": int(genuine.size),
        "impostor_score_count": int(impostor.size),
        "false_accept_count": false_accepts,
        "false_reject_count": false_rejects,
        "far_95ci_low": far_interval[0],
        "far_95ci_high": far_interval[1],
        "frr_95ci_low": frr_interval[0],
        "frr_95ci_high": frr_interval[1],
        "genuine_nc_mean": float(genuine.mean()),
        "genuine_nc_std": float(genuine.std(ddof=1)) if genuine.size > 1 else 0.0,
        "genuine_nc_min": float(genuine.min()),
        "impostor_nc_mean": float(impostor.mean()),
        "impostor_nc_std": float(impostor.std(ddof=1)) if impostor.size > 1 else 0.0,
        "impostor_nc_max": float(impostor.max()),
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    curve.to_csv(output / "threshold_curve.csv", index=False, encoding="utf-8")
    (output / "threshold_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
