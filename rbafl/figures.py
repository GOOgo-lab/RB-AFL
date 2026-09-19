"""Generate manuscript-facing figures exclusively from frozen run artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Figure generation requires matplotlib (install the 'full' extra)") from exc
    return plt


def _save(fig, output: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")


def _ci95(values: pd.Series) -> tuple[float, float, float]:
    array = values.astype(float).to_numpy()
    mean = float(array.mean())
    if array.size < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, array.size - 1) * array.std(ddof=1) / math.sqrt(array.size))
    return mean, mean - half, mean + half


def make_threshold_figure(calibration_dir: Path, output: Path) -> None:
    plt = _matplotlib()
    scores = pd.read_csv(calibration_dir / "calibration_scores.csv")
    curve = pd.read_csv(calibration_dir / "threshold_curve.csv")
    summary = json.loads((calibration_dir / "threshold_summary.json").read_text(encoding="utf-8"))
    threshold = float(summary["selected_threshold"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for score_type, color in (("impostor", "#dd8452"), ("genuine", "#4c72b0")):
        values = scores.loc[scores["score_type"] == score_type, "nc"].astype(float)
        axes[0].hist(values, bins=40, density=True, alpha=0.65, label=f"{score_type} (n={len(values):,})", color=color)
    axes[0].axvline(threshold, color="crimson", linestyle="--", label=f"$T_{{NC}}={threshold:g}$")
    axes[0].set(xlabel="NC score", ylabel="Density", title="(a) Calibration score distributions")
    axes[0].legend(frameon=False)
    axes[1].plot(curve["threshold"], curve["far"], label="FAR", color="#dd8452")
    axes[1].plot(curve["threshold"], curve["frr"], label="FRR", color="#4c72b0")
    axes[1].axvline(threshold, color="crimson", linestyle="--", label=f"$T_{{NC}}={threshold:g}$")
    axes[1].set(xlabel="Decision threshold $T_{NC}$", ylabel="Rate", ylim=(0, 1), title="(b) FAR–FRR operating characteristics")
    axes[1].legend(frameon=False)
    _save(fig, output, "threshold_calibration")
    plt.close(fig)


def make_uniqueness_figure(uniqueness_dir: Path, output: Path) -> None:
    plt = _matplotlib()
    pairs = pd.read_csv(uniqueness_dir / "uniqueness_unordered_pairs.csv")
    if len(pairs) != 8200:
        raise RuntimeError(f"Uniqueness figure requires exactly 8,200 pair scores; got {len(pairs)}")
    summary = json.loads((uniqueness_dir / "uniqueness_summary.json").read_text(encoding="utf-8"))
    threshold = float(summary["threshold"])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(pairs["nc"].astype(float), bins=40, color="#9ecae1", edgecolor="#5b7c8d")
    ax.axvline(threshold, color="crimson", linestyle="--", label=f"$T_{{NC}}={threshold:g}$")
    ax.set(xlabel="NC score", ylabel="Count", title="Unordered cross-identity uniqueness scores")
    ax.text(
        0.98,
        0.96,
        f"N={len(pairs):,}\nmax={float(summary['maximum_nc']):.4f}\n"
        f"NC≥$T_{{NC}}$: {int(summary['exceed_count'])}\nFAR={float(summary['empirical_far']):.4%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.legend(frameon=False)
    _save(fig, output, "uniqueness_scores")
    plt.close(fig)


def make_robustness_figure(evaluation_root: Path, threshold: float, output: Path) -> None:
    plt = _matplotlib()
    frames = []
    for path in sorted(evaluation_root.glob("seed_*/robustness_rows_all.csv")):
        frame = pd.read_csv(path)
        frame = frame[(frame["exp_id"].astype(str) == "E5") & (frame["status"].astype(str) == "ok")].copy()
        frame["training_seed"] = int(path.parent.name.replace("seed_", ""))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No E5 robustness rows were found")
    raw = pd.concat(frames, ignore_index=True)
    units = raw.groupby(["training_seed", "identity", "attack", "strength"], as_index=False)["nc"].mean()
    attacks = (("rotation", "Rotation (degrees)"), ("translation", "Translation (span fraction)"), ("scale", "Scaling factor"), ("object_delete", "Object deletion ratio"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for ax, (attack, xlabel) in zip(axes.flat, attacks):
        subset = units[units["attack"].astype(str) == attack]
        rows = []
        for strength, group in subset.groupby("strength", sort=True):
            mean, low, high = _ci95(group["nc"])
            rows.append((float(strength), mean, low, high))
        values = np.asarray(rows, dtype=float)
        ax.plot(values[:, 0], values[:, 1], marker="o", color="#4c72b0")
        ax.fill_between(values[:, 0], values[:, 2], values[:, 3], color="#4c72b0", alpha=0.2)
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1)
        ax.set(xlabel=xlabel, ylabel="NC", ylim=(0, 1.02), title=attack.replace("_", " ").title())
    _save(fig, output, "robustness_e5")
    plt.close(fig)


def make_runtime_figure(efficiency_dir: Path, output: Path) -> None:
    plt = _matplotlib()
    raw = pd.read_csv(efficiency_dir / "efficiency_raw.csv")
    warmup = raw["warmup"].astype(str).str.lower().isin({"true", "1"})
    measured = raw[~warmup].copy()
    stages = [
        "vector_read_time_s",
        "channel_construction_time_s",
        "feature_extraction_time_s",
        "quantization_time_s",
        "xor_record_time_s",
    ]
    unit_columns = ["feature_count", "end_to_end_time_s", *stages]
    identity_seed = measured.groupby(["training_seed", "identity"], as_index=False)[unit_columns].median()
    if len(identity_seed) != 410:
        raise RuntimeError(f"Runtime figure requires 41 identities x 10 seeds = 410 units; got {len(identity_seed)}")
    identities = identity_seed.groupby("identity", as_index=False)[unit_columns].median()
    if len(identities) != 41:
        raise RuntimeError(f"Runtime figure requires exactly 41 test identities; got {len(identities)}")
    identities = identities.sort_values("end_to_end_time_s").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    positive = identities[(identities["feature_count"] > 0) & (identities["end_to_end_time_s"] > 0)]
    axes[0].scatter(positive["feature_count"], positive["end_to_end_time_s"], s=22)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Number of vector features (log scale)", ylabel="End-to-end time (s; log scale)", title="(a) Feature count versus runtime")
    bottom = np.zeros(len(identities), dtype=float)
    labels = ["Vector reading", "Channel construction", "Feature extraction", "Quantization", "XOR + record"]
    for column, label in zip(stages, labels):
        values = identities[column].astype(float).to_numpy()
        axes[1].bar(np.arange(len(values)) + 1, values, bottom=bottom, width=0.9, label=label)
        bottom += values
    axes[1].set(xlabel="Test-identity rank (increasing total time)", ylabel="Median stage time (s)", title="(b) Stage-level timing breakdown")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, output, "runtime_breakdown")
    plt.close(fig)


def make_all_figures(paths: Mapping[str, Path]) -> Path:
    output = Path(paths["figures"])
    output.mkdir(parents=True, exist_ok=True)
    threshold_summary = json.loads((Path(paths["threshold"])).read_text(encoding="utf-8"))
    make_threshold_figure(Path(paths["calibration"]), output)
    make_uniqueness_figure(Path(paths["uniqueness"]), output)
    make_robustness_figure(Path(paths["evaluation"]), float(threshold_summary["selected_threshold"]), output)
    make_runtime_figure(Path(paths["efficiency"]), output)
    return output
