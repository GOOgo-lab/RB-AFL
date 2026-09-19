# Frozen revision protocol

The configuration fixes 50/20/41 identities, ten seeds, 256-dimensional embeddings, a 45-one watermark, median feature thresholds and predeclared NC=0.75. The 20-identity group measures diagnostic performance, not data-driven threshold selection. The utility `calibrate_threshold` retains its optional automatic mode for historical unit tests; the manuscript CLI passes fixed_threshold=0.75 and rejects other policies.

Geometric preprocessing and all explicit unresolved choices are specified in [MANUSCRIPT_ALIGNMENT](MANUSCRIPT_ALIGNMENT.md). Exact distances and Gaussian KDE replace the raster approximation. All old tensor/model/registry artifacts are incompatible.

`attack_plan()` contains only four families. Ten fixed deletion masks per positive strength are inherited from the public source, not established by the manuscript. Masks are paired across all seeds/ablations. Robust NC averages masks per identity/condition first and then all configured conditions, including clean settings. Seed summaries report SD across ten seeds.

Uniqueness removes self pairs and directional duplicates: 10*C(41,2)=8,200. Symmetry is checked; reduction takes the max. Repeated seeds and shared identities make pair scores dependent; binomial bounds are descriptive score-level summaries.

The discussion export is separate: 410 clean test self scores and 3,800 calibration base-impostor scores. It does not tune the threshold and must not be labelled attacked-sample robustness. Calibration augmented genuine scores remain separately available.

Never retune thresholds, splits or attacks after inspecting final test results. Record the actual original identity assignment and data provenance before claiming reproduction.
