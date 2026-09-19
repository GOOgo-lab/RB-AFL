# Generated outputs

All artifacts are generated under the configured new run root.

- `prepared/`: equation-based tensors and hashed manifest, schema 3.
- `split/`: 50/20/41 assignment and provenance audit.
- `models/seed_*/E*/`: v1.1 checkpoints and histories.
- `calibration/`: augmented genuine/base impostor diagnostics; fixed 0.75 summary/lock; original watermark hash.
- `attack_cache/`: four-family paired tensors, new cache version.
- `evaluation/seed_*/`: per-attack, registration, uniqueness and ablation rows.
- `uniqueness/`: 8,200 unordered test pair scores and summaries; `discussion_clean_self_scores.csv` (410 rows), `discussion_far_frr_curve.csv`, `discussion_diagnostic.json` with the distinct Section 5 cohorts.
- `efficiency/`: repeated measured timing rows and environment metadata.
- `reports/`: per-seed ablations, across-seed summaries and paired statistics. Feature extraction is stored in seconds; convert to ms for Table 1.
- `figures/`: diagnostic calibration, test uniqueness, E5-only robustness and runtime plots. These are not all six-method manuscript figures.

Raw counts and dependence assumptions must accompany summaries. No reported manuscript values are inserted by the code.
