# Validation scope

The revision is an audited implementation, not a verified numerical reproduction.

Tests cover analytic point/line/polygon fields, midpoint KDE and maximum normalization, median ties and dimension guards, sparse watermark XOR/NC, config/Z tampering, fixed threshold despite high FAR, calibration entry regression, user/center signature tampering, identity holdout, cache provenance, unordered pair counts, and a one-epoch two-identity synthetic CPU training/register/verify round trip.

See TEST_RESULTS.txt for the latest executed suite. The local audit used Windows, Python 3.12, PyTorch 2.2.2+cpu, Shapely 2.1.2 and GeoPandas 1.1.1, not the manuscript GPU environment. MKL_THREADING_LAYER=SEQUENTIAL resolved conflicting local OpenMP runtimes. A workspace-only temporary-directory adapter was needed for Python 3.12 Windows 0700 ACL behavior under the audit sandbox; it changes test scratch-directory creation, not algorithm computations. The normal unittest entry remains available for ordinary installations.

No 111-identity training run, final ablation table, baseline comparison, GPU timing reproduction or independent registration-center deployment was performed. CI workflow configuration is included; a remote CI run was not triggered.
