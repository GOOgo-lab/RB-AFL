# RB-AFL: manuscript-aligned revision 1.1.0

This is a source-code audit revision for the supplied Chinese manuscript, not a release of independently reproduced manuscript results. Read [the alignment report](docs/MANUSCRIPT_ALIGNMENT.md) before running or citing it.

The previous public archive disagreed with the paper in geometric-field formulas, copyright-watermark weight, feature quantization, and threshold policy. Historical V12.2 source and saved registry records confirm **median quantization**. Historical results supplied for audit use different cohorts and thresholds; they are not results of this revision.

## Implemented protocol

- 111 independent identities: 50 training, 20 calibration diagnostics, 41 final test; ten seeds 20260730–20260739.
- Four input channels in order **occupancy, distance, orientation, density**. Section 3.1 equations are evaluated from vector geometry at grid-cell centers.
- Five convolution blocks (32/64/128/192/256), global average pooling, 256→256 projection and L2 normalization.
- Identity classification + cosine consistency + Euclidean triplet loss; E5 weights 1 and 1, triplet margin 0.8.
- Original 256-bit copyright watermark: **45 ones and 211 zeros**, not a rebalanced image.
- `B = (F >= median(F))`, `Z = W XOR B`, recovered watermark `W' = Z XOR B'`.
- NC cosine similarity of binary copyright/recovered watermarks. Zero reference is rejected; zero recovery always fails.
- **Predeclared threshold 0.75**. `calibrate` reports diagnostic curves and locks that value; it never chooses a better threshold from data.
- Rotation, uniform scale, translation and object deletion. Additional historical attack implementations remain utilities but are excluded from the manuscript run.
- Exactly 8,200 unordered E5 test identity-pair scores across ten seeds.

## What remains unresolved

The paper omits numeric distance/occupancy bandwidths and PCA preprocessing details. This revision documents explicit defaults; they are not verified historical experiment settings. The scale plot includes an invalid zero factor and the translation plot says metres whereas the historical code uses dataset-span fractions. Five comparison-method implementations/results are absent. The manuscript's stated clean-self FRR claim is inconsistent with deterministic XOR recovery. See the alignment report for required manuscript corrections and reruns.

**Do not reuse previous tensors, checkpoints or registry records.** This revision changes features, invalidates old model/cache schemas and uses a new run directory. Runtime and robustness numbers must be remeasured.

## Installation and checks

Python 3.10 or 3.11 is recommended for the reported experiment environment. The audit was tested separately on Windows with Python 3.12 and CPU PyTorch 2.2.2; this does not validate the manuscript's GPU timings.

```bash
python -m pip install -e ".[full,test]"
python -m rbafl validate-config
python -m unittest discover -s tests -v
```

The active equation-based builder does not require Rasterio. Rasterio is retained in the full extra only for explicitly named legacy raster builders. Windows no longer fails at import on the Unix-only `resource` module. Large multiprocess workloads still need validation on the deployment machine.

## Inputs and execution

Place exactly 111 source identities under `data/raw/` and the **actual original** watermark at `data/watermark.png`. A random 45-one image is not a substitute. Provide a source manifest and the original identity assignment if available; a newly generated split is not evidence of the historical split. See [DATA](docs/DATA.md).

```bash
python -m rbafl all --config configs/protocol_50_20_41.json
```

Stages can be run separately: `prepare`, `split`, `validate-split`, `train-e5`, `calibrate`, `train-all`, `cache-attacks`, `evaluate`, `uniqueness`, `efficiency`, `aggregate`, `figures`, `manifest`. The same config must be used throughout. No donor map is required for the four default attack families.

The output contains measured data only. No manuscript numbers or baseline curves are hard-coded. The archive contains no raw vectors, trained weights or original watermark.

## Registration and center signatures

Unsigned evaluation records measure content matching only. A user signature and a separate center envelope are implemented as local reference operations; a real center must supply independent identity/provenance review, conflict checks, a trusted key and timestamp service.

```bash
python scripts/generate_signing_key.py --private-key keys/user.key --public-key keys/user.pub.pem
python scripts/register_watermark.py --vector data/raw/example.geojson --watermark data/watermark.png --checkpoint runs/manuscript_revision_v1_1/models/seed_20260730/E5_proposed/best.pt --registry registry.json --private-key keys/user.key --signer-id author --owner-id owner
python scripts/generate_signing_key.py --private-key keys/center.key --public-key keys/center.pub.pem
python scripts/issue_center_record.py --registry registry.json --record example --user-public-key keys/user.pub.pem --center-private-key keys/center.key --center-id local-demo --certificate-reference demo-only --output center_record.json
python scripts/verify_watermark.py --vector data/raw/example.geojson --watermark data/watermark.png --checkpoint runs/manuscript_revision_v1_1/models/seed_20260730/E5_proposed/best.pt --registry registry.json --record example --threshold-file runs/manuscript_revision_v1_1/calibration/threshold_summary.json --public-key keys/user.pub.pem --center-record center_record.json --center-public-key keys/center.pub.pem
```

Self-generated center keys illustrate the mechanism; they are not an independent authority. Verification reports `content_match_only`, `user_signature_only`, or `center_and_user_signature` explicitly.

## Release contents

[Protocol](docs/PROTOCOL.md) · [Outputs](docs/OUTPUTS.md) · [Validation](docs/RELEASE_VALIDATION.md) · [Release checklist](docs/RELEASE_CHECKLIST.md)

The inherited MIT license covers code only. Complete the author/repository identifiers in `CITATION.cff` before release. No repository was uploaded or published during this audit.

## Real-data smoke test

Supply two or three local vector datasets and your experimental watermark. Data and watermark are not included.

```bash
python smoke_test_real.py --repo . --inputs "data/raw/map_a.shp" "data/raw/map_b.shp" "data/raw/map_c.shp" --watermark data/watermark.png
```

The default is CPU, 256 x 256 fields and one E5 training epoch. The test copies inputs, checks training/checkpoint loading, clean registration/recovery, four attack families, integrity and signatures. Each execution writes a new directory under `smoke_runs/`. Attack authentication outcomes are recorded separately from execution success.

This is a pipeline check, not a generalization evaluation or numerical reproduction of the manuscript. The tiny internal validation partition can contain equivalent tensors after geometric normalization. See [the smoke-test guide](docs/SMOKE_TEST.md).
