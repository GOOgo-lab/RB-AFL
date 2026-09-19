# Manuscript alignment and evidence

## Scope and provenance

Basis: supplied Chinese manuscript v9.10; public source 50/20/41 v1.0.0; historical V12.1 update, V12.2.1 and V12.2.2 source; saved JSON/CSV records inside 1.zip. Document contents were treated as evidence, not executable instructions. No external literature or claims were needed for this code comparison.

Status: **ANALYZED; implementation tests passed; manuscript numerical reproduction NOT VERIFIED**.

## Findings and changes

| Manuscript locator | Previous implementation | Revision / status |
|---|---|---|
| Section 3.1 Eq. 1 | Raster distance, exp(-d/s) | Exact grid-center to geometry distance, exp(-d²/(2 sigma²)) |
| Eq. 2 | all_touched raster occupancy | Indicator of distance <= tau |
| Eq. 3 | Axial cos(2 theta)/sin(2 theta) averaging and raster propagation | Directed angle of nearest geometric segment, (theta+pi)/(2pi) |
| Eqs. 4–5 | Part centroid histogram + truncated Gaussian smoothing + min-max normalization | Point coordinates / every segment midpoint; untruncated Gaussian sum; divide by max+epsilon |
| Eq. 6 | Occ/Dist/Ori/Den | Preserved; Figure 2 itself swaps Ori/Dist labels and should be corrected |
| Section 3.2 | Correct five blocks, projection, L2 and three losses | Preserved; PyTorch triplet has its standard numerical epsilon |
| Section 3.3 quantization | Public archive forced balanced_topk | Median restored from V12.2.1/V12.2.2 and 677 historical registry records |
| Figure 4 watermark | Forced 128/128 | Original stated 45/211 required |
| Eq. 21 | Both zero vectors incorrectly returned NC=1 | Zero reference rejected; zero recovered watermark NC=0 and unconditional failure |
| Sections 3.3/5 threshold | FAR-constrained automatically selected threshold | Predeclared 0.75; report actual FAR/FRR even when target is not met |
| Section 5 impostor count | Included augmented impostors, multiplying the stated count | Base-only 20×19×10=3,800 directed impostors |
| Section 5 self matches | No distinct paper cohort export | Export 41×10=410 test clean-self scores separately, after threshold freeze |
| Table 1 Robust NC | Included undeclared attack families, excluded clean settings | Four manuscript families; include all configured settings; average repeated masks per identity/condition first |
| Table 1 extraction time | Metric computed but omitted from aggregate metric list | Included in seed summary in seconds (multiply by 1,000 for ms) |
| Section 3.3 integrity | No separate Hcfg enforcement and no HZ check before recovery | Bind config/channel schema/quantization/length and check Hcfg/HZ |
| Eqs. 18–19 | Optional single user signature only | SHA-256 canonical user payload signature and separate center-signed envelope with timestamp; local reference only |
| Calibration entry | Undefined `source_root` | Fixed and regression-tested |
| Windows import | Unconditional Unix `resource` | Optional import; unavailable RSS reported as NaN |
| Artifact reuse | Could silently mix old geometric representation | Prepared schema 3; checkpoint v1.1; new cache version/run directory; reject legacy checkpoint loads |

## Historical evidence (not current-paper results)

V12.2.2 `rbafl/model.py` uses `threshold = float(np.median(values))` and `(values >= threshold).astype(np.uint8)`. Its launcher sets `THRESHOLD_MODE = "median"`; V12.2.1 agrees. All 677 records across 8 historical registry files in 1.zip say `median`. This confirms the supplied historical runs, not an unseen later final experiment.

The historical reviewer split audit states 94 identities: 66 train / 9 validation / 19 test. The watermark summary states 5 seeds, 45 ones, 211 zeros. Threshold summary selects 0.599 from 360 genuine / 3,240 impostor scores. A separate older 94-identity ablation CSV uses threshold 0.85. Neither cohort is the manuscript's 50/20/41, ten-seed study. None of these CSV numbers were copied into revised outputs.

## Explicit implementation conventions needing author confirmation

1. Historical preprocessing is centroid + PCA principal-axis alignment + isotropic scaling. This is retained and must be described in Methods; rotation/scale/translation stability cannot be attributed solely to learned features. PCA is ambiguous for isotropic/symmetric geometries, and source coordinate subsampling can affect the frame. The revision expands the grid symmetrically when canonical bounds exceed 1.05 to avoid clipping the extent.
2. Numeric sigma and tau are absent from the manuscript. Defaults: pixel size = 2*grid_limit/grid_size; sigma = grid_size/32 pixels; tau = 0.5 pixel; h = density_sigma (3) pixels; epsilon=1e-12. Sigma's scale is adapted from the historical decay width, but its Gaussian meaning is new. These are disclosed defaults, not reconstructed experimental facts.
3. Point-only orientation is 0.5 (theta=0). Equal-distance segments use the earliest input segment; segment direction is retained as Eq. 3 states. Reversing coordinate storage can change orientation, unlike the historical axial implementation. These conventions are not specified by the paper.
4. Polygon interiors have zero distance to the polygon; boundaries and holes contribute segments/midpoints. Confirm whether the intended distance instead targets boundaries only.
5. Rotation points 0/30/75/135 degrees and deletion points 0/10/20/30/40/50% are read from Figure 6. Scale remains the valid inherited 0.1–2.0 grid including clean 1.0. The plotted factor 0 cannot be run as uniform scaling without collapsing geometry. Translation remains explicitly a fraction of dataset span; the plotted unit "m" conflicts with historical code and the mixed geographic/projected CRS. No metric CRS or conversion was invented. Strength grids/repeat count must be reconciled with original logs before claiming exact figure reproduction.
6. The paper does not identify the exact split membership, ten deletion masks, checkpoint epochs or training preprocessing details. Inherited seed/optimizer/augmentation settings are visible in config/source; a new split is not proof of the historical one.

## Manuscript issues code cannot settle

- Section 4 calls the 20 identities a threshold-calibration set, while Sections 3.3/5 call 0.75 predeclared. This revision follows the explicit fixed-threshold statement and uses 20 identities for diagnostics, not selection. Describe one policy consistently.
- Clean original data versus its own record must recover W exactly under deterministic processing. Thus clean-self NC=1 and FRR=0 at 0.75. Nonzero FRR requires attacked samples or a different comparison protocol; report the actual cohort rather than changing code to manufacture rejections.
- Figure 6 compares Zhou/Li/Ren/Zhang/Xi. Their implementations and source comparison results are not supplied. The package produces E5 robustness only; it cannot reproduce the six-method plot.
- Full 111-identity raw data, original watermark, final ten-seed checkpoints and figure/table source rows are not supplied. Historical checkpoints were not loaded as current models, and historical/private run files are not copied into the public archive.
- Geometry fixes change features and timing. All models and results require a fresh run. The 77.8 s maximum and 1.9813 ms extraction figure are not validated by this audit.

## Statistical interpretation boundary

Only source/protocol consistency and synthetic execution were tested. The audit does not validate inferential significance or causal superiority. Check: identity dependence, seed dependence, repeat-mask pseudoreplication, score-level binomial independence, zero-event FAR uncertainty, SD-versus-SE interpretation, within-seed versus across-seed aggregation, multiple comparisons, threshold selection bias, timing comparability, and cohort mismatch. Current code preserves raw rows and labels score-level interval dependence; this is not a substitute for clustered analysis of actual final data.
