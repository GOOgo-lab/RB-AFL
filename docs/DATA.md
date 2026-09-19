# Data preparation and provenance

## Identity layout

The frozen protocol requires exactly 111 independent geographic-vector identities. A file discovered below `data/raw/` becomes one identity; nested relative path components are included in its generated identity string. Shapefile sidecars belong to the same identity and must not be counted separately.

Before the split is generated, remove renamed copies, alternative exports, spatial subsets, or multiple formats of the same source. The public splitter rejects byte-identical source bundles, but it does not infer that two clipped or re-encoded files come from the same query or region. Such related files must therefore be consolidated into one identity before running the frozen split. Record `group_id` in the public provenance table for audit purposes; every identity supplied to the splitter must represent an independent group.

Recommended public `dataset_manifest.csv` fields are:

```text
identity_id,group_id,source_relpath,source_sha256,source_format,source_url,
retrieved_utc,license,crs,feature_count,point_layer_count,line_layer_count,
polygon_layer_count
```

Do not publish workstation absolute paths. Publish the manifest, retrieval scripts/queries, and split file alongside the paper.

## OpenStreetMap

If identities are derived from OpenStreetMap, provide attribution to **© OpenStreetMap contributors** and link to the [Open Database License](https://opendatacommons.org/licenses/odbl/). Record the exact snapshot or retrieval date, provider endpoint, Overpass query or bounding box, tag filters, geometry conversion, CRS, and preprocessing steps. If an OSM-derived database is redistributed, comply with the ODbL notice and share-alike requirements.

The code archive intentionally does not redistribute OSM extracts. Public OSM data should not be described as unavailable because of privacy unless a particular processed artifact genuinely contains restricted information.

## Watermark and donor data

The 256-bit watermark must contain exactly 45 one bits and 211 zero bits after conversion. Record its license and SHA-256. No merge donor is required for the four default manuscript attack families. Optional historical merge/object-add utilities require independently licensed donor data.

## Prepared cache

`prepare` writes base and augmented four-channel tensors below `runs/.../prepared/`. This cache is derived data, can be large, and is ignored by Git. Do not edit its manifest after the split is frozen. The split audit and model checkpoints bind subsequent stages to its hash.
