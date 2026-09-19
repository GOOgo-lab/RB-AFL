# Real-data smoke test

Run `python smoke_test_real.py --help` for options. Install the full dependencies with `python -m pip install -e ".[full,test]"`. Supply exactly two or three distinct SHP, GeoJSON or single-layer GeoPackage files and a nonconstant watermark image. Keep Shapefile sidecars together.

Defaults: CPU, one thread, 256 x 256 fields, one training epoch, two augmentations per identity, median feature quantization and NC threshold 0.75. Use `--grid-size 64` for a faster diagnostic. The script does not download data or modify source inputs.

Outputs include smoke_report.json/txt, input_manifest.json (local paths and hashes), clean_results.json, attack_results.json, model checkpoints, tensors, recovered images and temporary test signatures. Do not commit generated reports, input copies, private keys or weights by default. Exit code 0 means the pipeline assertions passed; it does not require all attacked cases to authenticate.

The four attacks are rotation 30 degrees, scale 1.3, translation by a 0.2 coordinate-span fraction, and deletion of approximately 30 percent of objects. Coordinates are not reprojected; translation is not a metric-distance test.

Only three internal validation samples exist with three input identities; positives in this partition can be the anchor itself. Equivalent tensors can arise after canonicalization. Validation accuracy and consistency loss are not evidence of independent generalization. This test does not estimate false acceptance or replace the full 50/20/41 protocol.
