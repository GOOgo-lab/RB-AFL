# Running the revision

Install the full extra, validate the config, run tests, then run `python -m rbafl all`. See README for staged commands. Use a fresh run directory and preserve all protocol locks, identity split, raw CSVs and checkpoints. Do not reuse historical weights: the input fields changed.

Exact Gaussian representative-point sums can cost substantially more than the historical centroid histogram. Query distances use a spatial tree; KDE is computed in bounded batches with separable Gaussian factors. Neither the historical runtimes nor hardware throughput are guaranteed. Run efficiency measurements on the actual reported environment.

The source archive does not contain the 111 source identities or original watermark. Historical 94-identity results are a different study. Source-code tests and a tiny synthetic CPU training/register/verify round trip are execution checks, not a reproduction of scientific results.

Windows imports are supported. Large process-pool scheduling has not been benchmarked here. Linux remains a suitable deployment option. Keep private signing keys and unpublished raw data out of the public repository.
