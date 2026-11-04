# Evaluation reports

[Documentation index](../docs/README.md) |
[Benchmark methodology](../docs/Benchmarking.md)

This directory contains experiments, not a portable production deployment.
Do not run historical cleanup or disk-preparation commands on shared machines
or existing data.

| Record | Scope | Status |
| --- | --- | --- |
| [2026-11-03 SIFT10M TiKV single-layer results](2026-11-03-sift10m-tikv-layer1/README.md) | FOLLY=ON; 80 build, 16 insert, 48 maintenance, 32 search threads; matching FILEIO workload | Completed; 22 measured rounds and exact 10M reload/count validation passed |
| [2026-11-03 SIFT10M FILEIO single-layer results](2026-11-03-sift10m-fileio-layer1/README.md) | FOLLY=ON; 80 build threads; 9.9M + ten 10K inserts; 10K queries/pass, 32 search threads | Completed; exact 10M final reload/count validation passed; measured source fixes are local-only |
| [2026-10-28 SIFT1M results](2026-10-28-sift1m/README.md) | Complete FILEIO baseline and post-fix TiKV build/update runs; [setup and run instructions](2026-10-28-sift1m/Reproduce.md) | Ten batches and 1M final vectors per backend; different executable revisions; single host |
| [2026-10-23 SIFT1M measurements](2026-10-23-sift1m/README.md) | Historical `12/12/8` FILEIO update measurements and TiKV queries | Partial measurements, superseded as the current entry point |
| [Historical 1M observations](historical/TiKV-1M.md) | Three old runs, single-machine multi-process TiKV | Revision and raw-log provenance incomplete; heap-bug impact unknown |
| [Historical 10M observations](historical/TiKV-10M.md) | Old multi-process TiKV run | Revision and cross-scale comparability incomplete; no post-fix 10M run |
| [2026-04-23 evaluation](2026-04-23/README.md) | Dated SIFT1B update experiments and subsequent analysis | Machine-specific historical record; distinguish local and three-host runs |
| [SPIRE distributed construction](distributed/Readme.md) | Research associated with arXiv:2512.17264; partition, shuffle, and parallel construction | Experimental, manually adapted workflow; not a full paper reproduction package |
| [SPFresh paper artifacts](../Script_AE/README.md) | Paper figure reproduction | Historical dedicated-machine artifact evaluation |

## New results

The latest [FILEIO single-layer report](2026-11-03-sift10m-fileio-layer1/README.md)
records successful completion/count validation and measured-source limitations.
The complete SIFT1M guide remains the setup entry point. General operating
instructions live in `docs/Benchmarking.md`; dated configurations, small result
tables, and diagnostic tools belong alongside their report in this directory.

Use the [report template](ReportTemplate.md), preserve the full configuration
and raw outputs, and link them from the report. Keep large/private datasets
outside Git. Store local runs under `benchmark-output` (ignored), or publish
shareable raw artifacts in an appropriate release/artifact store.

Do not substitute an unknown historical commit, toolchain version, image
digest, or random seed with a guessed value. Mark missing provenance explicitly
and avoid using those results as a release acceptance threshold.

## Machine-specific files

Dated scripts retain their original host/path assumptions for traceability.
Before use, audit their host list, data root, images, replication factor, and
destructive operations. Start with `--help` where available. Current user-facing
guides must use explicit environment parameters rather than copied personal
hostnames.
