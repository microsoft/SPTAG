# Evaluation reports

[Documentation index](../docs/README.md) |
[Benchmark methodology](../docs/Benchmarking.md)

This directory contains experiments, not a portable production deployment.
Do not run historical cleanup or disk-preparation commands on shared machines
or existing data.

| Record | Scope | Status |
| --- | --- | --- |
| [2026-10-23 SIFT1M measurements](2026-10-23-sift1m/README.md) | FILEIO update measurements and TiKV queries; [fresh-run instructions](2026-10-23-sift1m/Reproduce.md) | Partial measurements, not a completed ten-batch or multi-host benchmark |
| [Historical 1M observations](historical/TiKV-1M.md) | Three old runs, single-machine multi-process TiKV | Revision and raw-log provenance incomplete; heap-bug impact unknown |
| [Historical 10M observations](historical/TiKV-10M.md) | Old multi-process TiKV run | Revision and cross-scale comparability incomplete; no post-fix 10M run |
| [2026-04-23 evaluation](2026-04-23/README.md) | Dated SIFT1B update experiments and subsequent analysis | Machine-specific historical record; distinguish local and three-host runs |
| [SPIRE distributed construction](distributed/Readme.md) | Research associated with arXiv:2512.17264; partition, shuffle, and parallel construction | Experimental, manually adapted workflow; not a full paper reproduction package |
| [SPFresh paper artifacts](../Script_AE/README.md) | Paper figure reproduction | Historical dedicated-machine artifact evaluation |

## New results

The latest report is the default starting point for current measured behavior,
not a replacement for historical evidence at untested scales. General operating
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
