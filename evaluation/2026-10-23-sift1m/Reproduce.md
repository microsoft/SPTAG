# SIFT1M reproduction instructions

The current [setup and run guide](../2026-10-28-sift1m/Reproduce.md) uses
`PostingPageLimit=4`, `SearchPostingPageLimit=4` and `BufferLength=4`, and
includes the TiKV client-lifecycle fix.

The [2026-10-23 report](README.md) remains a historical record of partial
`12/12/8` measurements. Its [configuration](benchmark.ini) retains those
settings. Use the current guide for installation, data preparation and cluster
setup; explicitly select `12/12/8` only when studying the historical workload.
