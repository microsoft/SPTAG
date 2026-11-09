# SIFT10M FILEIO: 1 posting layer(s), matched measured binary

[Evaluation index](../README.md) | [Reproduce](Reproduce.md) |
[All query rounds](query-rounds.csv) | [Insertions](insertions.csv) |
[Machine-readable results](results.json)

**Status: full workload and independent final10M membership/metadata verified.**
Recorded preparation UTC: `2026-11-09T03:34:33Z`.
This is one independent fresh build, not a same-index read-only retest.

## 1. Provenance and environment

- Measured source: `95bcb20f48f7864072f9443ec91027875382b722`, already published remotely.
- Executable SHA256: `a38242ea9fac9ef9153af9c46ddd8389c57fc65da814d85cb2b13a72ac4c1323`.
- Checker SHA256: `60725cd58978ae86bfe0dc30973c334160c456b09ba17a21f7ee61b3a032ff3e`.
- This run shares the exact executable, checker, runtime library bytes, native
  inputs and truth with the other three matched November 6/9 reports.
  The older November 3 FILEIO1 run is historical and is not this comparison's baseline.
- Release; FOLLY/TBB/TIKV ON, GPU/SPDK/ROCKSDB/RABITQ/LARGEVID OFF.
  Folly `7088e27f4854c9088bcf35b980e54e89fa966e0f`;
  zstd `e47e674cd09583ff0503f0f6defd6d23d8b718d3`.
- GCC11.4, CMake3.31.12, Protobuf3.12.4, gRPC1.30.2; detailed
  [recorded build environment](../2026-11-05-sift10m-tikv-pool16-cq8/software-environment.txt).
- Shared AMD EPYC7V12 host, 96 logical CPUs, two sockets/four NUMA nodes,
  ~1.7TiB RAM and ~7TiB SSD filesystem. No fixed affinity/CPU isolation,
  cache flush or build seed; disk throughput and network RTT were not recorded.
- [Binary/library provenance](binary-provenance.json), [input hashes](input-checksums.json),
  [original manifest](run-manifest.json), [recorded recipe](recorded-recipe/)
  and [artifact hashes](artifacts.sha256) identify the actual measurement.
  Historical absolute paths are evidence only, not reproduction prerequisites.
  The historical Python recipe was an uncommitted snapshot on the original
  publishing checkout; its archived files/hashes, not `recipe_checkout` alone,
  identify it. The measured C++ source remains the fixed committed revision.

## 2. Workload and configuration

| Setting | Value |
| --- | --- |
| Runtime backend / posting layers | FILEIO / 1, plus final in-memory BKT head |
| Dataset / metric | First10M SIFT, UInt8,128D, squared L2, no quantizer |
| Initial / insert / delete | 9,900,000 / ten10,000 batches / 0 |
| Queries / TopK | 10,000 native SIFT queries / 5 |
| Passes | One excluded warmup before each measured pass; 22 measured passes |
| Build threads | SelectHead80 / BuildHead80 / BuildSSDIndex80 |
| Insert / maintenance / query threads | 16 / 48 / 32 |
| Search during insertion | 0; sequential query and foreground-insert phases |
| MaxCheck | 8192 in BuildHead and BuildSSDIndex |
| Posting/Search pages / BufferLength | 4 / 4 / 4 |
| SearchInternalResultNum / LatencyLimit | 64 / 50 |
| Fresh run | Rebuild=true, Resume=-1 |
| Version policy | FILEIO uses local VersionMaps on every posting layer. Local `Deleted` checks run inline in the posting scan and are included in computation time. The layer0-only switch is TiKV-only. The version timing/count columns measure TiKV batch checks, so zero for FILEIO is expected, not a bypass. |

[Portable configuration](benchmark.ini) fixes this profile's storage/layers.
[Recorded input INI](recorded.ini) and [saved effective INI](saved-indexloader.ini)
retain exact measured paths/settings. [Raw JSON](benchmark-output.json) and
[raw log](run.log) are byte-preserved.

## 3. Deployment

No PD or TiKV processes, network clients or distributed VersionMaps are used.

## 4. Measurement definitions

- Query latency is the elapsed `SearchIndex` call per query, in milliseconds.
  QPS is `10000 / query-pass wall seconds` across32 worker threads, including
  worker creation/join; result serialization and recall computation are outside
  this timer. It is not `1000 / mean latency`.
- Per-query percentiles sort10000 latencies and select zero-based
  `floor(N*p)` (P50=5000, P95=9500, P99=9900), without interpolation.
  Head traversal, setup and posting-layer work contribute to total latency;
  marginal component percentiles must not be added as if correlated.
- Every phase uses squared-L2 exact Top5 over that phase's active prefix.
  Ground-truth ties sort by source row ID. Concurrent inserted VIDs can differ
  from source IDs; metadata maps them back. Recall uses native10000 queries
  and five neighbors per query. Truth layout/hashes are in [reproduction](Reproduce.md).
- Each insert timer covers `InsertVectors`, including the background-work
  completion barrier. Per-batch vectors/s =10000/insert seconds. Aggregate
  vectors/s =100000/sum(insert seconds), **not** an average of batch rates.
  Build, query/warmup, clone, load and save are excluded. Therefore insertion
  throughput is not end-to-end pipeline throughput or concurrent-update QPS.
- Warmups and checkpoint work still contribute to process wall time.
  Historical cold/warm round labels do not establish a controlled cold-cache
  measurement: the harness executes a warmup before each measured pass.
  RSS in [resources](resources.txt) measures the benchmark client and excludes
  separately running PD/TiKV servers. No claim about server total RSS is made.

## 5. Results

| Metric | Value |
| --- | --- |
| Initial build seconds | 1199.6283 |
| Benchmark process wall time | 41:52.30 |
| Benchmark maximum RSS KiB | 6,301,828 |
| Final10M round2 Recall@5 | 95.76% |
| Final10M round2 QPS | 6066.3750 |
| Final10M round2 mean / P95 / P99 ms | 5.2555 / 6.373 / 9.026 |
| 100K insertion timer total seconds | 22.0896 |
| Aggregate insertion vectors/s | 4527.0172 |
| Included barrier wait seconds | 0.5851 |
| Separate clone / load / save totals seconds | 425.0460 / 29.4984 / 34.2314 |

### Queries: second measured pass at every phase

Round2 is selected consistently, not the faster pass. All22 passes, including
P50/P90/P95/P99/min/max and all per-layer breakdowns, remain in the CSV/JSON.

| Batch | Vectors | Recall@5 | QPS | Mean ms | P50 ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 9900000 | 95.77% | 6287.6553 | 5.0683 | 5.006 | 7.285 | 10.791 |
| 1 | 9910000 | 95.77% | 6287.2163 | 5.0708 | 5.179 | 6.264 | 7.644 |
| 2 | 9920000 | 95.77% | 5639.8208 | 5.6543 | 5.727 | 6.985 | 10.426 |
| 3 | 9930000 | 95.77% | 6314.5386 | 5.0467 | 5.167 | 6.138 | 6.815 |
| 4 | 9940000 | 95.77% | 5531.7139 | 5.7637 | 5.781 | 7.306 | 11.738 |
| 5 | 9950000 | 95.77% | 6278.0786 | 5.0772 | 5.200 | 6.193 | 7.169 |
| 6 | 9960000 | 95.77% | 5605.8555 | 5.6884 | 5.811 | 6.924 | 8.612 |
| 7 | 9970000 | 95.77% | 5653.1445 | 5.6388 | 5.781 | 6.911 | 8.237 |
| 8 | 9980000 | 95.78% | 6155.2324 | 5.1820 | 5.170 | 6.688 | 10.647 |
| 9 | 9990000 | 95.76% | 6138.7505 | 5.1928 | 5.233 | 6.438 | 9.961 |
| 10 | 10000000 | 95.76% | 6066.3750 | 5.2555 | 5.332 | 6.373 | 9.026 |

### Final10M round2 posting-layer latency breakdown

| Posting layer | Read mean ms | Comp mean ms | Version mean ms | Layer total mean ms | Postings/query | TiKV batch checks/query |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.4533 | 1.7448 | 0.0000 | 3.2330 | 64.0000 | 0.0000 |

FILEIO uses local VersionMaps on every posting layer. Local `Deleted` checks run inline in the posting scan and are included in computation time. The layer0-only switch is TiKV-only. The version timing/count columns measure TiKV batch checks, so zero for FILEIO is expected, not a bypass.

### All ten insertion batches

| Batch | Inserted | Insert s | Vectors/s | Barrier s | Clone s | Load s | Save s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10000 | 2.7376 | 3652.7919 | 0.1813 | 56.1340 | 3.4442 | 4.0119 |
| 2 | 10000 | 2.5671 | 3895.4418 | 0.1007 | 40.1480 | 3.0877 | 3.2633 |
| 3 | 10000 | 2.1157 | 4726.4945 | 0.1007 | 42.2983 | 2.8713 | 3.3479 |
| 4 | 10000 | 2.3473 | 4260.2155 | 0.0807 | 39.5956 | 2.9817 | 3.3480 |
| 5 | 10000 | 1.9475 | 5134.7643 | 0.0000 | 41.6995 | 2.8469 | 3.3648 |
| 6 | 10000 | 2.2175 | 4509.5747 | 0.0406 | 40.1478 | 2.8146 | 3.2696 |
| 7 | 10000 | 2.2769 | 4391.9381 | 0.0405 | 43.9906 | 2.8204 | 3.5665 |
| 8 | 10000 | 2.0211 | 4947.7539 | 0.0406 | 41.2463 | 2.9509 | 3.2100 |
| 9 | 10000 | 1.9149 | 5222.2050 | 0.0000 | 39.7778 | 2.7345 | 3.4119 |
| 10 | 10000 | 1.9440 | 5144.0010 | 0.0000 | 40.0081 | 2.9462 | 3.4375 |

## 6. Acceptance and reproducibility limits

All22 measured rounds, ten ordered batches, exact final `[0,10000000)`
membership and100K inserted-metadata batch bijections passed.
[Independent live reload log](final-count-check.log) validates logical membership
and metadata, **not** every physical posting payload. Upper sparse VersionMap
counts need not equal10M. No new recall, latency or throughput release threshold
was imposed. No TiKV containers were constructed or started.

Published: source/tools, full configuration, raw aggregate JSON/log, metrics,
provenance and acceptance evidence. Not shipped: executables/shared libraries,
SIFT vectors, truth payload, index snapshots or TiKV databases.
The [reproduction guide](Reproduce.md) builds fixed remotely available source
and creates a new workload without the original host paths or local patch.
An isolated source build and six focused regressions were previously verified;
the full measured runs reuse the archived binary. A clean-host dependency install
and a full10M rerun using a newly rebuilt binary have **not** been verified.
Reproduction preserves controls and acceptance, not exact binary hashes,
random heads, recall or timings. The publication verifier checks retained
evidence offline; it cannot independently enumerate an unshipped database.

The four profiles each have one build, on a shared host at different times;
two query passes are not independent builds. Do not attribute a backend/layer
difference solely to one setting, or infer that pre-splitting accelerates query
performance: the unsplit TiKV2 attempt was aborted and is not an accepted control.
Known pre-existing DedupCandidateTest failures remain outside focused coverage.
