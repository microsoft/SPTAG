# SIFT10M FILEIO: 2 posting layer(s), matched measured binary

[Evaluation index](../README.md) | [Reproduce](Reproduce.md) |
[All query rounds](query-rounds.csv) | [Insertions](insertions.csv) |
[Machine-readable results](results.json)

**Status: full workload and independent final10M membership/metadata verified.**
Recorded preparation UTC: `2026-11-09T02:38:47Z`.
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
| Runtime backend / posting layers | FILEIO / 2, plus final in-memory BKT head |
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
| Initial build seconds | 1351.6940 |
| Benchmark process wall time | 47:58.78 |
| Benchmark maximum RSS KiB | 6,718,792 |
| Final10M round2 Recall@5 | 95.05% |
| Final10M round2 QPS | 3990.2400 |
| Final10M round2 mean / P95 / P99 ms | 7.9918 / 10.175 / 11.338 |
| 100K insertion timer total seconds | 39.5238 |
| Aggregate insertion vectors/s | 2530.1211 |
| Included barrier wait seconds | 0.8277 |
| Separate clone / load / save totals seconds | 464.1705 / 29.2346 / 26.5173 |

### Queries: second measured pass at every phase

Round2 is selected consistently, not the faster pass. All22 passes, including
P50/P90/P95/P99/min/max and all per-layer breakdowns, remain in the CSV/JSON.

| Batch | Vectors | Recall@5 | QPS | Mean ms | P50 ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 9900000 | 95.05% | 3957.5403 | 8.0583 | 8.128 | 10.328 | 11.225 |
| 1 | 9910000 | 95.05% | 4065.6355 | 7.8454 | 7.867 | 10.003 | 13.890 |
| 2 | 9920000 | 95.06% | 3951.9006 | 8.0744 | 8.131 | 10.311 | 12.224 |
| 3 | 9930000 | 95.06% | 3991.6257 | 7.9941 | 8.096 | 10.144 | 10.987 |
| 4 | 9940000 | 95.05% | 4003.3547 | 7.9678 | 8.049 | 10.113 | 11.084 |
| 5 | 9950000 | 95.05% | 3983.9307 | 8.0063 | 8.090 | 10.112 | 11.417 |
| 6 | 9960000 | 95.05% | 4011.0078 | 7.9537 | 7.992 | 10.104 | 13.296 |
| 7 | 9970000 | 95.05% | 4001.9194 | 7.9708 | 8.098 | 10.108 | 10.953 |
| 8 | 9980000 | 95.05% | 3868.5959 | 8.2425 | 8.203 | 10.818 | 15.579 |
| 9 | 9990000 | 95.05% | 3916.1350 | 8.1469 | 8.288 | 10.169 | 11.040 |
| 10 | 10000000 | 95.05% | 3990.2400 | 7.9918 | 8.060 | 10.175 | 11.338 |

### Final10M round2 posting-layer latency breakdown

| Posting layer | Read mean ms | Comp mean ms | Version mean ms | Layer total mean ms | Postings/query | TiKV batch checks/query |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.6224 | 1.7229 | 0.0000 | 3.3802 | 64.0000 | 0.0000 |
| 1 | 1.5069 | 1.5340 | 0.0000 | 3.0758 | 64.0000 | 0.0000 |

FILEIO uses local VersionMaps on every posting layer. Local `Deleted` checks run inline in the posting scan and are included in computation time. The layer0-only switch is TiKV-only. The version timing/count columns measure TiKV batch checks, so zero for FILEIO is expected, not a bypass.

### All ten insertion batches

| Batch | Inserted | Insert s | Vectors/s | Barrier s | Clone s | Load s | Save s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10000 | 5.2763 | 1895.2643 | 0.2822 | 56.7096 | 3.1802 | 2.6211 |
| 2 | 10000 | 4.0917 | 2443.9750 | 0.0205 | 42.2372 | 2.7342 | 2.6975 |
| 3 | 10000 | 3.8242 | 2614.9363 | 0.0406 | 45.3148 | 2.6920 | 2.5459 |
| 4 | 10000 | 3.7189 | 2688.9550 | 0.0000 | 43.4657 | 2.7007 | 3.0102 |
| 5 | 10000 | 3.7611 | 2658.8263 | 0.0807 | 45.7561 | 2.9239 | 2.6720 |
| 6 | 10000 | 3.7879 | 2639.9761 | 0.1410 | 51.0026 | 3.8587 | 2.3484 |
| 7 | 10000 | 3.6817 | 2716.1564 | 0.0000 | 42.6809 | 2.8453 | 2.7225 |
| 8 | 10000 | 3.9544 | 2528.8178 | 0.1010 | 44.5036 | 2.8034 | 2.6074 |
| 9 | 10000 | 3.8057 | 2627.6314 | 0.0810 | 47.5140 | 2.7794 | 2.6282 |
| 10 | 10000 | 3.6219 | 2761.0001 | 0.0807 | 44.9860 | 2.7168 | 2.6641 |

## 6. Acceptance and reproducibility limits

All22 measured rounds, ten ordered batches, exact final `[0,10000000)`
membership and100K inserted-metadata batch bijections passed.
[Independent live reload log](final-count-check.log) validates logical membership
and metadata, **not** every physical posting payload. Upper sparse VersionMap
counts need not equal10M. No new recall, latency or throughput release threshold
was imposed. No TiKV containers were constructed or started.

The original runner incorrectly required positive TiKV-only counters for FILEIO
and marked validation failed even though the benchmark exited 0. The shared
validator was corrected after inspecting the pinned inline local check, then
all 22 rounds and the independent live final reload were verified.
[Recovery evidence](validation-recovery.json) and the original failed status
are retained. The workload and raw JSON/log were not rerun or modified.
Use the corrected shared runner, not the archived historical validator.

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
