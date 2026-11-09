# SIFT10M TiKV: 1 posting layer(s), matched measured binary

[Evaluation index](../README.md) | [Reproduce](Reproduce.md) |
[All query rounds](query-rounds.csv) | [Insertions](insertions.csv) |
[Machine-readable results](results.json)

**Status: full workload and independent final10M membership/metadata verified.**
Recorded preparation UTC: `2026-11-09T01:26:40Z`.
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
| Runtime backend / posting layers | TIKVIO / 1, plus final in-memory BKT head |
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
| Version policy | Distributed version checks remain enabled on layer0; upper TiKV search skips batch checks when layer0-only=true. |

[Portable configuration](benchmark.ini) fixes this profile's storage/layers.
[Recorded input INI](recorded.ini) and [saved effective INI](saved-indexloader.ini)
retain exact measured paths/settings. [Raw JSON](benchmark-output.json) and
[raw log](run.log) are byte-preserved.

## 3. Deployment

Three PD and three TiKV containers share this host, with three voting replicas.
Client transport uses 16 independent channels per client/store and 8 completion queues
per client. These constants are compiled into the measured source, not INI tunables.

Before the workload: 260 exact split boundaries,
259 owned regions, 128 buckets per active key family.
Posting keys split on the low byte of little-endian DBKey (`0x02,0x04,...,0xfe`);
VersionMap keys split on equal VID ranges in `prefix_v:layer:VID-padded-to-10-digits`.
Layer 1 uses posting/version0 families; layer 2 adds version1. Outer isolation
boundaries account for the extra regions. Store IDs are recorded identities, not
fixed reproduction requirements.

| Key family | Leaders by store ID |
| --- | --- |
| posting | 1: 42, 4: 43, 5: 43 |
| version0 | 1: 43, 4: 42, 5: 43 |

Readiness requires every boundary, three voters, no pending/down owned peers,
no active PD operators, and per-family leader count difference <=26 over three
consecutive observations. Store `Up` alone is insufficient. The retained
[region snapshot](regions-before-workload.json) and [split/scatter evidence](presplit-validation.json)
are independently rechecked by the publication verifier.

The [server TOML](tikv.toml) fixes 16GB memory limit/store, 4GB block cache,
gRPC concurrency 8, apply/store pools 2, RocksDB background jobs 4, and
512MB/384MB maximum/split region sizes. PD ports: 33991-33993 and peers
34001-34003; TiKV ports: 30361-30363 and status 30381-30383.
Immutable server digests and commands are in [deployment](deployment.json)
and the [shared deployment code](../2026-11-05-sift10m-tikv-pool16-cq8/deployment.py).
This is trusted localhost host-network deployment without RPC authentication.
It is not a multi-host or production-security evaluation.

The TOML requests `reserve-space=1GB`, but each observed placeholder reserved
~71GiB (1% of this ~7TiB filesystem). Capacity/cache settings were not changed.
Only specifically approved reservation files were removed **after** prior
clusters stopped; no databases or accepted results were deleted. There is no
automatic cleanup or equivalent deletion step in the reproduction recipe.

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
| Initial build seconds | 1543.5735 |
| Benchmark process wall time | 49:11.06 |
| Benchmark maximum RSS KiB | 5,761,428 |
| Final10M round2 Recall@5 | 95.82% |
| Final10M round2 QPS | 1765.6836 |
| Final10M round2 mean / P95 / P99 ms | 18.0972 / 22.678 / 24.969 |
| 100K insertion timer total seconds | 352.7258 |
| Aggregate insertion vectors/s | 283.5063 |
| Included barrier wait seconds | 280.2482 |
| Separate clone / load / save totals seconds | 34.5614 / 11.9302 / 17.5109 |

### Queries: second measured pass at every phase

Round2 is selected consistently, not the faster pass. All22 passes, including
P50/P90/P95/P99/min/max and all per-layer breakdowns, remain in the CSV/JSON.

| Batch | Vectors | Recall@5 | QPS | Mean ms | P50 ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 9900000 | 95.84% | 1698.4691 | 18.8205 | 18.729 | 23.222 | 25.325 |
| 1 | 9910000 | 95.84% | 1815.8560 | 17.5965 | 17.491 | 22.013 | 24.201 |
| 2 | 9920000 | 95.83% | 1836.9639 | 17.3954 | 17.323 | 21.610 | 23.741 |
| 3 | 9930000 | 95.82% | 1814.4902 | 17.6142 | 17.582 | 21.841 | 23.550 |
| 4 | 9940000 | 95.82% | 1775.2036 | 17.9997 | 17.879 | 22.530 | 24.617 |
| 5 | 9950000 | 95.82% | 1786.2623 | 17.8894 | 17.686 | 22.735 | 25.937 |
| 6 | 9960000 | 95.82% | 1775.5878 | 17.9990 | 17.918 | 22.402 | 24.201 |
| 7 | 9970000 | 95.81% | 1785.8291 | 17.8952 | 17.772 | 22.385 | 24.833 |
| 8 | 9980000 | 95.82% | 1823.1919 | 17.5268 | 17.482 | 21.532 | 23.529 |
| 9 | 9990000 | 95.81% | 1781.3326 | 17.9383 | 17.817 | 22.440 | 24.743 |
| 10 | 10000000 | 95.82% | 1765.6836 | 18.0972 | 17.931 | 22.678 | 24.969 |

### Final10M round2 posting-layer latency breakdown

| Posting layer | Read mean ms | Comp mean ms | Version mean ms | Layer total mean ms | Postings/query | TiKV batch checks/query |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 7.8181 | 0.3402 | 7.6380 | 15.8375 | 64.0000 | 64.0000 |

Distributed version checks remain enabled on layer0; upper TiKV search skips batch checks when layer0-only=true.

### All ten insertion batches

| Batch | Inserted | Insert s | Vectors/s | Barrier s | Clone s | Load s | Save s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10000 | 140.2470 | 71.3028 | 131.9683 | 3.5648 | 1.1913 | 1.6674 |
| 2 | 10000 | 46.6563 | 214.3333 | 38.7334 | 3.3423 | 1.1707 | 1.7246 |
| 3 | 10000 | 36.8528 | 271.3501 | 29.8992 | 3.4781 | 1.1600 | 1.7859 |
| 4 | 10000 | 32.7044 | 305.7694 | 25.5956 | 3.4064 | 1.1906 | 1.5896 |
| 5 | 10000 | 19.8836 | 502.9259 | 12.5732 | 3.4126 | 1.1837 | 1.8685 |
| 6 | 10000 | 16.9727 | 589.1828 | 10.1984 | 3.4939 | 1.1966 | 1.8525 |
| 7 | 10000 | 15.2816 | 654.3829 | 8.2669 | 3.4567 | 1.2161 | 1.7358 |
| 8 | 10000 | 17.4582 | 572.7974 | 10.3187 | 3.4720 | 1.1957 | 1.8142 |
| 9 | 10000 | 14.5550 | 687.0505 | 7.5054 | 3.4957 | 1.2095 | 1.7456 |
| 10 | 10000 | 12.1142 | 825.4765 | 5.1891 | 3.4389 | 1.2160 | 1.7268 |

## 6. Acceptance and reproducibility limits

All22 measured rounds, ten ordered batches, exact final `[0,10000000)`
membership and100K inserted-metadata batch bijections passed.
[Independent live reload log](final-count-check.log) validates logical membership
and metadata, **not** every physical posting payload. Upper sparse VersionMap
counts need not equal10M. No new recall, latency or throughput release threshold
was imposed. TiKV's six owned containers stopped exit0 without OOM.

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
