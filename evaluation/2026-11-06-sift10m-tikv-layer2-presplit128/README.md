# SIFT10M TiKV: 2 posting layer(s), matched measured binary

[Evaluation index](../README.md) | [Reproduce](Reproduce.md) |
[All query rounds](query-rounds.csv) | [Insertions](insertions.csv) |
[Machine-readable results](results.json)

**Status: full workload and independent final10M membership/metadata verified.**
Recorded preparation UTC: `2026-11-06T06:33:08Z`.
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
| Runtime backend / posting layers | TIKVIO / 2, plus final in-memory BKT head |
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

Before the workload: 391 exact split boundaries,
390 owned regions, 128 buckets per active key family.
Posting keys split on the low byte of little-endian DBKey (`0x02,0x04,...,0xfe`);
VersionMap keys split on equal VID ranges in `prefix_v:layer:VID-padded-to-10-digits`.
Layer 1 uses posting/version0 families; layer 2 adds version1. Outer isolation
boundaries account for the extra regions. Store IDs are recorded identities, not
fixed reproduction requirements.

| Key family | Leaders by store ID |
| --- | --- |
| posting | 1: 43, 4: 43, 5: 42 |
| version0 | 1: 43, 4: 43, 5: 42 |
| version1 | 1: 43, 4: 43, 5: 42 |

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
| Initial build seconds | 3210.7358 |
| Benchmark process wall time | 1:51:25 |
| Benchmark maximum RSS KiB | 5,773,976 |
| Final10M round2 Recall@5 | 95.22% |
| Final10M round2 QPS | 1191.6323 |
| Final10M round2 mean / P95 / P99 ms | 26.8215 / 31.682 / 33.830 |
| 100K insertion timer total seconds | 479.6448 |
| Aggregate insertion vectors/s | 208.4876 |
| Included barrier wait seconds | 297.9288 |
| Separate clone / load / save totals seconds | 21.9725 / 3.5728 / 8.8428 |

### Queries: second measured pass at every phase

Round2 is selected consistently, not the faster pass. All22 passes, including
P50/P90/P95/P99/min/max and all per-layer breakdowns, remain in the CSV/JSON.

| Batch | Vectors | Recall@5 | QPS | Mean ms | P50 ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 9900000 | 95.23% | 1214.6797 | 26.3176 | 26.207 | 31.798 | 34.121 |
| 1 | 9910000 | 95.22% | 1242.3876 | 25.7295 | 25.785 | 30.612 | 32.677 |
| 2 | 9920000 | 95.21% | 1210.0961 | 26.4151 | 26.568 | 31.241 | 33.547 |
| 3 | 9930000 | 95.22% | 1209.2654 | 26.4346 | 26.639 | 31.087 | 33.199 |
| 4 | 9940000 | 95.21% | 1216.1370 | 26.2828 | 26.391 | 31.092 | 33.214 |
| 5 | 9950000 | 95.21% | 1211.4591 | 26.3884 | 26.610 | 30.996 | 32.981 |
| 6 | 9960000 | 95.21% | 1195.4534 | 26.7276 | 27.055 | 31.268 | 32.978 |
| 7 | 9970000 | 95.21% | 1208.4666 | 26.4467 | 26.578 | 31.207 | 33.126 |
| 8 | 9980000 | 95.22% | 1199.6968 | 26.6393 | 26.961 | 31.255 | 33.373 |
| 9 | 9990000 | 95.22% | 1182.7255 | 27.0307 | 27.265 | 31.896 | 34.334 |
| 10 | 10000000 | 95.22% | 1191.6323 | 26.8215 | 26.978 | 31.682 | 33.830 |

### Final10M round2 posting-layer latency breakdown

| Posting layer | Read mean ms | Comp mean ms | Version mean ms | Layer total mean ms | Postings/query | TiKV batch checks/query |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 8.2000 | 0.3152 | 7.8074 | 16.3636 | 64.0000 | 64.0000 |
| 1 | 7.9421 | 0.2931 | 0.0000 | 8.2709 | 64.0000 | 0.0000 |

Distributed version checks remain enabled on layer0; upper TiKV search skips batch checks when layer0-only=true.

### All ten insertion batches

| Batch | Inserted | Insert s | Vectors/s | Barrier s | Clone s | Load s | Save s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10000 | 174.5436 | 57.2923 | 154.2189 | 2.1635 | 0.3437 | 0.9727 |
| 2 | 10000 | 65.2528 | 153.2502 | 46.3536 | 2.1971 | 0.3595 | 0.8987 |
| 3 | 10000 | 51.1460 | 195.5186 | 32.7980 | 2.2107 | 0.3577 | 0.8459 |
| 4 | 10000 | 39.1961 | 255.1276 | 20.1039 | 2.1443 | 0.3420 | 0.8578 |
| 5 | 10000 | 29.4366 | 339.7131 | 11.1870 | 2.2349 | 0.3998 | 0.8656 |
| 6 | 10000 | 28.8201 | 346.9802 | 10.7415 | 2.1900 | 0.3661 | 0.9051 |
| 7 | 10000 | 22.7829 | 438.9249 | 5.4288 | 2.2033 | 0.3472 | 0.8672 |
| 8 | 10000 | 26.9493 | 371.0670 | 9.4737 | 2.1030 | 0.3498 | 0.8857 |
| 9 | 10000 | 23.7482 | 421.0850 | 5.6727 | 2.2532 | 0.3478 | 0.8668 |
| 10 | 10000 | 17.7692 | 562.7718 | 1.9507 | 2.2725 | 0.3592 | 0.8773 |

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
