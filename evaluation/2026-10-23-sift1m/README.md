# SIFT1M benchmark: FILEIO and TiKV

**Historical partial measurements.** For complete ten-batch runs with
`4/4/4`, use the [2026-10-28 report](../2026-10-28-sift1m/README.md) and its
[setup and run guide](../2026-10-28-sift1m/Reproduce.md).

[Run the benchmark](Reproduce.md) | [Configuration template](benchmark.ini) |
[Query measurements](query-rounds.csv) | [Environment and provenance](results.json)

Use this workload to evaluate recall, query latency and insertion behavior on
the first 1M SIFT1B vectors. The
[step-by-step instructions](Reproduce.md) cover building SPTAG, preparing data,
running FILEIO, deploying an isolated TiKV cluster, running TiKV and checking
the output. No saved index or executable from the original machine is required.
Start there directly; no SDK quick-start build is needed. The guide includes
public data downloads and a FILEIO-only path. It runs a fresh build/update
workload, not a replay of the saved-index TiKV queries reported below.

## Recorded results

Measurements were collected on 2026-10-23 UTC:

| Backend | Measured workload | Recall@5 | Completion status |
| --- | --- | --- | --- |
| FILEIO | 990K initial vectors; seven completed 1K insertion batches | 97.9% initially; 97.8% in batches 1-7 | Partial update run; stopped during batch 8 |
| TiKV | Three query rounds against a saved 996K-vector index | 97.6% in each round | Query results only; a full build/update evaluation is not available |

The indexes were built separately. They use the same input vectors, queries,
phase-specific truth and search parameters, but these are **not** matched
end-to-end backend performance runs. In particular, the TiKV query measurements
do not supply current build or insertion-throughput results.

### Queries at 996K active vectors

| Backend | Round | Recall@5 | Mean ms | P95 ms | P99 ms | QPS |
| --- | --- | --- | --- | --- | --- | --- |
| FILEIO | 1 | 97.8% | 6.122 | 8.544 | 10.270 | 2520.352 |
| FILEIO | 2 | 97.8% | 6.155 | 8.259 | 10.458 | 2489.885 |
| TiKV | 1 | 97.6% | 24.022 | 42.068 | 44.313 | 651.509 |
| TiKV | 2 | 97.6% | 21.954 | 25.041 | 26.350 | 713.196 |
| TiKV | 3 | 97.6% | 22.120 | 25.463 | 27.104 | 707.115 |

[query-rounds.csv](query-rounds.csv) contains all 16 recorded FILEIO rounds
and the three TiKV query rounds at full precision. Cache states and the
measurement sequences differ; do not turn this table into a storage-only
speedup claim. The recorded FILEIO build took 277.4329 s; no corresponding
current TiKV full-build measurement is reported.

## Workload

| Setting | Value |
| --- | --- |
| Dataset | First 1,000,000 vectors of SIFT1B `bigann_base.u8bin` |
| Representation | UInt8, 128 dimensions, squared L2, 32-bit vector IDs |
| Queries | First 200 rows of `query.10K.u8bin`, TopK=5 |
| Planned updates | 990,000 initial vectors + 10,000 total inserts, in ten batches of 1,000; no deletes |
| Index | Two-layer dynamic SPANN/BKT; no quantizer |
| Threads | Search 16; insertion 4; build 8; no concurrent search during insertion |
| Posting limits | `PostingPageLimit=12`, `SearchPostingPageLimit=12`, `BufferLength=8` |
| Search settings | `SearchInternalResultNum=64`, `LatencyLimit=50` ms |
| TiKV topology | One host, 3 PD + 3 TiKV containers, loopback networking, replication factor 3 |

Start with the [complete configuration template](benchmark.ini), then use
the renderer in the [current run instructions](../2026-10-28-sift1m/Reproduce.md#4-render-the-two-complete-configurations)
to set absolute paths and a new TiKV key prefix for your experiment.
This historical template retains the recorded `12/12/8` settings; the
current report's template uses `4/4/4`.

## Interpreting the output

- **Phase:** phase `p` searches IDs `[0, 990000 + p * 1000)`. Compare recall
  against truth for that active corpus, not the initial 990K corpus.
- **Recall:** exact truth is generated for all eleven phases. Each measured
  round has 1,000 truth neighbors. The harness accepts matching IDs or
  distances within `1e-4 * (abs(truth_distance) + 1e-4)`. ID-only recall was
  also recorded for the TiKV queries and equaled 97.6%; it was not separately
  recorded for FILEIO.
- **Latency:** per-query `SearchIndex` wall time, in milliseconds. P95/P99
  use sorted zero-based elements 190/198 for 200 queries, without interpolation.
  QPS is 200 divided by the threaded query-round wall time.
- **Repetitions:** FILEIO has two recorded rounds per phase, each following
  primer queries. TiKV has three sequential rounds on one saved index.
  Neither is a set of independent index builds. Caches were not flushed.
- **Completion:** a valid full run requires ten finished batches, a final
  saved count of 1,000,000 and complete JSON.
  [Step 8](../2026-10-28-sift1m/Reproduce.md#8-decide-whether-a-full-run-actually-completed)
  checks these conditions; process exit alone is insufficient.

The truth file has an int32 header `(4400, 5)`, followed by eleven phases of
200 x 5 int32 IDs and the same number of float32 distances, totaling 88,008
bytes. Queries 0, 1 and 199 were independently checked at every phase using
NumPy squared L2.

## Environment and reproducibility

The recorded environment used GCC 11.4, CMake 3.31.12, Boost 1.74,
Protobuf 3.12.4, gRPC 1.30.2, Abseil 20210324 and oneTBB, in a Release build.
The shared host had an AMD EPYC 7V12, 96 logical CPUs, AVX2, approximately
1.7 TiB RAM and SSD-backed storage. Affinity, NUMA placement, host contention
and cache state were not controlled.

The [provenance file](results.json) records source identities, image digests,
input hashes and the interpretation of each measurement.
[tikv.toml](tikv.toml) contains the TiKV resource and storage settings.
Fresh ANN builds use an unpinned random seed: reproduce the workload and
measure its recall, rather than requiring identical IDs or exact timings.

## Scope and current limits

- No complete ten-batch FILEIO/TiKV comparison, 10M result or multi-host result
  is established by these measurements.
- TiKV update testing encountered an unresolved checkpoint-statistics wait
  and insertion-throughput anomaly. Treat a timeout as an incomplete run,
  retain its output and investigate before retrying.
- See the [support matrix](../../docs/SupportMatrix.md#known-validation-limits)
  for dynamic-update correctness and compatibility limits.
- Full logs, source patches, generated inputs and index/cluster data are
  retained locally under the ignored `benchmark-output` directory, not
  distributed with this checkout. The public instructions generate a new run;
  they do not depend on those local artifacts.
