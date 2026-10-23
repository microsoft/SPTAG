# Developer record: TiKV result-heap regression

[Benchmark report](../README.md) | [General methodology](../../../docs/Benchmarking.md) |
[Machine-readable evidence](results.json)

This is an engineering investigation and regression-validation record, not the
user-facing benchmark or its reproduction procedure. For fresh FILEIO/TiKV
runs, use the [benchmark instructions](../Reproduce.md).

## What was established

On 2026-10-23 UTC, replaying the same 200 queries against the **same saved
996K-vector TiKV index** increased Recall@5 from **77.6% to 97.6%** after
repairing a double heap-sort bug. All three rounds returned identical IDs and
distances within each version. No index rebuild or insertion occurred during
this comparison.

This is a correctness validation, **not a completed 1M update benchmark**:

| Experiment | Completed work | Recall@5 | Boundary |
| --- | --- | --- | --- |
| TiKV, before heap fix | 990K build; 6K inserts and queries through batch 6 | 77.6% at every measured phase | Stopped while collecting checkpoint statistics after batch 6; only five completion markers |
| FILEIO, same pre-fix executable | Separate 990K build; seven completed 1K batches | 97.9% initially; 97.8% in batches 1-7 | Stopped during batch 8 cloning; not ten complete batches |
| TiKV, fixed query-only replay | Same existing 996K index; 3 x 200 queries | **97.6% in all three rounds** | No post-fix insertion benchmark, new index build, or 10M run |

The FILEIO comparison used identical derived vectors, queries and phase truth,
but a separately built index. It located a backend-specific discrepancy; the
same-index replay, rather than a speculation about random seeds, isolated the
heap fix. The remaining 0.2 percentage-point difference from FILEIO is not a
controlled storage-only comparison.

## Workload and measurement

| Setting | Value |
| --- | --- |
| Dataset | First 1,000,000 vectors of SIFT1B `bigann_base.u8bin`; UInt8, 128 dimensions |
| Query set | First 200 vectors of `query.10K.u8bin` |
| Metric and IDs | Squared L2; zero-based IDs; 32-bit `SizeType` (`LARGEVID=OFF`) |
| Planned updates | 990,000 initial vectors + **10,000 total** inserts, 10 x 1,000; no deletes |
| Index | Two-layer dynamic SPANN/BKT; no quantizer |
| Threads | Search 16; insertion 4; build 8; no searches during insertion |
| Posting/search settings | `PostingPageLimit=12`, `SearchPostingPageLimit=12`, `BufferLength=8`, `SearchInternalResultNum=64`, `LatencyLimit=50` ms |
| Replay phase | Phase 6, active IDs `[0, 996000)`, 200 queries, TopK=5 |
| Repetitions | One partial update run per backend; three sequential query rounds per replay binary, not three independent index builds |
| Timing | Per-call `SearchIndex` wall time in ms; QPS = 200 / threaded round wall seconds, including thread startup/join |
| Percentiles | Sorted zero-based element `floor(200 * p)`; P95 index 190, P99 index 198; no interpolation |
| Cache policy | No OS/TiKV cache flush. Update-run recorded rounds followed primer queries. Replay ran before-fix then after-fix; first rounds are **not** controlled cold-cache measurements |
| Recall | 1,000 truth neighbors per round; both strict ID matching and harness-compatible distance-tolerant matching |

Exact truth was generated for all eleven active prefixes. Queries 0, 1 and 199
were independently checked with NumPy squared L2 at every phase. The truth
file has an int32 header `(4400, 5)`, followed by 11 x 200 x 5 int32 IDs and
the same number of float32 distances; total size 88,008 bytes. Phase `p` uses
offset `(p * 200 + query) * 5` within each array.

The harness also accepts a truth neighbor when a returned distance differs by
at most `1e-4 * (abs(truth_distance) + 1e-4)`. Strict ID-only and this
distance-tolerant recall were equal in every replay round.

## Controlled replay results

The saved index was `index_5`, written after batch 6, not an index inferred
from the last completion marker. Both binaries used that directory and the
same TiKV key prefix. Earlier snapshots share mutable postings and must not
be evaluated as if they still represented phase 0.

| Version | Round | Recall@5 | ID Recall@5 | Mean ms | P95 ms | P99 ms | QPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Before | 1 | 77.6% | 77.6% | 26.334 | 50.715 | 51.446 | 594.629 |
| Before | 2 | 77.6% | 77.6% | 21.638 | 25.358 | 28.357 | 721.106 |
| Before | 3 | 77.6% | 77.6% | 22.100 | 25.385 | 29.593 | 707.697 |
| Fixed | 1 | 97.6% | 97.6% | 24.022 | 42.068 | 44.313 | 651.509 |
| Fixed | 2 | 97.6% | 97.6% | 21.954 | 25.041 | 26.350 | 713.196 |
| Fixed | 3 | 97.6% | 97.6% | 22.120 | 25.463 | 27.104 | 707.115 |

Full-precision rows are in [replay-rounds.csv](replay-rounds.csv).
The 20.0 percentage-point improvement is 200 additional truth-neighbor hits
per 1,000. For **all 200 queries**, the old first four results became fixed
positions 2-5: the best retained candidate had previously been displaced.
For query 0, ID `504814` at squared distance `61125` was restored ahead of
ID `344333` at `62943`.

Do not infer a latency speedup from these ordered, cache-uncontrolled rounds.
Their main purpose is to verify output correctness on identical index data.

## Partial update results and unresolved issues

[partial-query-rounds.csv](partial-query-rounds.csv) preserves both recorded
rounds for each measured phase, including the anomalous batch 6. The initial
builds took 1,077.7032 s for TiKV and 277.4329 s for FILEIO. These are single
observations with different storage and version-map paths, not repeated
estimates of a backend speedup.

- TiKV batches 1-5 inserted at 323.7-360.2 vectors/s. Batch 6 fell to
  **2.4603 vectors/s** (406.454 s for 1,000 inserts). The cause is unresolved.
- Batch 6 saved its 996K-vector index successfully, then blocked in
  `LogCheckpointLayerStats -> TiKVVersionMap::GetContainedIDs ->
  TiKVIO::MultiGet(string keys) -> AsyncBatch::Wait`. This identifies a wait
  location, not a proven gRPC/server root cause.
- Batch 6 query means were 72.6298 and 76.1372 ms, unlike earlier phases.
  These rows were recovered from the log because they had not reached the
  incomplete output JSON. They are explicitly labeled in the CSV.
- Raw JSON files from both interrupted update runs remain incomplete and
  unchanged. Neither run is counted as a successful full benchmark.
- The new heap regression and recall-stride tests pass. The existing
  `SearchPathsMatchDeleteFirstAcrossVisibilityInterleavings` test still fails;
  its 45,901 failed assertions exactly match the preserved pre-fix executable.
  This result does not lift the documented dynamic-visibility correctness
  boundary or establish production readiness.

## Provenance and environment

| Field | Recorded evidence |
| --- | --- |
| Source baseline | `9e7fd9c401c9b2bb163fd6021503ade66aef16ef`, plus preserved working-tree patches |
| Build compatibility changes | Recorded later as `c3bcd8eac7a2689d604ea7db04737a77c159be74` |
| Functional changes | Recorded later as `192995914332cc2ecca9fbfde5742ee73166214c`; count fixes were present in both comparisons, heap fix only in the fixed replay |
| Toolchain | GCC 11.4; CMake 3.31.12; Boost 1.74; Protobuf 3.12.4; gRPC 1.30.2; Abseil 20210324; oneTBB |
| Build | Release; `TIKV=ON`, `TBB=ON`; `FOLLY`, `SPDK`, `ROCKSDB`, `GPU`, `RABITQ`, `LARGEVID` all `OFF` |
| Initialized submodule | zstd `e47e674cd09583ff0503f0f6defd6d23d8b718d3` (v1.5.2) |
| Host | Shared Linux host; AMD EPYC 7V12, 96 logical CPUs, AVX2, approximately 1.7 TiB RAM; SSD-backed workspace |
| Topology | **One host**, 3 PD + 3 TiKV containers, loopback networking, host-network mode, replication factor 3 |
| TiKV | v9.0.0-beta.2, commit `f5b820a91a6403ba11bba1bbcb3e3cca3e0d88db` |
| PD | Commit `f941a7ec5bf87222f937c88de1368dd122acaf47` |
| Resource settings | Per-TiKV 8 GB memory limit, 2 GB block cache, 1 GB gRPC pool; complete [tikv.toml](../tikv.toml) |
| Uncontrolled/unreported | Random seed not pinned; no controlled affinity/NUMA/cache protocol; no completed whole-run maximum-RSS measurement or isolated disk/network characterization |

Image digests, input hashes, compared binary/archive hashes, original config
hashes and raw-output hashes are in [results.json](results.json). The complete
billion-vector source file was not checksummed; the extracted 1M prefix was.
Generated protobuf/gRPC sources matched the local toolchain; their
machine-specific regeneration was not committed with the functional fix.

## Read-only replay

The [query probe](query_probe.cpp) is the diagnostic used for this workload,
not a general benchmark API. It fixes UInt8/dimension 128/200 queries/TopK=5/
16 threads and checks the index count, phase-truth header, operation status,
unique valid IDs and sorted results. It calls `LoadIndex` and `SearchIndex`,
not `AddIndex`, `SaveIndex` or the full update harness.

Build the TiKV-enabled library following the [integration guide](../../../docs/TiKV-Integration.md),
including local stub regeneration. From the repository root, the tested link
command, with the published probe path, is:

```bash
g++ -std=c++20 -O2 -fopenmp -DNUMA -DTBB -DTIKV \
  -IAnnService -IThirdParty/kvproto/generated -IThirdParty/zstd/lib \
  evaluation/2026-10-23-sift1m/development/query_probe.cpp \
  Release/libSPTAGLibStatic.a Release/libDistanceUtils.a Release/libzstd.a \
  -lboost_system -lboost_thread -lboost_serialization -lboost_wserialization \
  -lboost_regex -lboost_filesystem -ltbb -lnuma -lgrpc++ -lgrpc -lgpr \
  -lprotobuf -labsl_synchronization -lrt -lpthread \
  -o /absolute/path/to/run/query_probe
```

With PD/TiKV healthy and the matching saved index available:

```bash
/absolute/path/to/run/query_probe \
  /absolute/path/to/saved/index_5 \
  /absolute/path/to/query.10K.u8bin \
  /absolute/path/to/perftest_batchtruth.L2.UInt8_990000_128_200_5_1000_0_10 \
  6 3 /absolute/path/to/run/replay.json
```

For a before/after check, preserve the old static archive before rebuilding and
link two copies of the **same probe** against the respective archives. Both
must query the same saved index and truth phase, with no concurrent writers.
The old archive already contained the count fixes; simply checking out the
baseline commit would not reproduce that exact binary.

For a fresh update experiment, use the two-layer FILEIO example in the
[reproduction guide](../../../docs/Benchmarking.md), or change its backend to
`TIKVIO` with a fresh prefix and explicit PD addresses. These instructions
reproduce the protocol; without the retained index/postings they do not
guarantee the exact IDs or timings of this replay.

## Artifact availability and next validation

Only small, shareable summaries and the query probe are committed here.
Full local artifacts remain under the ignored paths:

```text
benchmark-output/sift1m-tikv-20261023-rerun/
benchmark-output/sift1m-fileio-20261023/
benchmark-output/tikv-heap-fix-20261023/
```

They include raw logs/JSON, original configurations and source patches, truth,
index snapshots, pre/post-fix binaries and static archives, per-query results,
cluster metadata and stop statuses. These are **local-only, not downloadable
from this Git checkout**; checksums identify them but do not publish them.
No public full-artifact bundle is claimed.

The replay's six experiment containers were stopped cleanly with exit code 0;
data was retained. The next step is to diagnose the wait/throughput anomaly,
then complete fresh post-fix update runs with independent repetitions. Only
after that should this report be extended to ten-batch, 10M, or multi-host
performance claims.
