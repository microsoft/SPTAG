# SIFT1M benchmark: complete FILEIO and TiKV runs

[Set up and run](Reproduce.md) | [Configuration template](benchmark.ini) |
[Query measurements](query-rounds.csv) | [Insertion measurements](insertions.csv) |
[Environment and provenance](results.json)

Evaluate recall, query latency and incremental inserts on the first 1M SIFT1B
vectors. The [run guide](Reproduce.md) covers installation, public data
downloads, a local TiKV cluster, execution and completion checks. Start there
directly; no prior SDK build or retained index is required.

## Recorded results

Both runs completed a 990K-vector build and ten batches of 1K inserts.
Each exited with code 0, produced 22 measured query rounds, wrote all ten
checkpoint markers and saved a final count of 1,000,000 vectors.

| Backend | Date (UTC) | Build seconds | Initial Recall@5 | Final Recall@5 | Batches |
| --- | --- | --- | --- | --- | --- |
| FILEIO baseline | 2026-10-27 | 307.3029 | 98.5% | 98.6% | 10/10 |
| TiKV | 2026-10-28 | 1059.9998 | 98.2% | 98.2% | 10/10 |

FILEIO recall was 98.5% through batch 6 and 98.6% in batches 7-10.
TiKV recall was 98.2% in all 22 measured rounds. These are measurements from
one independent build per backend, not guaranteed accuracy thresholds.

The FILEIO baseline predates the TiKV client-lifecycle fix; the TiKV run
includes it. **These runs used different executable revisions**, although
inputs, truth and workload/search settings match. Binary hashes and source
identities are recorded in [results.json](results.json). Do not interpret
the tables as a controlled storage-only speedup comparison.

### Queries at the final 1M vectors

| Backend | Round | Recall@5 | Mean ms | P95 ms | P99 ms | QPS |
| --- | --- | --- | --- | --- | --- | --- |
| FILEIO | 1 | 98.6% | 10.6238 | 15.570 | 19.940 | 1458.7466 |
| FILEIO | 2 | 98.6% | 10.9590 | 15.652 | 19.097 | 1411.7415 |
| TiKV | 1 | 98.2% | 14.3542 | 16.462 | 18.359 | 1086.1537 |
| TiKV | 2 | 98.2% | 14.4073 | 16.999 | 18.130 | 1081.8413 |

[query-rounds.csv](query-rounds.csv) includes all 44 measured rounds at the
precision emitted by the harness. Each reported round follows primer queries;
caches were not flushed. Timing changes between phases can reflect cache and
shared-host effects, not just corpus growth.

### Insertion performance

| Backend | Minimum batch vectors/s | Maximum batch vectors/s | Median batch vectors/s | Aggregate vectors/s |
| --- | --- | --- | --- | --- |
| FILEIO | 145.0077 | 188.6462 | 164.3827 | 164.7843 |
| TiKV | 34.4033 | 285.2437 | 69.1302 | 66.0969 |

Aggregate throughput is 10,000 divided by the sum of the ten measured insertion
durations, not the arithmetic mean of batch throughputs. It excludes build,
reload, clone, query, save and checkpoint-statistics time. The complete TiKV
benchmark process took approximately 1,600 seconds.

TiKV completed without reproducing the batch-transition/checkpoint wait.
**Insertion throughput still varied substantially**; completion does not
establish that throughput variability is resolved.
[insertions.csv](insertions.csv) preserves all twenty per-batch measurements,
including load, clone and save times.

## Workload and setup

| Setting | Value |
| --- | --- |
| Dataset | First 1,000,000 SIFT1B vectors, UInt8, 128 dimensions, squared L2 |
| Queries | First 200 rows of the public 10K query file, TopK=5 |
| Updates | 990,000 initial vectors + 10,000 total inserts in ten batches; no deletes |
| Index | Two-layer dynamic SPANN/BKT, no quantizer, 32-bit vector IDs |
| Threads | Search 16, insertion 4, build 8; no concurrent search during insertion |
| Posting settings | `PostingPageLimit=4`, `SearchPostingPageLimit=4`, `BufferLength=4` |
| Search settings | `SearchInternalResultNum=64`, `LatencyLimit=50` ms |
| TiKV | Single host, 3 PD + 3 TiKV containers, loopback networking, replication factor 3 |

Both runs used byte-identical generated base vectors, insertion vectors,
queries and eleven-phase exact truth. Checksums are in [results.json](results.json).
Recall uses the harness's ID-or-distance matching rule,
`1e-4 * (abs(truth_distance) + 1e-4)`; strict ID-only recall was not separately
measured in these full runs. Phase `p` evaluates IDs
`[0, 990000 + p * 1000)`. P95/P99 use sorted zero-based positions 190/198 for
200 queries; QPS is 200 divided by the threaded round's wall time.

The host used Ubuntu 22.04, GCC 11.4, CMake 3.31.12, Boost 1.74,
Protobuf 3.12.4, gRPC 1.30.2, Abseil 20210324 and oneTBB. Hardware was an
AMD EPYC 7V12 with 96 logical CPUs, approximately 1.7 TiB RAM and SSD-backed
storage. Other workloads shared the host. Random seeds, affinity, NUMA
placement, contention and cache state were not controlled.

## Artifacts and reproduction

- [Reproduce.md](Reproduce.md): fresh setup and execution with one current
  TiKV-enabled executable for both backends.
- [benchmark.ini](benchmark.ini): portable `4/4/4` template; render paths and
  a unique key prefix before use.
- [tikv.toml](tikv.toml): measured resource/storage settings. Immutable PD/TiKV
  image digests are recorded in [results.json](results.json).
- [FILEIO output](fileio-output.json) and [TiKV output](tikv-output.json):
  complete raw harness JSON, copied byte-for-byte, with original local paths.
- [FILEIO recorded config](fileio-recorded.ini) and
  [TiKV recorded config](tikv-recorded.ini): exact original configs for
  provenance, **not portable run templates**.
- [artifacts.sha256](artifacts.sha256): checksums of the published data/configs.

TiKV was measured with the runtime change committed as
[`bdb2b64`](https://github.com/microsoft/SPTAG/commit/bdb2b64be9b6fe0854c8f268a94cf779f8a4e8dc).
The measurement checkout preceded that commit; source-base and dirty-patch
hashes are retained rather than relabeling the measured binary as a later build.
Full logs, patches, inputs and index/cluster data remain in local
`benchmark-output` directories; they are not required by the fresh-run guide.

This report includes no incomplete pre-fix TiKV update rows. The
[2026-10-23 report](../2026-10-23-sift1m/README.md) retains earlier partial
`12/12/8` observations separately. Neither report establishes 10M, multi-host,
fault-recovery or production-readiness claims; see the
[support matrix](../../docs/SupportMatrix.md#known-validation-limits).
