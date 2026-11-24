# Multi-tenant release validation

**Status: validation campaign completed (`059c99ab` SIFT1M full campaign + rebased onto `62deacc6` with SIFT10M 1,024-tenant RSS & lock scalability fixes).**

This report covers the multi-tenant SPANN feature rebased onto
`users/qiazh/pre-merge-tikv-bugfix` (`62deacc681cc3218fd3eaa377a07d35340850374`,
advancing from `059c99ab00b31e4e155f69dfc1c60d5006c3801e` after upstream PR `#476`).
It does not relabel earlier unverified or local-version-map TiKV results as
current measurements.

## Summary

- All **117 formal matrix seats / 39 three-repeat groups** completed the defined
  data, accounting, normal-teardown and graceful-reopen checks. They verified
  **1,323,585 returned distance rows** and **7,242 acknowledged VID memberships**.
  There were zero unexpected API errors, and **3,666 explicit capacity
  rejections** across 18 pressure/mixed attempts; these are not write successes.
- All **21 native/current timing processes** (`6` RocksDB, `6` FILEIO, and `9`
  TiKV across `059c99ab` native, `196d1e58` native, and `4305c58b` managed) and
  their separate accuracy scans completed. Every native/current pair produced
  **50,000 identical ordered Top-5 answer rows** (`97.802%` tie-aware recall).
- Single-tenant median QPS (`059c99ab` native / current) is **2,286.3 / 4,503.5**
  (`+97.0%`) for RocksDB, **1,952.9 / 1,888.6** (`-3.3%`) for FILEIO, and
  **741.7 / 1,276.6** (`+72.1%`, or `+208.3%` vs `196d1e58` native `414.1` QPS)
  for TiKV. Both native and managed TiKV use distributed per-VID `TiKVVersionMap`
  (`2,000,000` keys across two layers, zero local cache).
- Large TiKV archive restore/rearchive across live region splits was verified on
  1M-SIFT (`2,240,481` keys restored via `RawBatchPut` and re-archived with
  exact SHA256 match `894e82aba2d24df6391ca01bf327fc99a8581c53827032fb51b7a7ae41c9a510`
  in `54.1s`).
- Capacity-pressure whole-head churn, safe write rejection when half-budgets
  cannot reserve head growth, and pre-existing native compatibility suite
  failures remain documented boundaries.

## Source and build provenance

| Item | Revision / configuration |
|---|---|
| Target base | `62deacc681cc3218fd3eaa377a07d35340850374` (`users/qiazh/pre-merge-tikv-bugfix`; SIFT1M full matrix measured on `059c99ab00b31e4e155f69dfc1c60d5006c3801e`) |
| PR branch | `feature/single-node-multitenant-spann` (**microsoft/SPTAG#475**) |
| Measured runtime tree | Equivalent source tree (`4305c58b9140c354c9300219780bf71dd1e232f5` for TiKV; `c59000c4ef2aac76d8c3f5f0073f95437f05da71` with zero non-TiKV runtime diff for RocksDB/FILEIO) |
| Builds | Linux Release; `ROCKSDB=ON`, `TIKV=ON`, `TBB=ON`, `FOLLY=OFF`, `RABITQ=OFF`; feature ON and OFF |

Key performance and correctness fixes included in this release:

1. **Managed FILEIO page-complete merges**: Preserves full posting pages on append/merge and isolates manager dispatch overhead to `< 0.003 ms/query` (`-3.3%` end-to-end vs native `O_DIRECT` FILEIO due to shared-file vs compacted-file block layout).
2. **Dense local version-map fast path & head-rebuild reservation gating**: Avoids redundant membership hash lookups when loaded local version maps have zero tombstones, and skips head-rebuild capacity reservations below the reachable enqueue threshold.
3. **Distributed `TiKVVersionMap` parity & TiKV hot-path optimizations**: Managed TiKV tenants honor and archive/restore distributed per-VID `TiKVVersionMap` keys with no local cache, perform post-heap Top-K version filtering with lazy sorted tombstone refill (`~5` VIDs/query instead of `~1,200`), use `std::shared_mutex` + binary search over sorted region routes, reuse thread-local `CompletionQueue`s without per-query hash maps, and use `RawBatchPut`/`RawBatchDelete` with `region_error` routing refresh.
4. **1,024-tenant RSS, lazy thread-pool/lock initialization & clean reopen fast path**: `Helper::ThreadPool` and `SPDKThreadPool` spawn worker threads lazily on first enqueued job (`0` background threads for read-only tenants), `FineGrainedLock` (`4,096` mutexes) and `FineGrainedRWLock` (`4,096` `alignas(64)` buckets; `1.49x-1.71x` higher lock-lookup throughput across `16-64` threads vs `256` unaligned buckets) allocate backing lazily on first write while reserving full preflight growth in `ResidentMemory`, `ExtraWorkSpace` releases pooled scratch on destruction and skips unused `AsyncReadRequest` arrays on synchronous stores, and `TenantIndexManager::Initialize()` validates clean tenant snapshot manifests without eagerly loading and resaving every clean tenant on `Open()`.
   - **Single-tenant SIFT1M re-verification vs rebased native base `62deacc6`** (`16` threads, `10,000` queries, 3 fresh-process repeats/backend, `9/9` pairs producing `50,000` identical ordered Top-5 rows at `97.802%` tie-aware recall): RocksDB `2,334.4` → **`4,680.2` QPS** (`+100.5%`, p99 `12.88` → `4.86 ms`), FILEIO `1,932.5` → **`2,184.5` QPS** (`+13.0%`, p99 `12.96` → `13.48 ms`), and live-Docker TiKV with distributed `TiKVVersionMap` (`0` local cache) `833.5` → **`1,253.2` QPS** (`+50.4%`, p99 `27.99` → `18.52 ms`).
   - **SIFT10M 1,024-tenant verification** (`10,000,000` vectors across `1,024` skewed tenants on shared RocksDB): all `1,024` resident tenants occupy `914.0 MiB` tracked `ResidentMemory` (`5.64-5.70 GiB` total process RSS including `3 GiB` RocksDB block cache and `1.22 GiB` recall-verification vectors) and sustain `8,909-9,580` uniform QPS (`p99 = 4.80 ms`, `Recall@5 = 0.9993`) and `10,968-11,181` small-tenant QPS (`p99 = 1.89 ms`, `Recall@5 = 1.0000`).

## Completed targeted validation

| Validation | Result | Scope |
|---|---|---|
| Tenant/backend/core tests, local | 130 cases / 7,409 assertions passed | Real RocksDB/FILEIO, distance/quantizer/filter/KV suites |
| Backend tests, owned live TiKV | 35 cases / 6,322 assertions passed | Live TiKV CRUD, split/merge, checkpoint/restore, distributed version map |
| TiKV client tests | 21 cases / 236 assertions passed | Routing binary search, `RawBatchPut`/`RawBatchDelete` split retry, deadline/CQ draining, unsafe-mutation rejection |
| 1M-SIFT live-TiKV roundtrip | Exact SHA256 match (`54.1s`) | `2,240,481` posting + per-VID version keys restored across splits and re-archived byte-for-byte |
| Tenant CLI / Python / socket smoke | Passed | CRUD, isolation, concurrent RPC/GIL release, malformed/late replies and process-crash recovery |
| Native Python / legacy socket, feature ON & OFF | Passed | Build/save/reload/search; no default-tenant fallback; tenant RPC rejected when OFF |

## Native compatibility suites

| Variant | Suites attempted | Successful suites | Failed suite names |
|---|---:|---:|---|
| Pinned native base | 22 | 14 | AlgoTest, ConcurrentHashMapTest, DedupCandidateTest, IterativeScanTest, ReconstructIndexSimilarityTest, SPFreshTest, SSDServingTest, SpannWorkspaceTest |
| Feature ON | 22 | 15 | AlgoTest, ConcurrentHashMapTest, DedupCandidateTest, ReconstructIndexSimilarityTest, SPFreshTest, SSDServingTest, SpannWorkspaceTest |
| Feature OFF | 22 | 15 | Same suite names as feature ON |

These are **suite outcomes, not an all-green 160-case result**. Some native
processes abort before all cases finish. The overlap in failed suite names
does not prove every individual assertion failure is pre-existing.

The base process had not reached `SPANNParallelBuildTest` and
`SPFreshTest/IterativeSearch` in the full-suite attempts. Running each separately
on the unchanged base also failed (exit 1 and 201 respectively). This extends
baseline failure evidence without altering or suppressing the native tests.

The initial system-disk attempts and an invalid NVMe attempt that exhausted
temporary space are retained. The space-affected ON tail and previously
unexecuted OFF tail were run with explicit free-space gates and per-suite
archival. Only this task's generated artifacts were moved; historical frozen
fixtures and other users' data were not removed.

## Representative performance campaign

**Formal measurements are complete.** All 21 backend fixtures and all seven
allocation calibrations completed before timing. Source/binary hashes remained
unchanged; no feature builds or fixture generation ran during timing.

| Stratum | Tenant counts | Data sizes | Access | Workload | Head budget |
|---|---|---|---|---|---|
| A | 1 / 8 / 32 / 128 | Equal | Independent random | Search | 32 GiB |
| B | 8 / 32 / 128 | Equal | Cyclic intent trace | Search | 50% of measured all-resident allocation |
| C | 8 / 32 / 128 | Mixed, Zipf 1.15 sizes | 80% to approximately 20% hot tenants | Approximately 90% search / 10% insert | 32 GiB |
| D | 8 / 32 / 128 | Mixed, Zipf 1.15 sizes | Independent random | Approximately 90% search / 10% insert | 50% of all-resident allocation |

Coverage is three real backends and three fresh-process repeats: 117 matrix
attempts, plus 18 native/current single-tenant timing processes and separate
fresh-process accuracy scans.
This is stratified coverage, not a full Cartesian product.

Equal and mixed-size inputs use the same first one million SIFT vectors and
seeded permutation. The matrix reuses 200 distinct public queries over 2,048
measured operations after a primer. The native comparison uses all 10,000
public queries. These are different protocols and will not be merged into one
throughput table.

The native/current timing protocol uses 16 persistent workers, a five-second
warmup and a 15-second measured window. It retains per-request timing/status
rows. A separate fresh process scans all 10,000 queries, checks every returned
distance against the input vectors, and compares 50,000 ordered answer rows
between native and current. Those accuracy scans are not timed throughput
samples. Matrix attempts, unlike the native timing driver, retain answers for
every measured read directly.

Every successful matrix attempt must finish normal teardown, verify returned
VIDs/distances against its tenant vectors, report capacity rejections separately,
and verify acknowledged VID membership after graceful reopen. The latter is
not crash-durability proof; dedicated crash/recovery tests cover that separately.
Committed inserts per second are separated from attempted operations.

All timed working data belongs on the same NVMe device, without concurrent
feature builds or fixture generation. No host cache dropping or host tuning is
used. The machine is shared, so three-repeat ranges are necessary and do not
establish a production SLA.

### Setup findings and resolved TiKV restore / hot-path gates

- RocksDB, FILEIO, and TiKV harness checks passed exact-distance validation and
  graceful-reopen membership checks across all 117 formal seats (`0` failures).
- Earlier 1M-SIFT TiKV archive restore experienced slow single-key puts (`~840s`)
  and region-split routing stalls; `ManagedRawClient` now uses region-grouped
  `RawBatchPut` and `RawBatchDelete` with `region_error` routing refresh and
  bounded retry, completing the 1M-SIFT (`2,240,481` keys) live-split restore +
  checkpoint roundtrip in `54.1s` with an exact SHA256 match.
- Every formal TiKV run (`3/3` single-tenant pairs and `39/39` multi-tenant
  matrix runs) explicitly verified from runtime logs that distributed per-VID
  `TiKVVersionMap` (`no local cache`, `1,000,000` VIDs per layer in single-tenant)
  was active and no local `VersionMap` fallback occurred.

## Historical evidence, not current-release measurements

The earlier formal cache matrix (`143/144`) and earlier `196d1e58` campaign
(where managed TiKV accidentally initialized local `VersionMap` instead of
distributed `TiKVVersionMap`) are retained strictly as historical records and are
not mixed with the current `059c99ab` campaign below.

Native/private versus shared postings have identical logical values, and on TiKV
both native and managed variants use distributed per-VID `TiKVVersionMap`.
However, private vs shared RocksDB SST/blob/merge-operand layouts, compacted vs
append-shared FILEIO posting layouts, and little-endian (`ExtraTiKVController`)
vs big-endian (`ManagedTiKVBackend`) TiKV key encodings still differ physically,
so native/current throughput deltas are not isolated head-cache-manager overhead.

Read-only inspection of newly owned copies of the exact native-comparison
inputs reconfirmed:

| Posting representation | SST files | Blob files | Merge operands |
|---|---:|---:|---:|
| Native private RocksDB | 3 | 6 | 0 |
| Shared RocksDB postings CF | 65 | 0 | 240,479 |

Both have 240,479 logical postings. The native blob files occupy 1,213,325,598
bytes; shared merge operands contain 1,204,667,982 raw value bytes. Blob-file
bytes and logical payload bytes are different quantities. No compaction was
forced for these measurements.

## Interpretation of the completed RocksDB, FILEIO, and TiKV groups

The three-repeat groups across all three backends confirm that capacity pressure
remains a whole-head LRU boundary, not an optimized path. In the 32/128-tenant
cyclic cases, all 2,048 measured operations missed, loaded and evicted a head:
zero resident hits. On TiKV, each whole-head load additionally initializes
distributed `TiKVVersionMap` metadata from PD/TiKV (`36.0 / 23.8 / 24.0` QPS vs
`~140 / 59 / 118` QPS on local RocksDB/FILEIO). Bucket/DataBlockSize changes
were deliberately not made.

The 32-GiB groups had zero measured evictions. Random/resident versus
cyclic/pressure changes both budget and trace, so the throughput difference
must not be presented as an isolated causal measurement of the byte budget.
Likewise, hot/resident versus uniform/pressure mixed workloads change more than
one factor.

At half-allocation budgets, mixed 8/32-tenant workloads accepted no inserts on
any backend (`9.9%` capacity-rejected operations, `0` errors): attempted writes
were safely rejected before VID/log allocation. The 128-tenant half-budget is
larger and admits writes (`~27.8–27.9` ACK/s on RocksDB/FILEIO, `4.2` ACK/s on
TiKV under heavy head-eviction + distributed version-map reload churn).

Process RSS is also not a head-cache charge. Backend caches, query scratch and
other excluded memory matter: equal logical RocksDB postings have very different
SST/blob/merge layouts, which also affects cache residency. RSS differences here
are not attributed solely to the head manager.

A separate post-campaign RocksDB diagnostic using the same frozen runtime
measured **133,113,202 head bytes**, **1,267,346,656 backend block-cache bytes**
and **16,384 memtable bytes** after reads, with a 1,678,972-KiB process high-water
mark. This directly confirms that backend cache memory is substantial outside
the approximately 127-MiB head ledger; it is not a complete attribution of every
RSS byte. Diagnostic timing is excluded from all formal performance summaries.

## Results and reproduction

The full per-group tables follow below. Raw operation/answer CSVs, per-run JSON
summaries, logs, and fixture inputs are kept outside the source repository so
large benchmark artifacts are not checked into git.

## Reproduction outline

Use separate **source worktrees**, not just separate CMake build directories,
for native / feature-ON / feature-OFF variants. Non-Windows CMake outputs go
into the source tree's `Release` directory.

```sh
cmake -S . -B build-mt -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMULTITENANT=ON -DROCKSDB=ON -DTIKV=ON \
  -DTBB=ON -DFOLLY=OFF -DRABITQ=OFF
cmake --build build-mt --target SPTAGTenantTest SPTAGTiKVClientTest \
  tenanttool tenantperf postingbackendperf server _SPTAG _SPTAGClient --parallel 2
Release/SPTAGTenantTest --run_test=MultiTenant,PostingBackends
Release/SPTAGTiKVClientTest --run_test=TiKVPostingClient
python3 Test/tenant_smoke.py --bin-dir Release
```

Real TiKV integration additionally requires an explicitly owned endpoint and
namespace:

```sh
SPTAG_TIKV_PD=127.0.0.1:OWNED_PORT SPTAG_TIKV_PREFIX=owned-unique-prefix/ \
  Release/SPTAGTenantTest --run_test=MultiTenant,PostingBackends
```

Create inputs only in fresh owned directories. A manifest line is
`TENANT_NAME UINT8_128D_DEFAULT_VECTOR_FILE COUNT`. Query/vector DEFAULT files
start with two little-endian 32-bit integers (row count, dimension), followed
by their UInt8 payload.

```sh
TENANT_PERF_CONFIG=Test/tenant_sift1m.ini \
  Release/tenantperf build-many OWNED_NEW_ROCKS_ROOT MANIFEST
Release/postingbackendperf convert OWNED_ROCKS_ROOT OWNED_NEW_IMAGE fileio - -
# Clone the immutable image into a fresh OWNED_WORK_ROOT before each attempt.
Release/postingbackendperf prepare OWNED_WORK_ROOT fileio - -
Release/postingbackendperf matrix OWNED_WORK_ROOT QUERIES OWNED_NEW_OUTPUT \
  fileio - - random mixed 34359738368 2048 20261118 16
```

`prepare` mutates the owned clone's namespace identity and backend
configuration; never point it at a frozen fixture. Use a newly calculated
positive byte budget for pressure runs, not a resident-count proxy.

## Product and PR boundaries

The cache limits tracked index allocation capacity plus reservations, not RSS.
Query scratch, backend caches and coordinator/staging memory are outside that
ledger. Positive byte admission, pre-VID/pre-log safety, read-epoch reclamation,
and fail-closed uncertain mutation semantics remain enabled.

This is trusted-caller, single-node functionality. There is no tenant
authentication, cross-node failover, Windows-backend certification, or
machine-power-loss durability claim. Process-restart tests do not imply
fsync-per-ACK behavior.

This feature targets `users/qiazh/pre-merge-tikv-bugfix`
(`059c99ab00b31e4e155f69dfc1c60d5006c3801e`) in **microsoft/SPTAG#475**.

<!-- generated-results:start -->
## Current-release performance results

Values are medians [minimum, maximum] over accepted fresh processes.
Rows with fewer than three completed repeats are incomplete, not headline comparisons.
Capacity rejections are not committed writes or successful reads.

| Backend | Case / trace / workload | Budget | Completed repeats | Successful read QPS | ACK inserts/s | Capacity rejected % | Successful-read p99 ms | Process RSS MiB |
|---|---|---|---:|---|---|---|---|---|
| rocksdb | 1m-1-equal / random / search | 32 GiB | 3/3 | 4,830.1 [4,631.0, 4,928.4] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 5.8 [5.8, 5.9] | 573.7 [572.7, 574.0] |
| rocksdb | 1m-8-equal / random / search | 32 GiB | 3/3 | 4,633.7 [4,528.9, 4,801.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 11.4 [10.7, 11.8] | 1,259.2 [1,254.9, 1,259.7] |
| rocksdb | 1m-32-equal / random / search | 32 GiB | 3/3 | 3,919.5 [3,900.9, 3,928.9] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 14.5 [14.1, 14.5] | 2,004.3 [2,001.2, 2,005.0] |
| rocksdb | 1m-128-equal / random / search | 32 GiB | 3/3 | 4,264.6 [4,095.7, 4,353.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 12.6 [11.6, 13.7] | 3,905.2 [3,905.2, 3,909.8] |
| rocksdb | 1m-8-equal / cyclic / search | 50% allocation | 3/3 | 140.1 [138.3, 141.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 337.5 [330.5, 342.4] | 2,044.6 [2,024.2, 2,077.4] |
| rocksdb | 1m-32-equal / cyclic / search | 50% allocation | 3/3 | 57.1 [55.7, 58.3] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 321.4 [296.7, 349.2] | 1,959.2 [1,957.2, 1,966.0] |
| rocksdb | 1m-128-equal / cyclic / search | 50% allocation | 3/3 | 115.3 [113.5, 116.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 194.8 [194.1, 195.2] | 4,957.5 [4,909.1, 4,959.2] |
| rocksdb | 1m-8-mixed / hot / mixed | 32 GiB | 3/3 | 4,173.5 [3,823.9, 4,429.0] | 455.5 [432.3, 456.7] | 0.0 [0.0, 0.0] | 6.9 [6.7, 7.0] | 995.0 [974.6, 997.3] |
| rocksdb | 1m-32-mixed / hot / mixed | 32 GiB | 3/3 | 4,517.9 [4,049.1, 4,565.3] | 494.4 [416.5, 516.1] | 0.0 [0.0, 0.0] | 14.3 [12.8, 14.3] | 1,757.3 [1,748.6, 1,790.4] |
| rocksdb | 1m-128-mixed / hot / mixed | 32 GiB | 3/3 | 5,241.4 [5,126.7, 5,281.9] | 578.0 [539.1, 579.5] | 0.0 [0.0, 0.0] | 12.8 [11.7, 12.9] | 3,625.9 [3,620.1, 3,626.2] |
| rocksdb | 1m-8-mixed / random / mixed | 50% allocation | 3/3 | 193.9 [193.8, 194.8] | 0.0 [0.0, 0.0] | 9.9 [9.7, 10.2] | 270.3 [261.1, 277.1] | 2,293.4 [2,118.3, 2,354.9] |
| rocksdb | 1m-32-mixed / random / mixed | 50% allocation | 3/3 | 220.3 [220.1, 247.1] | 0.0 [0.0, 0.0] | 9.9 [9.7, 10.2] | 255.3 [253.7, 257.8] | 2,383.3 [2,266.5, 2,405.5] |
| rocksdb | 1m-128-mixed / random / mixed | 50% allocation | 3/3 | 256.3 [240.3, 258.3] | 27.8 [26.4, 29.1] | 0.0 [0.0, 0.0] | 158.2 [135.9, 159.8] | 3,635.9 [3,259.3, 3,872.8] |
| fileio | 1m-1-equal / random / search | 32 GiB | 3/3 | 2,215.6 [2,139.8, 2,216.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 8.9 [8.7, 9.1] | 360.0 [358.6, 360.4] |
| fileio | 1m-8-equal / random / search | 32 GiB | 3/3 | 2,274.1 [2,272.3, 2,288.2] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 8.3 [8.3, 8.4] | 523.1 [522.6, 523.1] |
| fileio | 1m-32-equal / random / search | 32 GiB | 3/3 | 2,257.5 [2,257.3, 2,259.2] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 8.7 [8.6, 9.2] | 1,124.3 [1,124.1, 1,124.3] |
| fileio | 1m-128-equal / random / search | 32 GiB | 3/3 | 2,262.3 [2,229.2, 2,292.3] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 10.0 [9.7, 10.9] | 3,034.9 [3,034.4, 3,035.3] |
| fileio | 1m-8-equal / cyclic / search | 50% allocation | 3/3 | 140.6 [134.1, 151.3] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 343.2 [327.6, 385.6] | 1,268.9 [1,059.7, 1,381.2] |
| fileio | 1m-32-equal / cyclic / search | 50% allocation | 3/3 | 59.3 [58.6, 59.8] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 297.5 [296.1, 335.0] | 1,087.5 [1,081.0, 1,091.7] |
| fileio | 1m-128-equal / cyclic / search | 50% allocation | 3/3 | 117.7 [117.4, 119.5] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 195.4 [194.0, 195.8] | 4,005.9 [3,888.2, 4,030.2] |
| fileio | 1m-8-mixed / hot / mixed | 32 GiB | 3/3 | 1,716.4 [1,696.3, 1,828.2] | 188.0 [185.6, 194.0] | 0.0 [0.0, 0.0] | 8.3 [7.6, 8.5] | 528.2 [519.1, 537.7] |
| fileio | 1m-32-mixed / hot / mixed | 32 GiB | 3/3 | 1,921.4 [1,796.9, 1,964.7] | 215.0 [184.8, 217.2] | 0.0 [0.0, 0.0] | 9.5 [9.1, 9.7] | 1,105.8 [1,105.8, 1,131.2] |
| fileio | 1m-128-mixed / hot / mixed | 32 GiB | 3/3 | 1,989.1 [1,982.9, 1,994.5] | 217.0 [205.1, 224.9] | 0.0 [0.0, 0.0] | 9.7 [9.5, 10.4] | 2,990.4 [2,987.7, 2,991.4] |
| fileio | 1m-8-mixed / random / mixed | 50% allocation | 3/3 | 187.2 [180.4, 188.9] | 0.0 [0.0, 0.0] | 9.9 [9.7, 10.2] | 284.9 [275.8, 287.0] | 1,482.0 [1,457.7, 1,500.4] |
| fileio | 1m-32-mixed / random / mixed | 50% allocation | 3/3 | 224.1 [222.0, 248.1] | 0.0 [0.0, 0.0] | 9.9 [9.7, 10.2] | 243.2 [235.5, 249.3] | 1,739.1 [1,729.3, 1,781.1] |
| fileio | 1m-128-mixed / random / mixed | 50% allocation | 3/3 | 258.9 [244.3, 264.5] | 27.9 [26.9, 30.1] | 0.0 [0.0, 0.0] | 152.7 [135.2, 167.4] | 3,021.4 [3,007.7, 3,161.4] |
| tikv | 1m-1-equal / random / search | 32 GiB | 3/3 | 1,538.8 [1,489.9, 1,560.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 14.9 [13.3, 15.8] | 314.6 [314.3, 315.0] |
| tikv | 1m-8-equal / random / search | 32 GiB | 3/3 | 2,131.2 [2,107.7, 2,199.8] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 13.1 [11.4, 13.3] | 465.3 [461.8, 469.1] |
| tikv | 1m-32-equal / random / search | 32 GiB | 3/3 | 2,280.1 [2,229.5, 2,295.8] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 12.7 [12.3, 12.8] | 1,095.8 [1,088.0, 1,098.8] |
| tikv | 1m-128-equal / random / search | 32 GiB | 3/3 | 2,529.1 [2,377.4, 2,549.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 11.3 [10.6, 11.6] | 2,991.2 [2,986.7, 2,998.0] |
| tikv | 1m-8-equal / cyclic / search | 50% allocation | 3/3 | 36.0 [36.0, 36.3] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1,678.9 [1,671.1, 1,680.7] | 1,080.5 [1,069.3, 1,119.7] |
| tikv | 1m-32-equal / cyclic / search | 50% allocation | 3/3 | 23.8 [23.7, 24.2] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 2,056.7 [2,022.5, 2,060.1] | 1,597.2 [1,571.9, 1,608.7] |
| tikv | 1m-128-equal / cyclic / search | 50% allocation | 3/3 | 24.0 [23.9, 24.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 750.4 [746.8, 755.8] | 4,352.4 [4,289.3, 4,410.2] |
| tikv | 1m-8-mixed / hot / mixed | 32 GiB | 3/3 | 551.2 [492.9, 935.1] | 60.3 [55.7, 96.2] | 0.0 [0.0, 0.0] | 9.7 [9.1, 9.7] | 488.9 [469.2, 489.9] |
| tikv | 1m-32-mixed / hot / mixed | 32 GiB | 3/3 | 1,690.0 [590.8, 1,707.8] | 186.9 [60.8, 191.0] | 0.0 [0.0, 0.0] | 14.3 [11.9, 17.1] | 1,075.9 [1,066.7, 1,082.3] |
| tikv | 1m-128-mixed / hot / mixed | 32 GiB | 3/3 | 1,953.2 [1,943.2, 1,981.0] | 216.8 [200.9, 219.7] | 0.0 [0.0, 0.0] | 11.0 [10.5, 11.1] | 2,951.0 [2,945.5, 2,960.9] |
| tikv | 1m-8-mixed / random / mixed | 50% allocation | 3/3 | 43.5 [43.1, 45.2] | 0.0 [0.0, 0.0] | 9.9 [9.7, 10.2] | 2,330.4 [2,320.5, 2,622.8] | 1,063.1 [1,029.1, 1,079.5] |
| tikv | 1m-32-mixed / random / mixed | 50% allocation | 3/3 | 37.6 [37.2, 38.0] | 0.0 [0.0, 0.0] | 9.9 [9.7, 10.2] | 2,428.6 [2,296.4, 2,429.3] | 1,679.3 [1,643.3, 1,725.0] |
| tikv | 1m-128-mixed / random / mixed | 50% allocation | 3/3 | 39.3 [37.6, 41.3] | 4.2 [4.1, 4.7] | 0.0 [0.0, 0.0] | 1,747.0 [1,683.7, 1,771.0] | 3,220.9 [3,107.1, 3,365.3] |

### Native/current single-tenant comparison

These inputs have matched logical postings (and distributed per-VID `TiKVVersionMap` on both native and managed TiKV).
RocksDB and FILEIO physical layouts differ between native and managed backends; deltas are not isolated manager overhead.

| Backend | Complete paired repeats | Native QPS | Current QPS | Current vs native | Native/current p99 ms | Native/current RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| rocksdb | 3/3 | 2,286.3 | 4,503.5 | +97.0% | 12.7 / 5.0 | 267.6 / 1536.4 |
| fileio | 3/3 | 1,952.9 | 1,888.6 | -3.3% | 13.1 / 20.1 | 274.2 / 356.8 |
| tikv | 3/3 | 741.7 | 1,276.6 | +72.1% | 28.9 / 18.2 | 224.0 / 325.4 |

For TiKV, `Native QPS` above reflects the target base `059c99ab` (`741.7` QPS, p99 `28.9` ms, which includes upstream transport fix `95bcb20f`); the earlier `196d1e58` native baseline measured `414.1` QPS (p99 `49.9` ms) on the identical three clusters (`+208.3%` managed vs `196d1e58`).

### Coverage and retained failures

Matrix: 117 completed, 0 failed, 0 blocked, 0 missing out of 117 planned attempts.
Complete three-repeat groups: 39/39.

Completed matrix attempts verified 1,323,585 returned distance rows and 7,242 acknowledged VID memberships after graceful reopen, with 3,666 explicit capacity rejections under half-budget pressure and 0 unexpected API errors.

<!-- generated-results:end -->
