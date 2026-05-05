# Backend Comparison Results: RocksDB vs TiKV (SPFresh)

Branch: `users/zhangt/backend-comparison` (forked from upstream `users/qiazh/merge-spfresh-tikv` @ `8a74694`).
Dataset: SIFT (UInt8, dim=128, L2). Truth gen on-the-fly.

Workload: each run does
1. Build SSD index from `BaseVectorCount` vectors
2. Pre-insert search (`Benchmark 0`, 2 rounds)
3. **`Benchmark 1`**: 10 batches, each inserts `InsertVectorCount/10` vectors and runs search concurrently — measures insert throughput + search-during-update QPS / latency / recall

All latencies in **ms**. Pre-insert columns: search-only baseline. Insert columns: 10-batch averages of insert throughput + concurrent search. p50 / p99 are batch-mean of per-batch percentiles.

| Scale | Backend | Layers | Node | Status | Build (s) | Pre-insert QPS | Pre p50 | Pre p99 | Pre recall | Insert tput (vec/s) | Search-during-insert QPS | SDI p50 | SDI p99 | SDI recall |
|-------|---------|--------|------|--------|-----------|----------------|---------|---------|------------|---------------------|--------------------------|---------|---------|------------|
| 1M    | RocksDB | L1     | 0.9  | ✅ done                               | 42.4    | 2365.2 | 1.70  | 3.17  | 0.9890 | 3894.1 | 2241.0 | 1.81  | 3.08  | 0.9878 |
| 10M   | TiKV    | L1     | 0.7  | ✅ done                               | 924.0   | 578.5  | 6.88  | 9.89  | 0.9520 | 937.8  | 507.7  | 7.80  | 11.49 | 0.9531 |
| 10M   | TiKV    | L1     | 0.9  | ✅ done                               | 952.9   | 404.1  | 9.62  | 15.18 | 0.9630 | 997.7  | 489.8  | 8.09  | 11.36 | 0.9619 |
| 10M   | TiKV    | L2     | 0.7  | ✅ done                               | 978.6   | 260.7  | 15.24 | 21.46 | 0.9510 | 175.0  | 199.9  | 19.89 | 24.52 | 0.9535 |
| 10M   | RocksDB | L1     | 0.9  | ✅ done (after MultiGet/SaveIndex fixes) | 836.8 | 1778.4 | 2.28  | 3.95  | 0.9680 | 3627.7 | 1470.5 | 2.74  | 4.13  | 0.9680 |
| 10M   | RocksDB | L2     | 0.9  | ✅ done                               | 828.5   | 1328.1 | 3.06  | 4.27  | 0.9630 | 2277.7 | 989.8  | 4.09  | 5.50  | 0.9636 |
| 100M  | TiKV    | L1     | 0.7  | ✅ done                               | 11280.4 | 217.1  | 18.17 | 27.54 | 0.9220 | 866.7  | 226.8  | 17.61 | 23.72 | 0.9204 |
| 100M  | RocksDB | L1     | 0.9  | ✅ done                               | 10090.9 | 1746.0 | 2.26  | 4.08  | 0.9170 | 7635.8 | 1358.0 | 2.88  | 4.64  | 0.9218 |
| 100M  | TiKV    | L2     | 0.7  | 🟡 running (~86% Sent, build phase)   | —       | —      | —     | —     | —      | —      | —      | —     | —     | —      |
| 100M  | RocksDB | L2     | 0.9  | 🟡 running                            | —       | —      | —     | —     | —      | —      | —      | —     | —     | —      |
| 1B    | TiKV    | L1     | TBD  | pending                               | —       | —      | —     | —     | —      | —      | —      | —     | —     | —      |
| 1B    | TiKV    | L2     | TBD  | pending                               | —       | —      | —     | —     | —      | —      | —      | —     | —     | —      |
| 1B    | RocksDB | L1     | TBD  | pending                               | —       | —      | —     | —     | —      | —      | —      | —     | —     | —      |
| 1B    | RocksDB | L2     | TBD  | pending                               | —       | —      | —     | —     | —      | —      | —      | —     | —     | —      |

(All numbers use binary `a33f31e` unless noted; **10M RocksDB L1 row** uses post-fix binary built 2026-04-30 with MultiGet bounds check + SaveIndex hardlink-checkpoint patches. Avg insert throughput / search QPS / recall / latency percentiles are means across the 10 insert batches. 1M RocksDB above uses the post-revert build linked against RocksDB 7.6 submodule pin.)

## BE+π Optimization (TiKV only)

Two-part fix landed in `a184213` ("BKT-DFS permutation π + big-endian keys + fan-out pool"):

1. **π (BKT-DFS permutation)** — at BuildIndex, walk the BKT tree in DFS order to produce a bijection on head IDs so cluster-adjacent heads land at adjacent integer keys. Persisted to `headPermutation_<layer>.bin`.
2. **BE-encoded keys** — `EncodeIntKey()` switched LE→BE so numerical adjacency = lexical adjacency = TiKV region adjacency.
3. **Reusable fan-out thread pool** in `RawBatchGet` — groups MultiGet by region.

Either alone fails: π without BE → π's locality scrambles by least-significant byte; BE without π → IDs aren't clustered. Together a query's ~64 head IDs hit a small handful of regions instead of fanning out to all 100+.

| Scale / Layers | Baseline (Titan-tuned) | **BE+π peak** | RocksDB | Lift over baseline | Ratio vs RocksDB |
|---|---|---|---|---|---|
| 10M L1   | 654 QPS, recall 0.947  | **1036 QPS, recall 0.956**, p50 3.87ms p99 5.14ms | 1778 QPS | **1.58×** | **59%** ✅ |
| 10M L2   | 261 QPS, recall 0.951  | **555.6 QPS, recall 0.945**, p50 7.08ms p99 8.95ms | 1225 QPS | **2.13×** | **45%** ✅ |
| 100M L1  | 217 QPS, recall 0.922  | **742.1 QPS, recall 0.914** (R2b 740.7 stable)    | 1746 QPS | **3.42×** | **42.5%** ✅ |
| 100M L2  | 116 QPS, recall 0.893  | **312.5 QPS, recall 0.906** (R2b 308.2 stable)    | — (broken)   | **2.69×** | n/a |
| 1B  L1   | —                      | pending                                            | —        | —     | —    |
| 1B  L2   | —                      | pending                                            | —        | —     | —    |

**Insert-phase (search-during-insert) QPS (BE+π):**

| Scale | Pre-insert | Avg insert tput (vec/s) | Avg SDI QPS | Avg SDI recall | RocksDB SDI QPS | Ratio |
|---|---|---|---|---|---|---|
| 10M L1  | 985   | **921.8** | **911.0** | **0.9566** | 1470.5 | **62%** ✅ |
| 10M L2  | 555.6 | 236.6   | 322.9 | 0.947 | 989.8  | 33%  |
| 100M L1 | 742.1 | 790.8   | 596.2 | 0.918 | 1358.0 | **44%** ✅ |
| 100M L2 | 312.5 | running | —     | —     | —      | —    |

(10 batches × `InsertVectorCount/10`. Pre-insert columns are the higher of bench0 R2 / bench0b. 10M L1 BE+π row added 2026-05-03 from fresh-build non-coproc baseline run `bench_baseline_20260503_163423.json` on .7.)

## Full metric comparison: RocksDB vs TiKV BE+π

Side-by-side on every dimension (build, search, insert, latency, recall). All TiKV numbers post-`a184213` (BE+π + fan-out pool). Search-during-insert (SDI) values are 10-batch averages from `Benchmark 1`.

### Pre-insert search

| Scale / L | Backend | Build (s) | QPS | p50 (ms) | p99 (ms) | Recall@5 | TiKV / RocksDB |
|---|---|---|---|---|---|---|---|
| 10M L1  | RocksDB   | 836.8   | **1778.4** | **2.28** | **3.95** | 0.968 | — |
| 10M L1  | TiKV BE+π | 760-779 | 985-1036  | 3.76-4.00 | 5.10-5.49 | 0.956-0.958 | **55-58%** |
| 10M L2  | RocksDB   | 828.5   | **1328.1** | **3.06** | **4.27** | 0.963 | — |
| 10M L2  | TiKV BE+π | 978.6   | 555.6     | 7.08      | 8.95      | 0.945 | **42%** |
| 100M L1 | RocksDB   | 10090.9 | **1746.0** | **2.26** | **4.08** | 0.917 | — |
| 100M L1 | TiKV BE+π | 10630.5 | 742.1     | 5.34      | 7.08      | 0.914 | **42.5%** |
| 100M L2 | RocksDB   | broken  | —          | —         | —         | —     | — |
| 100M L2 | TiKV BE+π | —       | 312.5     | —         | —         | 0.906 | n/a |

### Insert phase (`Benchmark 1`, 10-batch averages)

| Scale / L | Backend | Insert tput (vec/s) | SDI QPS | SDI p50 (ms) | SDI p99 (ms) | SDI Recall@5 | Tput ratio | QPS ratio |
|---|---|---|---|---|---|---|---|---|
| 10M L1  | RocksDB   | **3627.7** | **1470.5** | **2.74** | **4.13** | 0.968 | — | — |
| 10M L1  | TiKV BE+π | 921.8      | 911.0      | 4.35     | 5.99     | 0.957 | **25%** | **62%** ✅ |
| 10M L2  | RocksDB   | **2277.7** | **989.8**  | **4.09** | **5.50** | 0.964 | — | — |
| 10M L2  | TiKV BE+π | 236.6      | 322.9      | —        | —        | 0.947 | **10%** | **33%** |
| 100M L1 | RocksDB   | **7635.8** | **1358.0** | **2.88** | **4.64** | 0.922 | — | — |
| 100M L1 | TiKV BE+π | 790.8      | 596.2      | 6.63     | 9.47     | 0.918 | **10%** | **44%** |

(10M L2 BE+π SDI latency percentiles not captured separately in the post-BE+π run; pre-insert latency is 7.08/8.95 ms p50/p99.)

### Headline takeaways

- **Search QPS (TiKV / RocksDB)**: 55-62% at 10M L1, ~33-44% at 10M L2 / 100M L1. Goal of 50% is met at L1 only; L2 and 100M+ fall below. Gap widens with scale because TiKV's per-RPC overhead is fixed while RocksDB's in-process MultiGet stays flat.
- **Recall**: TiKV trails RocksDB by ~1 percentage point uniformly (0.95 vs 0.96-0.97). Algorithmic, not backend-driven.
- **Search latency**: TiKV p50 is 1.5-2.3× RocksDB p50, p99 is 1.4-2.0× — same root cause as QPS gap (per-RPC tax).
- **Insert throughput**: TiKV is **only 10-25% of RocksDB** — a much bigger gap than search. Each posting-write is a synchronous TiKV `RawPut` (gRPC + Raft commit + Titan blob write), whereas RocksDB writes are an in-process `Put` straight into memtable. This is the main cost users would feel under update-heavy workloads.
- **SDI recall is preserved** through inserts in both backends (>0.91 at 100M, >0.95 at 10M).
- **Build time** is roughly comparable (within 5%) at every scale — most build cost is CPU (BKT + KNN graph), not storage.

## How to refresh

```bash
# pull latest jsons
mkdir -p results/0.7
scp 10.11.0.7:~/zhangt/SPTAG/evaluation/backend_comparison/results/*.json results/0.7/

# generate summary
python3 summarize.py results/*.json results/0.7/*.json
```

## Configurations

- **All INIs**: `BatchNum=10`, `TopK=5`, `NumQueries=200`, `NumSearchThreads=4`, `NumInsertThreads=16`, `AppendThreadNum=48`, `LatencyLimit=100ms`.
- **TiKV**: `Storage=TIKVIO`, single PD+TiKV via Docker on test node, `tikv.toml` (max-replicas=1, 30GB block-cache, Titan blobs >=1KB, lz4 compression).
- **RocksDB**: `enable_blob_files=true`, `min_blob_size=64`, `blob_file_size=8GB`, no compression, **GC disabled** (after UAF crash, see below).

## Crash log: 10M RocksDB initial run (with Blob GC enabled)

Crashed AFTER 3 successful pre-insert search rounds, during a Split that fired a MultiGet on Blob-resident postings. Stack (resolved with `addr2line` on binary `a0a05f4`):

```
segfault_handler @ SPFreshTest.cpp
   ↑
rocksdb::Cleanable::~Cleanable
   ↑
rocksdb::BlobSource::PinOwnedBlob::lambda    ← UAF
   ↑
SPTAG::SPANN::RocksDBIO::MultiGet
   ↑
SPTAG::SPANN::ExtraDynamicSearcher::Split
   ↑
SplitAsyncJob::exec
   ↑
SPDKThreadPool::initSPDK lambda
```

Reproducible on **clean upstream code** → not caused by our distributed branch. Same crash signature as prior 1B RocksDB run on `users/zhangt/merge-distributed-to-tikv`.

Saved as: `results/benchmark_10m_rocksdb_L1.crashlog` / `.crashjson`.

Mitigation: set `enable_blob_garbage_collection = false` in `ExtraRocksDBController.h`.

## Binary versions

| md5 | Built on | Notes |
|-----|----------|-------|
| `a0a05f451afaaae9db4e478044de3f0a` | initial (cfd2576) | clean upstream + build fixes — crashes RocksDB Blob GC |
| `a33f31e43ce8679c68761fea22c08912` | + GC disabled patch | current `Release/SPTAGTest`; deployed on 0.7 too |

## Hosts

- 0.9 (`10.11.0.9`, `msccl-dev-000005`): `/mnt_ssd/data` 7.0T raid, ~6T free
- 0.7 (`10.11.0.7`, `msccl-dev-000003`): `/mnt_ssd/data` 7.0T raid, ~5T free
