# Backend Comparison Results: RocksDB vs TiKV (SPFresh)

Branch: `users/zhangt/backend-comparison` (forked from upstream `users/qiazh/merge-spfresh-tikv` @ `8a74694`).
Dataset: SIFT (UInt8, dim=128, L2). Truth gen on-the-fly.

Workload: each run does
1. Build SSD index from `BaseVectorCount` vectors
2. Pre-insert search (`Benchmark 0`, 2 rounds)
3. **`Benchmark 1`**: 10 batches, each inserts `InsertVectorCount/10` vectors and runs search concurrently — measures insert throughput + search-during-update QPS / latency / recall

| Scale | Backend | Layers | Node | Status | Build (s) | Pre-insert QPS | Pre-insert recall | Insert tput (vec/s) | Search-during-insert QPS | Search-during-insert recall |
|-------|---------|--------|------|--------|-----------|----------------|-------------------|---------------------|--------------------------|------------------------------|
| 10M   | TiKV    | L1     | 0.7  | ✅ done | 924.0 | 578.5 | 0.9520 | 937.8 | 507.7 | 0.9531 |
| 10M   | RocksDB | L1     | 0.9  | 🔄 rerun | — | — | — | — | — | — |
| 100M  | TiKV    | L1     | 0.7  | 🟡 running | — | — | — | — | — | — |
| 100M  | RocksDB | L1     | 0.9  | pending | — | — | — | — | — | — |
| 10M   | TiKV    | L2     | TBD  | pending | — | — | — | — | — | — |
| 10M   | RocksDB | L2     | TBD  | pending | — | — | — | — | — | — |
| 100M  | TiKV    | L2     | TBD  | pending | — | — | — | — | — | — |
| 100M  | RocksDB | L2     | TBD  | pending | — | — | — | — | — | — |
| 1B    | TiKV    | L1     | TBD  | pending | — | — | — | — | — | — |
| 1B    | TiKV    | L2     | TBD  | pending | — | — | — | — | — | — |
| 1B    | RocksDB | L1     | TBD  | pending | — | — | — | — | — | — |
| 1B    | RocksDB | L2     | TBD  | pending | — | — | — | — | — | — |

(All numbers use binary `a33f31e` unless noted. Avg insert throughput / search QPS / recall are means across the 10 insert batches.)

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
