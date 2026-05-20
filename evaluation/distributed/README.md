# Distributed Benchmark Evaluation — Insert Dominant

Multi-machine SPTAG SPANN distributed benchmark for an **insert-dominant** workload
(1M base + 10M inserts in 10 batches, with concurrent search-during-insert) on
SIFT1B. Each physical node runs its own independent PD + TiKV (no shared Raft
replication — see "TiKV deployment model" below).

## Files in this folder

| File | Purpose |
| --- | --- |
| `configs/benchmark_insert_dominant_template.ini` | Benchmark template; `run_distributed.sh` fills `IndexPath`, `TiKVPDAddresses`, `TiKVKeyPrefix`, and `[Distributed]` from `cluster.conf`. |
| `run_distributed.sh` | Orchestrator: `deploy` / `start-tikv` / `run` / `stop-tikv` / `cleanup`. |
| `README.md` | This file. |

## Architecture

```
                    ┌──────────────┐
                    │   Driver     │  (node 0)
                    │  RunBenchmark│
                    │   + Router   │
                    └──┬───┬───┬──┘
           TCP Dispatch│   │   │
              ┌────────┘   │   └────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Worker 1 │ │ Worker 2 │ │ Worker N │
        │  + Router│ │  + Router│ │  + Router│
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  TiKV 1  │ │  TiKV 2  │ │  TiKV N  │ (one PD + one TiKV per node)
        └──────────┘ └──────────┘ └──────────┘
```

- **Driver** (node 0): Builds the index, sends Search/Insert/Stop commands via TCP dispatch.
- **Workers** (nodes 1..N): Receive commands, execute their shard locally, report results back.
- **TiKV (per node)**: Each node runs its own independent PD + TiKV pair. Postings
  for a head live on the node that owns that head's hash partition.
- **PostingRouter**: Hash-based head routing, remote append, head sync, dispatch protocol.

## TiKV deployment model

Unlike a single-machine multi-docker TiKV (3 PD + 3 TiKV behind 127.0.0.1 ports
22791-3 / 20161-3 sharing one Raft cluster), in this multi-machine setup **each
node runs its own isolated PD + TiKV pair** under host networking. Heads are
routed to nodes by hash, and each node's TiKV stores only its own shard. There
is no Raft replication between nodes (no cross-node region quorum), which is
intentional for insert-dominated benchmarks where Raft log overhead would dominate.

Per-node ports (defaults from `cluster.conf`):

| Service | Port | Notes |
| --- | --- | --- |
| PD client | `2379` | Local app uses `<node_ip>:2379`. |
| PD peer | `2380` | Inter-PD; isolated cluster of 1 PD per node. |
| TiKV client | `20161` | The node-local SPTAG worker connects here. |
| Router | `30001+` | TCP dispatch / posting routing between nodes. |

## Prerequisites

- `Release/SPTAGTest` built with TiKV support on the driver node:
  ```bash
  cd <SPTAG_ROOT>
  cd ThirdParty/kvproto && ./generate_cpp.sh && cd ../..
  mkdir -p Release && cd Release
  cmake .. -DTIKV=ON -DTBB=ON -DCMAKE_BUILD_TYPE=Release -DGPU=OFF
  cmake --build . --target SPTAGTest -j$(nproc)
  ```
  *Note: building the full project may fail on the Java wrapper (`JAVASPTAGFileIO`)
  due to a pre-existing `FileIOInterface.h` signature mismatch — the `SPTAGTest`
  target alone is sufficient.*
- Passwordless SSH from driver to every other node (configure `ssh_key` in `cluster.conf`).
- Docker installed on every node (TiKV/PD run as containers in host network mode).
- Same dataset path on every node (default `/mnt/nvme/sift1b/`):
  - `/mnt/nvme/sift1b/bigann_base.u8bin` (1B × 128 × u8)
  - `/mnt/nvme/sift1b/query.10K.u8bin`
- Same fast-storage path for index + TiKV data on every node (`data_dir` in `cluster.conf`,
  default `/mnt/nvme`).

## Step 1 — Cluster config

```bash
cp evaluation/distributed/cluster.conf.example cluster.conf
vim cluster.conf
```

Example:

```ini
[cluster]
ssh_user=superbench
sptag_dir=/home/superbench/zhangt/SPTAG
data_dir=/mnt/nvme
tikv_version=v7.5.1
pd_version=v7.5.1

[nodes]
# host           router_port
10.0.1.1         30001          # driver (always first)
10.0.1.2         30002          # worker 1
10.0.1.3         30003          # worker 2

[tikv]
# host           pd_client  pd_peer  tikv_port
10.0.1.1         2379       2380     20161
10.0.1.2         2379       2380     20161
10.0.1.3         2379       2380     20161
```

`run_distributed.sh` reads this file to fill the template's `[Distributed]`,
`TiKVPDAddresses`, `IndexPath`, and `TiKVKeyPrefix` automatically.

## Step 2 — Deploy

```bash
./evaluation/distributed/run_distributed.sh deploy cluster.conf
```

This rsyncs `Release/SPTAGTest` (and required shared libs) to every node and
ensures the per-node TiKV / PD data directories exist under `data_dir`.

## Step 3 — Start TiKV (per-node, independent)

```bash
./evaluation/distributed/run_distributed.sh start-tikv cluster.conf
```

This starts one PD + one TiKV per node in host-network containers. Single-replica
placement (`max-replicas=1`) is set so we measure benchmark performance without
3-way Raft replication.

Health check (run on driver, repeat per node):

```bash
for ip in 10.0.1.1 10.0.1.2 10.0.1.3; do
  curl -s "http://$ip:2379/pd/api/v1/stores" \
    | python3 -c 'import json,sys; print([s["store"]["state_name"] for s in json.load(sys.stdin)["stores"]])'
done
# Each node should report ['Up'].
```

### Pre-split & scatter (optional but recommended)

For the insert-dominant workload to spread region writes evenly across regions
within a node's TiKV, pre-split the keyspace at boundaries derived from
`DBKey(headID) = MaxID*layer + headID` little-endian byte 0. The TiKV raw key is
`TiKVKeyPrefix + "_" + uint32_le(DBKey)`; for multi-chunk it appends `\x00` /
`\x02` for chunk / count keys, but we split *only* on the head-key prefix so all
chunk and count variants for a head share a region. Boundaries used: `0x02, 0x04,
…, 0xfe` (127 split points → 128 regions).

Driver-side helper (each PD is independent, so run per node):

```bash
PREFIX="bench_insert_dominant_3node"   # keep in sync with KEY_PREFIX in run_distributed.sh
for ip in 10.0.1.1 10.0.1.2 10.0.1.3; do
  PD="http://$ip:2379"
  PDCTL=(docker run --rm --network host --entrypoint /pd-ctl pingcap/pd:v7.5.1 -u "$PD")
  python3 - "$PREFIX" "${PDCTL[@]}" <<'PY'
import json, subprocess, sys
prefix = sys.argv[1].encode() + b'_'
pdctl = sys.argv[2:]
def run(args): return subprocess.check_output(pdctl + args, text=True)
def region_for(hex_key): return json.loads(run(['region', 'key', '--format=hex', hex_key]))['id']
for b in range(2, 256, 2):
    key = (prefix + bytes([b, 0, 0, 0])).hex()
    rid = region_for(key)
    run(['operator', 'add', 'split-region', str(rid), '--policy=usekey', '--keys', key])
for r in json.loads(run(['region', 'scan']))['regions']:
    run(['operator', 'add', 'scatter-region', str(r['id'])])
PY
done
```

Skip this on the very first run if you don't have load skew — `start-tikv` works
without it. For 1B-scale insert-dominant runs on a single node it materially
reduces head-region hot-spotting.

## Step 4 — Run the benchmark

```bash
# Single scale, explicit node count (driver + (N-1) workers):
./evaluation/distributed/run_distributed.sh run cluster.conf insert_dominant 3

# Or sweep 1-node baseline + N-node distributed for one or more scales:
./evaluation/distributed/run_distributed.sh bench cluster.conf insert_dominant
```

What `run` does:

1. **Build** (driver only): driver builds the index locally with router
   *disabled* (`Rebuild=true`, no `[Router]`). Output goes to `…_n0/spann_index`.
2. **Distribute**: rsync head index + perftest files from driver to each worker.
3. **Workers**: SSH-launches `SPTAGTest` on each worker with `WORKER_INDEX=i` and
   the per-node ini (router enabled, `Rebuild=false`).
4. **Driver**: relaunches `SPTAGTest` with router enabled, `Rebuild=false`. The
   driver dispatches Insert / Search commands across batches via TCP.
5. **Collect**: driver sends Stop, joins worker logs into `benchmark_logs/`.

Useful environment overrides (see header of `run_distributed.sh`):

- `NOCACHE=1` — disable TiKV block cache, OS pagecache, and `VersionCacheMaxChunks`.
- `BUILD_WITH_CACHE=1` — build with caches, then drop caches before search/insert (NOCACHE only).
- `SKIP_TIKV_SWAP=1` — when using `BUILD_WITH_CACHE`, skip the destructive TiKV
  container restart that has corrupted recall at 100M scale.
- `SKIP_SAVE_LOAD=1` — skip post-build SaveIndex / per-batch Load+Clone+Save (NOCACHE only).
- `SKIP_HEAD_BUILD=1` — reuse existing HeadIndex if present (RebuildSSDOnly).

## Step 5 — Stop / cleanup

```bash
./evaluation/distributed/run_distributed.sh stop-tikv cluster.conf
./evaluation/distributed/run_distributed.sh cleanup cluster.conf   # remove deployed files
```

## Key knobs in `benchmark_insert_dominant_template.ini`

| Key | Value | Meaning |
| --- | --- | --- |
| `BaseVectorCount` | 1_000_000 | Initial index build size. |
| `InsertVectorCount` / `BatchNum` | 10_000_000 / 10 | 10 batches × 1M inserts. |
| `NumSearchThreads` | 4 | Threads for the standalone post-batch query benchmark. |
| `NumInsertThreads` | 16 | Threads driving `AddIndex` calls on the driver. |
| `AppendThreadNum` | 144 | Async append worker pool size — overprovisioned (≈3× cores) because each thread is I/O-bound on TiKV RPCs, so high concurrency increases in-flight RPCs. |
| `NumSearchDuringInsertThreads` | 1 | Concurrent search threads while inserting (continuous loop, ~1s sleep per query). |
| `NumQueries` | 200 | Size of the rotating query pool (in-insert search loops over it). |
| `WorkerTimeout` | 14400 | Seconds a worker waits for the driver before exiting. |
| `Storage` / `TiKVKeyPrefix` / `TiKVPDAddresses` | `TIKVIO` / filled / filled | Filled by `run_distributed.sh` from `cluster.conf`. |
| `Layers` | 2 | SPANN multi-layer head. |
| `BuildSSDIndex.UseMultiChunkPosting` | false | Single-key posting layout (one TiKV value per head). |
| `BuildSSDIndex.PostingPageLimit` | 8 | Posting page limit; runtime cap is logged as ~246 vectors. |
| `BuildSSDIndex.PostingCountCacheCapacity` | 1_000_000 | Posting-count cache capacity. |
| `BuildSSDIndex.DistributedVersionMap` | true | Use TiKV-backed distributed version map. |
| `BuildSSDIndex.ReassignK` | 64 | Split/reassign target fanout knob. |
| `BuildSSDIndex.AsyncMergeInSearch` | true | Async merge during search. |
| `BuildSSDIndex.VersionCacheMaxChunks` | 100_000 | Local version-chunk cache (set ≤0 to disable). |
| `BuildSSDIndex.LatencyLimit` | 100 | ms latency cap fed to SPANN. |
| `BuildSSDIndex.MaxCheck` | 8192 | Max posting checks per query. |
| `BuildSSDIndex.SearchInternalResultNum` | 64 | Internal candidate count during search. |

## Output JSON structure (per batch)

For each insert batch, `output.json/results.benchmark1_insert.batch_N` contains:

- `Load timeSeconds` / `Load vectorCount` — reload of previous batch.
- `Clone timeSeconds`.
- In-insert concurrent search stats (continuous-loop variant):
  `numQueries` (actual count issued), `meanLatency`, `p50/p90/p95/p99`, `qps`,
  `batch barrier waitSeconds`.
- `inserted`, `insert timeSeconds`, `insert throughput`.
- `search` and `search_round2` — standalone `BenchmarkQueryPerformance` results
  against the post-batch index (cold + warm), independent of the in-insert numbers.
- `save timeSeconds`.

Pre-insert baseline lives at `results.benchmark0_query_before_insert` and
`results.benchmark0b_query_before_insert_round2`.

## Dispatch Protocol

The TCP dispatch protocol replaces file-based barriers. Communication flows through
PostingRouter's existing TCP transport:

| Packet | Direction | Purpose |
|--------|-----------|---------|
| `DispatchCommand (0x09)` | Driver → Worker | Search/Insert/Stop with `dispatchId` + round. |
| `DispatchResult (0x89)` | Worker → Driver | Status + wallTime for aggregation. |

- **Search**: Driver broadcasts to workers, runs local queries in parallel, collects
  wall times for percentile stats.
- **Insert**: Driver broadcasts batch index, workers insert their shard, driver
  waits for all to finish.
- **Stop**: Driver sends at end of benchmark; workers exit gracefully.

Each command has a unique `dispatchId` (monotonic uint64) to avoid round collisions
between search and insert operations.

## Troubleshooting

- **Workers don't connect**: confirm `RouterNodeAddrs` ports (default 30001+) are
  reachable between every pair of nodes — the router uses TCP with 2 io_context
  threads.
- **TiKV timeout**: ensure each node's PD `advertise-client-urls` use a reachable
  IP (not 127.0.0.1) — `start-tikv` sets this from `cluster.conf`. Check
  `docker logs sptag-pd-0` on the affected node.
- **Worker exits prematurely**: check the worker logs in `benchmark_logs/`.
  Common causes: TiKV not ready, index path mismatch, router connection failure.
- **Build fails on Java wrapper**: pre-existing issue unrelated to the benchmark.
  Build only what's needed:
  ```bash
  cmake --build . --target SPTAGTest -j$(nproc)
  ```
