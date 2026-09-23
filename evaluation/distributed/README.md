# Distributed Benchmark Evaluation — Insert Dominant

Multi-machine SPTAG SPANN distributed benchmark for an **insert-dominant** workload
(1M base + 1M-10M inserts in batches, with concurrent search-during-insert) on
SIFT1B. All nodes share a single TiKV raft cluster (see "TiKV deployment model"
below).

## Files in this folder

| File | Purpose |
| --- | --- |
| `configs/benchmark_insert_dominant_template.ini` | 1M base + 1M insert, search-during-insert workload. |
| `configs/benchmark_10m_template.ini` | 9M base + 1M insert, growing-index workload. |
| `configs/benchmark_100m_template.ini` | 99M base + 1M insert, steady-state/freshness workload. |
| `configs/cluster_2node.conf`, `configs/cluster_3node.conf` | Example cluster topologies. Pick one (or write your own) and pass to the orchestrator. |
| `configs/tikv.toml` | TiKV server config baked into the containers. |
| `run_distributed.sh` | Orchestrator: `deploy` / `setup-bins` / `start-tikv` / `run` / `bench` / `stop-tikv` / `cleanup`. |
| `bin/` | `tikv-server` + `pd-server` binaries used by the containers (`setup-bins` downloads them if missing). |
| `README.md` | This file. |

`run_distributed.sh` fills the template's `IndexPath`, `TiKVPDAddresses`,
`TiKVKeyPrefix`, and `[Distributed]` section from the cluster config.

## Architecture

```
                    ┌────────────────────┐
                    │   Driver = Worker 0│  (node 0)
                    │   + Dispatcher     │
                    └─┬──┬──┬────────────┘
       TCP Dispatch  │  │  │       ▲ ▲ ▲
        (broadcast)  │  │  │       │ │ │  status replies
              ┌──────┘  │  └──────┐│ │ │
              ▼         ▼         ▼│ │ │
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Worker 1 │ │ Worker 2 │ │ Worker N │
        └──┬───▲───┘ └──┬───▲───┘ └──┬───▲───┘
           │   │        │   │        │   │
           └───┴────────┴───┴────────┴───┘
              PostingRouter peer-to-peer
              (remote append / head sync /
               merge hints, by hash owner)
                          │
                          ▼
                ┌───────────────────┐
                │ Shared TiKV raft  │  N PDs (one raft group) +
                │ cluster           │  N TiKV stores (max-replicas=1)
                └───────────────────┘
```

- **Driver** (node 0): also runs as **worker 0**. On top of the worker role,
  it owns the dispatcher: builds the initial index, then broadcasts
  Search/Insert/Stop commands to the other workers over TCP dispatch.
- **Workers** (nodes 0..N-1): each owns a shard of the head index by hash.
  Workers talk to each other peer-to-peer through PostingRouter for remote
  append, head sync, and merge hints — there is no driver-mediated forwarding.
  On each `DispatchCommand` they execute the local part of the request and
  report status back to the dispatcher.
- **Shared TiKV cluster**: every node runs a PD + TiKV container; all PDs join
  one raft group, all TiKVs point to all PDs. PD routes each key to the store
  that owns its region.
- **PostingRouter**: hash-based head routing, remote append, head sync, and
  the TCP dispatch transport used by the dispatcher.

## TiKV deployment model

All nodes share **one** TiKV raft cluster: every node's PD joins the same raft
group, every node's TiKV registers as a store in that cluster, and PD routes
reads/writes to whichever store owns the region. `max-replicas=1` is set so
each region lives on exactly one store — we measure benchmark performance
without 3-way Raft replication. Compute nodes are stateless TiKV clients; they
read any posting through the shared client, so there is no cross-compute fetch
RPC during RNGSelection.

Per-node ports (defaults from `configs/cluster_2node.conf`):

| Service | Default port | Notes |
| --- | --- | --- |
| PD client | `23791` | TiKV client + `pd-ctl` connect here. |
| PD peer | `23801` | Inter-PD raft traffic. |
| TiKV client | `20171` | Per-node TiKV listens here. |
| Router | `30002+` | TCP dispatch / posting routing between nodes. **Driver's `router_port` must NOT be `30001`** — the dispatcher listens on `30001` and a collision will silently break worker registration. The shipped 2-node config uses `30011` on the driver for this reason. |

## Prerequisites

- `Release/SPTAGTest` built with TiKV support on the driver node:
  ```bash
  cd <SPTAG_ROOT>
  cd ThirdParty/kvproto && ./generate_cpp.sh && cd ../..
  mkdir -p Release && cd Release
  cmake .. -DTIKV=ON -DTBB=ON -DCMAKE_BUILD_TYPE=Release -DGPU=OFF
  cmake --build . --target SPTAGTest -j$(nproc)
  ```
  *Note: building the full project may fail on the Java wrapper
  (`JAVASPTAGFileIO`) due to a pre-existing `FileIOInterface.h` signature
  mismatch — the `SPTAGTest` target alone is sufficient.*
- Passwordless SSH from driver to every other node (configure `ssh_key` in
  the cluster config).
- Docker installed on every node (TiKV/PD run as containers in host network mode).
- Same dataset path on every node (default `/mnt/nvme/sift1b/`):
  - `/mnt/nvme/sift1b/bigann_base.u8bin` (1B × 128 × u8)
  - `/mnt/nvme/sift1b/query.10K.u8bin`
- Same fast-storage path for index + TiKV data on every node (`data_dir` in
  the cluster config, default `/mnt/nvme`).

## Step 1 — Cluster config

Pick one of the shipped templates and edit it for your hosts/paths:

```bash
cp evaluation/distributed/configs/cluster_2node.conf my_cluster.conf
vim my_cluster.conf
```

Layout:

```ini
[cluster]
ssh_user=superbench
ssh_key=/home/superbench/.ssh/id_rsa
sptag_dir=/home/superbench/zhangt/SPTAG
data_dir=/mnt/nvme
tikv_version=v8.5.1
pd_version=v8.5.1

[nodes]
# host         router_port    (driver is first; router_port must not equal 30001)
10.0.1.1       30011          # driver
10.0.1.2       30002          # worker 1

[tikv]
# host         pd_client_port  pd_peer_port  tikv_port
10.0.1.1       23791           23801         20171
10.0.1.2       23791           23801         20171
```

`run_distributed.sh` reads this file to fill the template's `[Distributed]`,
`TiKVPDAddresses`, `IndexPath`, and `TiKVKeyPrefix` automatically.

## Step 2 — Deploy

```bash
./evaluation/distributed/run_distributed.sh deploy      my_cluster.conf
./evaluation/distributed/run_distributed.sh setup-bins  my_cluster.conf
```

`deploy` rsyncs `Release/SPTAGTest` (and required shared libs) to every node
and ensures per-node TiKV / PD data directories exist under `data_dir`.
`setup-bins` downloads `tikv-server` / `pd-server` into `bin/` on every node
(idempotent; skipped automatically by `start-tikv` if binaries are already
present).

## Step 3 — Start the shared TiKV cluster

```bash
./evaluation/distributed/run_distributed.sh start-tikv my_cluster.conf
```

This starts one PD + one TiKV container per node in host-network mode and
joins them into a single raft cluster (`max-replicas=1`, no 3-way replication).

Health check (single PD endpoint is enough — the cluster is shared):

```bash
curl -s "http://10.0.1.1:23791/pd/api/v1/stores" \
  | python3 -c 'import json,sys; print([s["store"]["state_name"] for s in json.load(sys.stdin)["stores"]])'
# Expected: ['Up', 'Up'] (one entry per TiKV store).
```

## Step 4 — Run the benchmark

```bash
# Single scale, explicit node count (driver + (N-1) workers):
./evaluation/distributed/run_distributed.sh run my_cluster.conf insert_dominant 2

# Or sweep 1-node baseline + N-node distributed for one or more scales:
./evaluation/distributed/run_distributed.sh bench my_cluster.conf insert_dominant
./evaluation/distributed/run_distributed.sh bench my_cluster.conf all
```

What `run` does:

1. **Build** (driver only): driver builds the index locally with router
   *disabled* (`Rebuild=true`, no `[Distributed]`). Output goes to
   `…_n0/spann_index`. Because the TiKV cluster is shared, the driver writes
   all postings straight to TiKV via PD-routed RPCs — there is no need for a
   distributed build phase.
2. **Distribute**: rsync head index + perftest files from driver to each worker.
3. **Workers**: SSH-launches `SPTAGTest` on each remote worker (nodes 1..N-1)
   with `WORKER_INDEX=i` and the per-node ini (router enabled,
   `Rebuild=false`). Workers wire PostingRouter so they can reach every peer
   directly for remote append / head sync.
4. **Driver**: relaunches `SPTAGTest` on node 0 with router enabled,
   `Rebuild=false`. The same process acts as **worker 0** (owns its hash
   shard like any other worker) **and** as the dispatcher (broadcasts Insert
   / Search / Stop over TCP and waits for status replies).
5. **Collect**: driver sends Stop, joins worker logs into `benchmark_logs/`.

> The "build on the driver, then distribute and run" split is a workaround:
> we don't yet have a real distributed SelectHead/BuildHead implementation, so
> Phase 1 is single-node-with-shared-TiKV. The `BuildOnly=true` /
> `RebuildSSDOnly=true` / `SkipSaveLoadCycles=true` /
> `tikv_switch_to_nocache` / `drop_caches` choreography exists because of
> this split; it is not a feature of the steady-state design.

Useful environment overrides (see the header of `run_distributed.sh` for the
authoritative list):

- `NOCACHE=1` — disable TiKV block cache, OS pagecache, and
  `VersionCacheMaxChunks` for the search/insert phase.
- `BUILD_WITH_CACHE=1` — build with caches enabled, then drop caches before
  search/insert (requires `NOCACHE=1`). Used at 100M scale where building
  under nocache is impractical.
- `SKIP_TIKV_SWAP=1` — with `BUILD_WITH_CACHE`, skip the destructive TiKV
  container restart that has corrupted recall at 100M scale. Relies on
  drop_caches + `VersionCacheMaxChunks=0` for nocache semantics.
- `SKIP_SAVE_LOAD=1` — skip the post-build SaveIndex / per-batch
  Load+Clone+Save cycle (`SkipSaveLoadCycles=true`). Required at 100M scale.
- `SKIP_HEAD_BUILD=1` — reuse existing HeadIndex if present
  (`RebuildSSDOnly=true`); falls back to full build if HeadIndex is missing.

## Step 5 — Stop / cleanup

```bash
./evaluation/distributed/run_distributed.sh stop-tikv my_cluster.conf
./evaluation/distributed/run_distributed.sh cleanup   my_cluster.conf   # remove deployed files
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
