# Evaluation 2026-04-23 — SPFresh + TiKV (SIFT1B, 10× insert, single-key posting)

This run measures SPANN/SPFresh insert + concurrent search performance against a
3-node local TiKV cluster, using the SIFT1B dataset (1M base + 10M inserts in
10 batches). Concurrent in-insert search threads now run continuously (1 query
per ~1s) until inserts finish — see code change in `Test/src/SPFreshTest.cpp`
(`InsertVectors`).

## Files in this folder

| File | Purpose |
| --- | --- |
| `tikv.toml` | TiKV server config mounted into all 3 `tikv-server` containers as `/opt/tikv.toml`. |
| `benchmark_spfresh_sift1b_v10_multichunk.ini` | Benchmark config consumed by `SPTAGTest` via `BENCHMARK_CONFIG` env var. |
| `README.md` | This file. |

## 1. Prerequisites

- Built `SPTAGTest` binary (Release): `/home/azureuser/qiazh/SPTAG/Release/SPTAGTest`
  (rebuild with `cd build && make -j4` if source changed).
- Docker daemon running.
- Dataset present on disk:
  - `/mnt/data_disk/sift1b/base.1B.u8bin`
  - `/mnt/data_disk/sift1b/query.public.10K.u8bin`
- TiKV data directories under `/mnt/nvme_striped/qiazh/tikv/` (`pd{1,2,3}-data`,
  `tikv{1,2,3}-data`, `tikv{1,2,3}-logs`, `pd{1,2,3}-logs`).

## 2. Start the TiKV cluster (3 PD + 3 TiKV, host network)

All 6 containers run with `--network host`. The TiKV containers mount
`/mnt/nvme_striped/qiazh/tikv/tikv.toml` as read-only `/opt/tikv.toml` and pass
`--config=/opt/tikv.toml`.

### 2a. (Optional) Wipe previous data for a clean run

```bash
docker rm -f tikv-pd1 tikv-pd2 tikv-pd3 tikv-tikv1 tikv-tikv2 tikv-tikv3 2>/dev/null
sudo rm -rf /mnt/nvme_striped/qiazh/tikv/{pd1-data,pd2-data,pd3-data}/*
sudo rm -rf /mnt/nvme_striped/qiazh/tikv/{tikv1-data,tikv2-data,tikv3-data}/*
sudo rm -rf /mnt/nvme_striped/qiazh/tikv/{pd1-logs,pd2-logs,pd3-logs,tikv1-logs,tikv2-logs,tikv3-logs}/*
```

### 2b. Start PD (placement driver) — 3 nodes

```bash
for i in 1 2 3; do
  port_client=2379$i  # 23791 / 23792 / 23793
  port_peer=2380$i    # 23801 / 23802 / 23803
  docker run -d --name tikv-pd$i --network host \
    -v /mnt/nvme_striped/qiazh/tikv/pd$i-data:/data \
    -v /mnt/nvme_striped/qiazh/tikv/pd$i-logs:/logs \
    pingcap/pd:nightly \
    --name=pd$i \
    --client-urls=http://0.0.0.0:$port_client \
    --peer-urls=http://0.0.0.0:$port_peer \
    --advertise-client-urls=http://127.0.0.1:$port_client \
    --advertise-peer-urls=http://127.0.0.1:$port_peer \
    --initial-cluster=pd1=http://127.0.0.1:23801,pd2=http://127.0.0.1:23802,pd3=http://127.0.0.1:23803 \
    --data-dir=/data/pd$i \
    --log-file=/logs/pd.log
done
```

### 2c. Start TiKV — 3 nodes

```bash
for i in 1 2 3; do
  addr_port=2016$i   # 20161 / 20162 / 20163
  status_port=2018$i # 20181 / 20182 / 20183
  docker run -d --name tikv-tikv$i --network host \
    -v /mnt/nvme_striped/qiazh/tikv/tikv$i-data:/data \
    -v /mnt/nvme_striped/qiazh/tikv/tikv$i-logs:/logs \
    -v /mnt/nvme_striped/qiazh/tikv/tikv.toml:/opt/tikv.toml:ro \
    pingcap/tikv:nightly \
    --pd-endpoints=127.0.0.1:23791,127.0.0.1:23792,127.0.0.1:23793 \
    --addr=0.0.0.0:$addr_port \
    --advertise-addr=127.0.0.1:$addr_port \
    --status-addr=0.0.0.0:$status_port \
    --data-dir=/data/tikv$i \
    --log-file=/logs/tikv.log \
    --config=/opt/tikv.toml
done
```

### 2d. Configure single-replica placement

For this local performance benchmark, keep one TiKV replica per region to avoid
measuring 3-way Raft replication overhead. PD accepts this setting after the
TiKV cluster has bootstrapped.

```bash
docker exec tikv-pd1 /pd-ctl -u http://127.0.0.1:23791 config set max-replicas 1
docker exec tikv-pd1 /pd-ctl -u http://127.0.0.1:23791 config show \
  | grep -E 'max-replicas|location-labels|strictly-match-label'
# Expected: "max-replicas": 1
```

### 2e. Verify cluster health

```bash
docker ps --filter name=tikv- --format "table {{.Names}}\t{{.Status}}"
# Should show 6 containers Up.

# Confirm stores are alive (need pd-ctl or curl). Quick check:
curl -s http://127.0.0.1:23791/pd/api/v1/stores | grep -c '"state_name": "Up"'
# Expected: 3
```

### 2f. Pre-split and scatter the SPFresh TiKV keyspace

Do this after the cluster is healthy and before starting `SPTAGTest`. The split
keys must match SPTAG's raw TiKV key layout, not printable decimal IDs:

- single-key posting: `TiKVKeyPrefix + "_" + uint32_le(DBKey(headID))`
- multi-chunk posting: `TiKVKeyPrefix + "_" + uint32_le(DBKey(headID)) + chunk_suffix`

Split method:

- `DBKey(postingID) = MaxID * layer + postingID` in `ExtraDynamicSearcher::DBKey`.
- TiKV prefixing is byte-oriented: `MakePrefixedKey(rawKey)` stores `TiKVKeyPrefix`, then ASCII `_`, then the raw 4-byte little-endian `DBKey`.
- Multi-chunk uses the same prefix and raw head key, then adds a suffix:
  - base/chunk keys: `TiKVKeyPrefix + "_" + uint32_le(DBKey(headID)) + "\x00" + optional_chunk_id`
  - count keys: `TiKVKeyPrefix + "_" + uint32_le(DBKey(headID)) + "\x02"`
- Region split keys deliberately stop at `TiKVKeyPrefix + "_" + uint32_le(boundary)` and do not include `"\x00"` or `"\x02"`. This cuts the shared head-key prefix, so single-key postings, multi-chunk base/chunk keys, and multi-chunk count keys for the same head range land in the same region bucket.
- Boundaries are the low byte of the little-endian raw key: `0x02, 0x04, ..., 0xfe`. Each split key is `b"spfresh_sift1b_" + bytes([boundary, 0, 0, 0])`.

Examples:

```text
boundary 0x02 -> raw bytes: spfresh_sift1b_\x02\x00\x00\x00
boundary 0x02 -> hex:       737066726573685f7369667431625f02000000
boundary 0x04 -> hex:       737066726573685f7369667431625f04000000
```

For this benchmark `TiKVKeyPrefix=spfresh_sift1b`, `Layers=2`, and the hot
partitioning byte is the low byte of the raw little-endian head key. The script
below creates split points at `0x02, 0x04, ..., 0xfe`, then scatters the resulting
regions so leaders/peers are balanced across the 3 TiKV stores.

```bash
PD=http://127.0.0.1:23791
PDCTL='docker run --rm --network host --entrypoint /pd-ctl pingcap/pd:nightly'

python3 - <<'PY'
import json
import subprocess

pd = 'http://127.0.0.1:23791'
pdctl = ['docker', 'run', '--rm', '--network', 'host', '--entrypoint', '/pd-ctl', 'pingcap/pd:nightly', '-u', pd]
prefix = b'spfresh_sift1b_'

def run(args):
  return subprocess.check_output(pdctl + args, text=True)

def region_for_key(hex_key):
  out = run(['region', 'key', '--format=hex', hex_key])
  return json.loads(out)['id']

split_ok = 0
for i, b in enumerate(range(2, 256, 2), 1):
  key = (prefix + bytes([b, 0, 0, 0])).hex()
  region_id = region_for_key(key)
  print(f'split {i}/127 region={region_id} key={key}')
  out = run(['operator', 'add', 'split-region', str(region_id), '--policy=usekey', '--keys', key])
  if 'Success' in out:
    split_ok += 1

regions = json.loads(run(['region', 'scan']))['regions']
print(f'split_ok={split_ok} region_count={len(regions)}')
for region in regions:
  region_id = region['id']
  print(f'scatter region={region_id}')
  run(['operator', 'add', 'scatter-region', str(region_id)])
PY
```

Verify the split/scatter result:

```bash
$PDCTL -u "$PD" region scan \
  | python3 -c 'import json,sys; data=json.load(sys.stdin); print("regions", data["count"])'

curl -s "$PD/pd/api/v1/stores" | python3 - <<'PY'
import json, sys
stores = json.load(sys.stdin)['stores']
for item in stores:
  store = item['store']
  status = item.get('status', {})
  print(store['id'], store['address'], store['state_name'], 'leaders', status.get('leader_count'))
PY
```

## 3. Run the benchmark

```bash
cd /home/azureuser/qiazh/SPTAG

# Make sure prior index dir is removed for a clean rebuild
rm -rf /mnt/nvme_striped/qiazh/sptag_bench/spfresh_sift1b_v10_multichunk/spann_index

LOG=benchmark_$(date +%Y%m%d_%H%M%S).log
BENCHMARK_CONFIG=$(pwd)/evaluation/2026-04-23/benchmark_spfresh_sift1b_v10_multichunk.ini \
BENCHMARK_OUTPUT=$(pwd)/evaluation/2026-04-23/output.json \
nohup ./Release/SPTAGTest --run_test=SPFreshTest/BenchmarkFromConfig \
  > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
```

Monitor:

```bash
tail -F "$LOG"
# Per-batch results are streamed into output.json as the benchmark runs.
```

Stop / cleanup:

```bash
pkill -f SPTAGTest                 # stop benchmark
docker stop $(docker ps -q --filter name=tikv-)   # stop cluster
```

## 4. Key knobs in `benchmark_spfresh_sift1b_v10_multichunk.ini`

| Key | Value | Meaning |
| --- | --- | --- |
| `BaseVectorCount` | 1_000_000 | initial index build size |
| `InsertVectorCount` / `BatchNum` | 10_000_000 / 10 | 10 batches × 1M inserts |
| `NumSearchThreads` | 4 | threads for standalone post-batch query benchmark |
| `NumInsertThreads` | 16 | threads driving `AddIndex` calls |
| `NumSearchDuringInsertThreads` | 1 | concurrent search threads while inserting (continuous loop, ~1s sleep per query) |
| `NumQueries` | 200 | size of the rotating query pool (in-insert search loops over it) |
| `Storage` / `TiKVPDAddresses` / `TiKVKeyPrefix` | TIKVIO / 127.0.0.1:23791-3 / spfresh_sift1b | TiKV backend wiring |
| `Layers` | 2 | SPANN multi-layer head |
| `BuildSSDIndex.UseMultiChunkPosting` | false | uses single-key posting list layout |
| `BuildSSDIndex.PostingPageLimit` | 8 | posting page limit; runtime posting size limit is logged as 246 vectors |
| `BuildSSDIndex.PostingCountCacheCapacity` | 1_000_000 | posting count cache capacity |
| `BuildSSDIndex.DistributedVersionMap` | true | uses TiKV-backed distributed version map |
| `BuildSSDIndex.ReassignK` | 64 | split/reassign target fanout knob |
| `BuildSSDIndex.AsyncMergeInSearch` | true | async merge during search |
| `BuildSSDIndex.VersionCacheMaxChunks` | 100_000 | enables version chunk cache when greater than 0 |
| `BuildSSDIndex.AsyncRpcMaxInflight` | 512 | caps total in-flight async TiKV RPCs per `TiKVIO`; `0` disables throttling |
| `BuildSSDIndex.LatencyLimit` | 100 | ms latency cap fed to SPANN |

## 5. Output JSON structure (per batch)

For each insert batch, `output.json/results.benchmark1_insert.batch_N` contains:

- `Load timeSeconds` / `Load vectorCount` — reload of previous batch
- `Clone timeSeconds`
- In-insert concurrent search stats (now from continuous loop):
  `numQueries` (actual count issued), `meanLatency`, `p50/p90/p95/p99`, `qps`
- `inserted`, `insert timeSeconds`, `insert throughput`
- `search` and `search_round2` — standalone `BenchmarkQueryPerformance`
  results against the post-batch index (cold + warm), independent of the
  in-insert search numbers above
- `save timeSeconds`

Pre-insert baseline lives at:
`results.benchmark0_query_before_insert` and `results.benchmark0b_query_before_insert_round2`.
