# SPTAG + TiKV Vector Search Benchmark Reproduction Guide

## 1. Repositories

| Project | Repository | Branch | Description |
|---------|-----------|--------|-------------|
| SPTAG | https://github.com/microsoft/SPTAG | `users/qiazh/merge-spfresh-tikv` | TiKV storage backend + Region routing cache |
| TiKV | https://github.com/zqxjjz/tikv | `<compatible-tikv-branch>` | TiKV storage backend benchmark support |

```bash
git clone https://github.com/microsoft/SPTAG.git && cd SPTAG && git checkout users/qiazh/merge-spfresh-tikv
git clone https://github.com/zqxjjz/tikv.git && cd tikv && git checkout <compatible-tikv-branch>
```

## 2. Prerequisites

- **OS**: Ubuntu 24.04 (tested)
- **CPU**: 48 cores
- **Memory**: 377GB (3 TiKV instances x 80GB + system)
- **Disk**: NVMe SSD, 11TB
- **Software**:
  - Docker (for PD)
  - cmake >= 3.12, gcc >= 5.0, protoc >= 3.21
  - libboost-all-dev (>= 1.67), libtbb-dev, libgrpc++-dev, libprotobuf-dev
  - Rust nightly (rustc 1.95.0-nightly, auto-installed via TiKV's rust-toolchain.toml)

```bash
sudo apt install -y build-essential cmake protobuf-compiler libprotobuf-dev \
    libgrpc++-dev libgrpc-dev protobuf-compiler-grpc \
    libboost-all-dev libtbb-dev docker.io

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## 3. Building

### 3.1 Build SPTAG

```bash
cd SPTAG

# Generate kvproto gRPC stubs
cd ThirdParty/kvproto && ./generate_cpp.sh && cd ../..

# Build
mkdir build && cd build
cmake .. -DTIKV=ON -DTBB=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Output: build/Release/SPTAGTest
```

### 3.2 Build TiKV

```bash
cd tikv
make release
# Output: target/release/tikv-server
# First build takes ~30-60 minutes
```

## 4. Deploy TiKV Cluster

### 4.1 Start PD (3 nodes, Docker)

Create `docker-compose.yml`:

```yaml
services:
  pd1:
    image: pingcap/pd:nightly
    container_name: tikv-pd1
    network_mode: host
    volumes:
      - ./pd1-data:/data
      - ./pd1-logs:/logs
    command:
      - --name=pd1
      - --client-urls=http://0.0.0.0:23791
      - --peer-urls=http://0.0.0.0:23801
      - --advertise-client-urls=http://127.0.0.1:23791
      - --advertise-peer-urls=http://127.0.0.1:23801
      - --initial-cluster=pd1=http://127.0.0.1:23801,pd2=http://127.0.0.1:23802,pd3=http://127.0.0.1:23803
      - --data-dir=/data/pd1
      - --log-file=/logs/pd.log
    healthcheck:
      test: ["CMD", "curl", "-f", "http://127.0.0.1:23791/pd/api/v1/health"]
      interval: 10s
      timeout: 5s
      retries: 10
      start_period: 40s

  pd2:
    image: pingcap/pd:nightly
    container_name: tikv-pd2
    network_mode: host
    volumes:
      - ./pd2-data:/data
      - ./pd2-logs:/logs
    command:
      - --name=pd2
      - --client-urls=http://0.0.0.0:23792
      - --peer-urls=http://0.0.0.0:23802
      - --advertise-client-urls=http://127.0.0.1:23792
      - --advertise-peer-urls=http://127.0.0.1:23802
      - --initial-cluster=pd1=http://127.0.0.1:23801,pd2=http://127.0.0.1:23802,pd3=http://127.0.0.1:23803
      - --data-dir=/data/pd2
      - --log-file=/logs/pd.log

  pd3:
    image: pingcap/pd:nightly
    container_name: tikv-pd3
    network_mode: host
    volumes:
      - ./pd3-data:/data
      - ./pd3-logs:/logs
    command:
      - --name=pd3
      - --client-urls=http://0.0.0.0:23793
      - --peer-urls=http://0.0.0.0:23803
      - --advertise-client-urls=http://127.0.0.1:23793
      - --advertise-peer-urls=http://127.0.0.1:23803
      - --initial-cluster=pd1=http://127.0.0.1:23801,pd2=http://127.0.0.1:23802,pd3=http://127.0.0.1:23803
      - --data-dir=/data/pd3
      - --log-file=/logs/pd.log
```

```bash
docker compose up -d
# Verify
curl http://127.0.0.1:23791/pd/api/v1/health
```

### 4.2 TiKV Configuration (`tikv.toml`)

```toml
memory-usage-limit = "80GB"

[server]
grpc-concurrency = 8
grpc-memory-pool-quota = "4GB"

[raftstore]
region-max-size = "512MB"
region-split-size = "384MB"
region-max-keys = 5120000
region-split-keys = 3840000

[rocksdb]
max-background-jobs = 8
max-sub-compactions = 3
rate-bytes-per-sec = "0"

[rocksdb.defaultcf]
write-buffer-size = "256MB"
max-write-buffer-number = 5
min-write-buffer-number-to-merge = 2
level0-file-num-compaction-trigger = 8
level0-slowdown-writes-trigger = 20
level0-stop-writes-trigger = 36
max-bytes-for-level-base = "1GB"
compression-per-level = ["no", "no", "lz4", "lz4", "lz4", "zstd", "zstd"]
target-file-size-base = "64MB"

[rocksdb.writecf]
write-buffer-size = "128MB"
max-write-buffer-number = 5

[storage]
reserve-space = "1GB"

[storage.block-cache]
capacity = "40GB"

[coprocessor]
region-max-size = "512MB"
region-split-size = "384MB"
region-max-keys = 5120000
region-split-keys = 3840000
```

> **Memory tuning**: Adjust based on your machine's total memory and number of instances:
> - `memory-usage-limit` ≈ total_memory / num_instances × 0.7
> - `block-cache.capacity` ≈ memory-usage-limit × 0.5

### 4.3 Start TiKV (3 nodes)

```bash
TIKV_BIN=/path/to/tikv/target/release/tikv-server
DATA_DIR=/path/to/tikv-data
TIKV_CONF=${DATA_DIR}/tikv.toml

# Stop Docker TiKV if previously running
docker stop tikv-tikv1 tikv-tikv2 tikv-tikv3 2>/dev/null || true

# TiKV 1
$TIKV_BIN --config=$TIKV_CONF \
    --pd-endpoints=127.0.0.1:23791,127.0.0.1:23792,127.0.0.1:23793 \
    --addr=0.0.0.0:20161 --advertise-addr=127.0.0.1:20161 \
    --status-addr=0.0.0.0:20181 \
    --data-dir=${DATA_DIR}/tikv1-data/tikv1 \
    --log-file=${DATA_DIR}/tikv1-logs/tikv.log &

# TiKV 2
$TIKV_BIN --config=$TIKV_CONF \
    --pd-endpoints=127.0.0.1:23791,127.0.0.1:23792,127.0.0.1:23793 \
    --addr=0.0.0.0:20162 --advertise-addr=127.0.0.1:20162 \
    --status-addr=0.0.0.0:20182 \
    --data-dir=${DATA_DIR}/tikv2-data/tikv2 \
    --log-file=${DATA_DIR}/tikv2-logs/tikv.log &

# TiKV 3
$TIKV_BIN --config=$TIKV_CONF \
    --pd-endpoints=127.0.0.1:23791,127.0.0.1:23792,127.0.0.1:23793 \
    --addr=0.0.0.0:20163 --advertise-addr=127.0.0.1:20163 \
    --status-addr=0.0.0.0:20183 \
    --data-dir=${DATA_DIR}/tikv3-data/tikv3 \
    --log-file=${DATA_DIR}/tikv3-logs/tikv.log &

# Wait and verify
sleep 10
curl http://127.0.0.1:23791/pd/api/v1/stores | python3 -m json.tool
# All 3 stores should show state "Up"
```

## 5. Test Data

Data files use SPTAG standard binary format:

| File | Description | Used in |
|------|-------------|---------|
| `perftest_vector.bin.UInt8_1000000_128` | 1M UInt8 vectors, dim=128 | 1M test |
| `perftest_vector.bin.UInt8_10000000_128` | 10M UInt8 vectors, dim=128 | 10M test |
| `perftest_vector.bin.UInt8_100000000_128` | 100M UInt8 vectors, dim=128 | 100M test |
| `perftest_query.bin.UInt8_200_128` | 200 query vectors | All tests |

File format: raw binary, each vector is 128 consecutive bytes (UInt8). Place all data files in the benchmark working directory.

## 6. Benchmark Configuration Files

### 6.1 1M dataset (`benchmark_1m_l2_batchget.ini`)

```ini
[Benchmark]
VectorPath=perftest_vector.bin.UInt8_1000000_128
QueryPath=perftest_query.bin.UInt8_200_128
TruthPath=truth_1m_l2_batchget
IndexPath=/path/to/bench/proidx_1m_l2_batchget/spann_index
ValueType=UInt8
Dimension=128
BaseVectorCount=990000
InsertVectorCount=10000
DeleteVectorCount=0
BatchNum=10
TopK=5
NumThreads=16
NumQueries=200
DistMethod=L2
Rebuild=true
Resume=-1
Layers=2

Storage=TIKVIO
TiKVPDAddresses=127.0.0.1:23791,127.0.0.1:23792,127.0.0.1:23793
TiKVKeyPrefix=batchget_1m_l2

[BuildSSDIndex]
PostingPageLimit=12
BufferLength=8
```

### 6.2 10M dataset (`benchmark_10m_l2_batchget.ini`)

```ini
[Benchmark]
VectorPath=perftest_vector.bin.UInt8_10000000_128
QueryPath=perftest_query.bin.UInt8_200_128
TruthPath=truth_10m_l2_batchget
IndexPath=/path/to/bench/proidx_10m_l2_batchget/spann_index
ValueType=UInt8
Dimension=128
BaseVectorCount=9900000
InsertVectorCount=100000
DeleteVectorCount=0
BatchNum=10
TopK=5
NumThreads=16
NumQueries=200
DistMethod=L2
Rebuild=true
Resume=-1
Layers=2

Storage=TIKVIO
TiKVPDAddresses=127.0.0.1:23791,127.0.0.1:23792,127.0.0.1:23793
TiKVKeyPrefix=batchget_10m_l2

[BuildSSDIndex]
PostingPageLimit=12
BufferLength=8
```

### 6.3 100M dataset (`benchmark_100m_l2_batchget.ini`)

```ini
[Benchmark]
VectorPath=perftest_vector.bin.UInt8_100000000_128
QueryPath=perftest_query.bin.UInt8_200_128
TruthPath=truth_100m_l2_batchget
IndexPath=/path/to/bench/proidx_100m_l2_batchget/spann_index
ValueType=UInt8
Dimension=128
BaseVectorCount=99000000
InsertVectorCount=1000000
DeleteVectorCount=0
BatchNum=10
TopK=5
NumThreads=16
NumQueries=200
DistMethod=L2
Rebuild=true
Resume=-1
Layers=2

Storage=TIKVIO
TiKVPDAddresses=127.0.0.1:23791,127.0.0.1:23792,127.0.0.1:23793
TiKVKeyPrefix=batchget_100m_l2

[BuildSSDIndex]
PostingPageLimit=12
BufferLength=8
```

## 7. Running Benchmarks

```bash
cd /path/to/bench   # directory containing data files

# Run (BENCHMARK_CONFIG = config file, BENCHMARK_OUTPUT = JSON result file)
BENCHMARK_CONFIG=benchmark_1m_l2_batchget.ini \
BENCHMARK_OUTPUT=output_1m.json \
./SPTAGTest --run_test=BenchmarkFromConfig --log_level=message

# Background run (recommended for large datasets)
nohup bash -c 'BENCHMARK_CONFIG=benchmark_10m_l2_batchget.ini \
BENCHMARK_OUTPUT=output_10m.json \
./SPTAGTest --run_test=BenchmarkFromConfig --log_level=message' \
> test_10m.log 2>&1 &

# Monitor progress
tail -f test_10m.log
```

### Estimated Build Times

| Dataset | Build Time | TiKV Disk Usage |
|---------|-----------|-----------------|
| 1M | ~16 min | ~a few GB |
| 10M | ~3 hours | ~tens of GB |
| 100M | ~27 hours | ~hundreds of GB |

### Query-only (skip build)

After the index is built, set `Rebuild=false` to run queries only:

```ini
Rebuild=false
```

### Resume from interruption

If a build is interrupted, resume from the last completed batch:

```ini
Rebuild=true
Resume=3          # resume from batch 4 (0-based index)
```

## 8. Configuration Parameters

### Benchmark Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `Storage` | `TIKVIO` | Use TiKV as posting list storage backend |
| `Layers` | `2` | Two-layer SPANN index |
| `TopK` | `5` | Return top 5 nearest vectors |
| `NumThreads` | `16` | Number of search threads |
| `NumQueries` | `200` | Number of queries per benchmark round |
| `BatchNum` | `10` | Number of insert batches |
| `TiKVKeyPrefix` | varies | TiKV key prefix; use different prefixes for different tests to avoid conflicts |
| `PostingPageLimit` | `12` | Max pages per posting list |
| `BufferLength` | `8` | Write buffer size |

### Internal Search Parameters (hardcoded, no configuration needed)

These are set in source code and written to `indexloader.ini` during index build:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SearchInternalResultNum` | `64` | Number of candidates returned by head graph search |
| `SearchPostingPageLimit` | `12` | Max posting pages during search (overridden by ini PostingPageLimit) |
| `MaxCheck` | `4096` | Max BKT search check count |
| `LatencyLimit` | `50.0` | Latency limit (ms) |
| `AllowZeroReplica` | `false` | Do not allow zero replicas |
| `InternalResultNum` | `64` | Internal result count |

## 9. Reference Results

### Initial Query Latency (benchmark0_query_before_insert)

| Test | Date | Mean (ms) | P50 (ms) | P99 (ms) | QPS | Recall@5 |
|------|------|----------|---------|---------|-----|----------|
| 100K FileIO (local disk) | Mar 24 | 3.49 | 3.40 | 5.24 | 4335 | 0.916 |
| 100K TiKV RawGet (per-key) | Mar 24 | 56.20 | 56.63 | 62.70 | 278 | 0.918 |
| 100K TiKV RawBatchGet | Mar 25 | 7.58 | 7.51 | 10.59 | 2054 | 0.924 |
| 1M TiKV (old binary) | Mar 27 | 63.23 | 63.67 | 73.74 | 248 | 0.783 |
| **1M TiKV (new binary)** | **Mar 27** | **14.98** | **13.98** | **27.11** | **1049** | **0.783** |
| 1M TiKV fresh build (warm cache) | Mar 30 | 9.99 | 10.16 | 14.37 | 1563 | 0.792 |
| 10M TiKV (old binary) | Mar 25 | 51.97 | 52.10 | 64.96 | 302 | 0.692 |
| 100M TiKV (old binary) | Mar 25 | 48.09 | 48.36 | 59.66 | 326 | N/A |

> **"old binary" vs "new binary"**: The new binary includes RawBatchGet region-grouped dispatch + Region routing cache optimization. 10M/100M have not been re-tested with the new binary yet.

### Key Factors Affecting Latency

1. **Block Cache warm/cold**: TiKV block cache (`capacity = 40GB`) — if it can hold all posting data, queries approach in-memory speed. 1M posting data is ~a few hundred MB (fits entirely in cache); 10M+ may exceed cache capacity.
2. **Binary version**: The new binary has Region routing cache, avoiding PD lookups on every query, significantly reducing latency.
3. **Dataset size**: Larger datasets have more scattered postings, leading to higher cache miss rates.

### Expected Results When Reproducing

| Dataset | Cold Cache (first query) | Warm Cache (after warmup) |
|---------|------------------------|--------------------------|
| 1M | ~50-65ms | ~10-15ms |
| 10M | ~50ms | ~20-30ms |
| 100M | ~50ms | ~40-50ms |

**Recommendation**: After the index build completes, run one warmup query round first, then use the second round as the official result.

## 10. Troubleshooting

### TiKV OOM
If TiKV instances get OOM-killed, check `memory-usage-limit` and `block-cache.capacity`. Total memory across all instances should not exceed 80% of physical memory.

### Build interrupted by TiKV crash
Use the `Resume` parameter to continue from the last completed batch — no need to rebuild from scratch.

### Recall is 0
In the 100M test, Recall=0 occurs because the truth file is mismatched (truth was generated with FileIO, not matching TiKV-stored postings). Use `TruthPath=none` or regenerate the truth file.

### Will different key prefixes conflict?
No. `TiKVKeyPrefix` determines the key prefix in TiKV. Different tests use different prefixes (`batchget_1m_l2`, `batchget_10m_l2`, etc.) and do not interfere with each other.
