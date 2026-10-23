# Reproduce the SIFT1M experiments

[Project home](../../README.md) | [Results and limitations](README.md) |
[Parameter and methodology reference](../../docs/Benchmarking.md)

Build and evaluate your own FILEIO and TiKV indexes from the first 1M SIFT1B
vectors. This procedure covers installation, inputs, configuration, cluster
setup, execution and output validation. It requires no index snapshots,
archived binaries or data from the original experiment.

These are Linux, single-host instructions. The recorded measurements are
partial runs, not guaranteed timings or a completed ten-batch benchmark.
The procedure runs a new 990K build followed by ten 1K insertion batches.
It reproduces the workload and measures new results; it does not replay the
saved index used for the report's TiKV queries. TiKV updates have an unresolved
checkpoint-statistics wait and throughput anomaly, so a full run may time out.
Do not interpret the recipe as evidence that those issues are fixed.

## Before you start

**Start here, even if you have not installed SPTAG.** The SDK quick start,
general benchmark reference and TiKV integration guide are not prerequisites.

| Requirement | What to prepare |
| --- | --- |
| Host | Linux on AVX2-capable x86-64; Bash and sudo access for installation |
| Storage | SSD space for generated data, indexes and checkpoints; the recorded TiKV data alone occupied roughly 235 GiB, excluding build and FILEIO artifacts |
| Memory | Three TiKV processes each have an 8 GB configured limit, plus PD, build and client memory; this is not a measured minimum host-memory requirement |
| Dataset | Existing SIFT1B UInt8 files **or** the public downloads in step 3; only about 128 MB of base vectors and 1.28 MB of queries need downloading |
| TiKV only | Docker Engine, permission to use it, and twelve free local ports; the recipe creates 3 PD + 3 TiKV containers on one host |
| Time | Exact truth generation, index construction and updates are substantial work; the two-hour per-backend timeout is a configurable cap, not an estimate |

Choose a path:

- **FILEIO only:** steps 1-5, then validate FILEIO in step 8. Skip steps 6, 7
  and 9; Docker and cluster resources are not required.
- **FILEIO and TiKV:** follow all nine steps in order. Both use the same
  TiKV-enabled executable, so both paths need the build libraries in step 1.

On completion, each backend has `run.log`, `exit-code.txt`, `output.json`,
its rendered config and index checkpoints under a new run directory. Step 8
checks whether all ten batches actually completed. See the
[recorded results](README.md) for interpretation, not an exact pass/fail target
for recall or latency.

Run the steps in one Bash session and stop on any failed command. Do not paste
the whole page as an unattended script: service-readiness checks may need to be
repeated. Each experiment gets a new directory and TiKV key prefix. Do not
reuse a cluster or data directory from an existing experiment.

## 1. Build one executable for both backends

The recorded experiment used Ubuntu 22.04-era dependencies, GCC 11.4,
Protobuf 3.12.4 and gRPC 1.30.2. The SDK baseline also documents Ubuntu 24.04;
using its newer packages is a different environment and must be recorded.
Python 3 and curl are installed below. For the TiKV path, install
[Docker Engine for Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
if it is not already available. Before building or running experiments,
confirm the daemon and your access:

```bash
sudo -v
sudo -n docker info
```

Skip that Docker check for FILEIO-only runs. If your account already has
Docker access without sudo, use `docker info` and omit `sudo -n` from later
Docker commands. Do not continue with TiKV setup if the check fails.

On Ubuntu, install the build dependencies if they are missing:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git \
  libboost-system-dev libboost-thread-dev libboost-serialization-dev \
  libboost-regex-dev libboost-filesystem-dev libboost-test-dev \
  libtbb-dev libnuma-dev protobuf-compiler libprotobuf-dev libgrpc++-dev \
  protobuf-compiler-grpc libabsl-dev pkg-config python3 curl
```

For a new checkout, skip unrelated Git LFS datasets:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch users/qiazh/doc \
  https://github.com/microsoft/SPTAG.git SPTAG-repro
cd SPTAG-repro
export SPTAG_ROOT="$(pwd -P)"
export RECIPE="$SPTAG_ROOT/evaluation/2026-10-23-sift1m"
git submodule update --init ThirdParty/zstd
git merge-base --is-ancestor 192995914332cc2ecca9fbfde5742ee73166214c HEAD

bash ThirdParty/kvproto/generate_cpp.sh
cmake -S . -B build-tikv -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DFOLLY=OFF -DTBB=ON -DSPDK=OFF -DROCKSDB=OFF -DTIKV=ON \
  -DGPU=OFF -DRABITQ=OFF -DLARGEVID=OFF
cmake --build build-tikv --target SPTAGTest --parallel 8
(cd Release && ./SPTAGTest \
  --run_test=CommonHelperTest:IniReaderTest:TestDataGeneratorTest \
  --log_level=message)
```

Use lower build parallelism if memory is constrained. Regenerating stubs is
necessary when the installed Protobuf differs from the vendored generator
version. Preserve that diff as provenance; do not accidentally submit it.
For an existing checkout, set `SPTAG_ROOT` and `RECIPE` to its absolute paths
instead of cloning again. Record its actual commit and dirty state.

Both backends deliberately use this **same TiKV-enabled executable**; FILEIO
selects its local posting/version-map path at runtime. Do not rebuild a different
configuration into the shared `Release` directory between the runs.

## 2. Allocate isolated run paths and preserve provenance

```bash
export RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ROOT="$SPTAG_ROOT/benchmark-output/sift1m-repro-$RUN_TAG"
mkdir -p "$SPTAG_ROOT/benchmark-output"
mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/input" "$RUN_ROOT/fileio" "$RUN_ROOT/tikv" "$RUN_ROOT/bin"
export FILEIO_RUN="$RUN_ROOT/fileio"
export TIKV_RUN="$RUN_ROOT/tikv"
export BENCH_BIN="$RUN_ROOT/bin/SPTAGTest"
cp -p "$SPTAG_ROOT/Release/SPTAGTest" "$BENCH_BIN"
git -C "$SPTAG_ROOT" rev-parse HEAD > "$RUN_ROOT/source-commit.txt"
git -C "$SPTAG_ROOT" status --short > "$RUN_ROOT/source-status.txt"
git -C "$SPTAG_ROOT" diff --binary HEAD > "$RUN_ROOT/source.patch"
git -C "$SPTAG_ROOT" submodule status > "$RUN_ROOT/submodules.txt"
cp "$SPTAG_ROOT/build-tikv/CMakeCache.txt" "$RUN_ROOT/CMakeCache.txt"
{
  g++ --version
  cmake --version
  protoc --version
  pkg-config --modversion grpc++ protobuf absl_synchronization
  lscpu
  free -h
  df -h "$RUN_ROOT"
} > "$RUN_ROOT/environment.txt"
sha256sum "$BENCH_BIN" > "$RUN_ROOT/binary.sha256"
```

The `mkdir "$RUN_ROOT"` above must succeed for a **new** directory; do not
continue after a collision.

## 3. Prepare the actual SIFT inputs

Choose **one** input source below, then run the extraction and checks on this
page. The expected format is two little-endian uint32 header fields followed
by UInt8 vectors, not `.bvecs` or `.fvecs`.

### Option A: use existing local data

Replace the source paths if your dataset is mounted elsewhere:

```bash
export SIFT_BASE=/mnt_ssd/data/sift1b/bigann_base.u8bin
export QUERY_SOURCE=/mnt_ssd/data/sift1b/query.10K.u8bin
test -r "$SIFT_BASE"
test -r "$QUERY_SOURCE"
```

### Option B: download the required public data

The [Big ANN Benchmarks dataset definitions](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/datasets.py)
identify the SIFT/BigANN files `base.1B.u8bin` and
`query.public.10K.u8bin`. Their public filenames differ from the local names
in option A. Check the dataset provider's terms before use.

Fetch only the first 1M base rows with an HTTP range request, not the full
128 GB base file. The downloaded prefix still has the original 1B-row header,
so it is **not** a runnable input until the extraction below rewrites it.
The size cap makes a server that ignores the range fail rather than download
the whole corpus. Use a fresh run directory; do not overwrite existing inputs.

```bash
export SIFT_BASE="$RUN_ROOT/input/base-first-1m.download"
export QUERY_SOURCE="$RUN_ROOT/input/query.public.10K.u8bin"
test ! -e "$SIFT_BASE"
test ! -e "$QUERY_SOURCE"
curl --fail --location --show-error --max-time 1800 \
  --range 0-128000007 --max-filesize 128000008 \
  --output "$SIFT_BASE" --write-out '%{http_code}\n' \
  https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin \
  > "$RUN_ROOT/input/base-http-status.txt"
test "$(cat "$RUN_ROOT/input/base-http-status.txt")" = 206
test "$(stat -c %s "$SIFT_BASE")" = 128000008
curl --fail --location --show-error --max-time 300 \
  --max-filesize 1280008 --output "$QUERY_SOURCE" \
  https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin
```

If a download fails or the server does not return HTTP 206 for the base,
stop and investigate; do not feed a partial/error response to the harness.

### Extract and verify the 1M input (both options)

This writes a new header `(1000000, 128)`, copies only those vectors, and
refuses to overwrite output. It works with either the complete local base
or the downloaded prefix. Keep the source unchanged.

```bash
export SIFT_1M="$RUN_ROOT/input/sift-first-1m.u8bin"
python3 - <<'PY'
import os
import struct

count = 1_000_000
with open(os.environ["SIFT_BASE"], "rb") as source:
    rows, dimension = struct.unpack("<II", source.read(8))
    if rows < count or dimension != 128:
        raise ValueError("Expected at least 1M UInt8 vectors of dimension 128")
    with open(os.environ["SIFT_1M"], "xb") as target:
        target.write(struct.pack("<II", count, dimension))
        remaining = count * dimension
        while remaining:
            block = source.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise EOFError("Vector data ended before the requested prefix")
            target.write(block)
            remaining -= len(block)
PY
```

Then verify the inputs:

```bash
python3 - <<'PY'
import os
import struct
from pathlib import Path

vectors = Path(os.environ["SIFT_1M"])
queries = Path(os.environ["QUERY_SOURCE"])
with vectors.open("rb") as stream:
    assert struct.unpack("<II", stream.read(8)) == (1000000, 128)
assert vectors.stat().st_size == 128000008
with queries.open("rb") as stream:
    count, dimension = struct.unpack("<II", stream.read(8))
assert count >= 200 and dimension == 128
assert queries.stat().st_size == 8 + count * dimension
print("Inputs valid; the harness will use the first 200 query rows")
PY
sha256sum "$SIFT_1M" "$QUERY_SOURCE" > "$RUN_ROOT/inputs.sha256"
```

For the exact source data used in the report, expected SHA256 values are:

| File | SHA256 |
| --- | --- |
| Extracted 1M vectors | `7cdd76b29c338b78bf17c2ad5060b383110b0bf6ba914e8a77e7e642ba7a6111` |
| Original 10K query file | `eca755831fc9a8004e14886df48b81109a8ed3bfc6b632509c93c0460a30d552` |

Different hashes mean different inputs; do not claim an exact-data reproduction.

## 4. Render the two complete configurations

The checked-in [benchmark.ini](benchmark.ini) is a template, not a directly
runnable config with valid local paths. This block fills every input/output
path, chooses the runtime backend, and sets a unique TiKV prefix.

```bash
export PD_CLIENT_BASE=33990 PD_PEER_BASE=34000
export TIKV_BASE=30360 TIKV_STATUS_BASE=30380
export PD_ENDPOINTS="127.0.0.1:$((PD_CLIENT_BASE+1)),127.0.0.1:$((PD_CLIENT_BASE+2)),127.0.0.1:$((PD_CLIENT_BASE+3))"
export TRUTH_NAME=perftest_batchtruth.L2.UInt8_990000_128_200_5_1000_0_10
python3 - <<'PY'
import configparser
import os
from pathlib import Path

for directory, storage in [("FILEIO_RUN", "FILEIO"), ("TIKV_RUN", "TIKVIO")]:
    run = Path(os.environ[directory])
    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str
    with (Path(os.environ["RECIPE"]) / "benchmark.ini").open() as stream:
        config.read_file(stream)
    config["Benchmark"]["VectorPath"] = os.environ["SIFT_1M"]
    config["Benchmark"]["QueryPath"] = os.environ["QUERY_SOURCE"]
    config["Benchmark"]["IndexPath"] = str(run / "index")
    config["BuildSSDIndex"]["Storage"] = storage
    if storage == "TIKVIO":
        config["BuildSSDIndex"]["TiKVPDAddresses"] = os.environ["PD_ENDPOINTS"]
        config["BuildSSDIndex"]["TiKVKeyPrefix"] = "sift1m_repro_" + os.environ["RUN_TAG"]
    with (run / "benchmark.ini").open("x") as stream:
        config.write(stream, space_around_delimiters=False)
    print(run / "benchmark.ini")
PY
sha256sum "$FILEIO_RUN/benchmark.ini" "$TIKV_RUN/benchmark.ini" \
  > "$RUN_ROOT/configs.sha256"
```

Keep `BaseVectorCount=990000`, `InsertVectorCount=10000`, `BatchNum=10`.
That means 1,000 inserts per batch, **not** 10,000 per batch. Do not change
`BatchNum` alone to stop earlier while reusing the original phase truth.
Each independent rerun needs a new `RUN_TAG`, directories and prefix.

## 5. Run FILEIO first

This command starts real index construction and updates, not a smoke test.
The two-hour timeout is an operator-selected upper bound, not an expected
runtime or a completion guarantee; adjust it to your machine before starting.

```bash
(
  cd "$FILEIO_RUN"
  test ! -e output.json
  set +e
  BENCHMARK_CONFIG="$PWD/benchmark.ini" BENCHMARK_OUTPUT="$PWD/output.json" \
    timeout --signal=TERM --kill-after=30s 2h "$BENCH_BIN" \
    --run_test=SPFreshTest/BenchmarkFromConfig --log_level=message \
    > run.log 2>&1
  status=$?
  printf '%s\n' "$status" > exit-code.txt
  exit "$status"
)
```

Monitor `tail -f "$FILEIO_RUN/run.log"` from another terminal.
Exit 124 means timeout, not success; interrupted runs can leave incomplete JSON.
Do not rerun in that directory, silently repair the JSON, or treat a partial run
as ten completed batches. Diagnose the log and preserve the original output.

The generator writes the 990K base, 10K insertion vectors, first 200 queries,
and exact truth for all eleven phases **before** building the index.
There is no separate truth download. A nonempty `TruthPath` enables generation;
the automatically named truth file is reused when already present.

After FILEIO has stopped, verify and copy its truth for TiKV:

```bash
python3 - <<'PY'
import hashlib
import os
import struct
from pathlib import Path

path = Path(os.environ["FILEIO_RUN"]) / os.environ["TRUTH_NAME"]
with path.open("rb") as stream:
    assert struct.unpack("<ii", stream.read(8)) == (4400, 5)
assert path.stat().st_size == 88008
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == "b41cb5504f00cb49f00af2fc228f8128fcebc6d5b6ffa09ecf3d644d90911320"
print("Exact phase truth verified:", digest)
PY
cp -p "$FILEIO_RUN/$TRUTH_NAME" "$TIKV_RUN/$TRUTH_NAME"
```

The hardcoded hash applies to the exact SIFT inputs above. A mismatch must be
investigated, not bypassed while retaining an exact-reproduction claim.
Copying this small file is sufficient: TiKV derives its own vectors from the
same source. Do not copy index directories or reuse FILEIO's index as TiKV.

## 6. Create an isolated local TiKV cluster

Skip this section for FILEIO-only evaluation. The commands use `sudo -n docker`,
as on the original machine. If your account already has Docker access, omit
`sudo -n`. Do not run unrelated cleanup scripts or stop other users' containers.

Check that the twelve configured ports are free:

```bash
python3 - <<'PY'
import os
import socket

sockets = []
try:
    for name in ("PD_CLIENT_BASE", "PD_PEER_BASE", "TIKV_BASE", "TIKV_STATUS_BASE"):
        for offset in (1, 2, 3):
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(("127.0.0.1", int(os.environ[name]) + offset))
    print("All twelve experiment ports are free")
finally:
    for sock in sockets:
        sock.close()
PY
```

If any port is occupied, choose different bases and rerender the TiKV config
in a new run directory. A port precheck is not a reservation; any later bind
failure must also be investigated.

Pin the images used by the report; do not substitute a moving `nightly` tag:

```bash
export PD_IMAGE=pingcap/pd@sha256:2a79836fcfe4948103b9eed9f69cc664208c3e6d5bae5e12d0c01e29c3d1598e
export TIKV_IMAGE=pingcap/tikv@sha256:b6893e84e1914cff9fd262daac479ab24ef07ad99b15fd2b33457d02cb9ee38a
sudo -n docker pull "$PD_IMAGE"
sudo -n docker pull "$TIKV_IMAGE"
sudo -n docker image inspect "$PD_IMAGE" "$TIKV_IMAGE" \
  > "$RUN_ROOT/images.json"
export CLUSTER_ROOT="$RUN_ROOT/cluster"
export CONTAINER_PREFIX="sptag-repro-$RUN_TAG"
mkdir "$CLUSTER_ROOT"
cp "$RECIPE/tikv.toml" "$CLUSTER_ROOT/tikv.toml"
INITIAL="repro-pd1=http://127.0.0.1:$((PD_PEER_BASE+1)),repro-pd2=http://127.0.0.1:$((PD_PEER_BASE+2)),repro-pd3=http://127.0.0.1:$((PD_PEER_BASE+3))"
for i in 1 2 3; do
  mkdir "$CLUSTER_ROOT/pd$i"
  sudo -n docker run -d --name "$CONTAINER_PREFIX-pd$i" --network host \
    -v "$CLUSTER_ROOT/pd$i:/data" "$PD_IMAGE" \
    --name="repro-pd$i" \
    --client-urls="http://127.0.0.1:$((PD_CLIENT_BASE+i))" \
    --advertise-client-urls="http://127.0.0.1:$((PD_CLIENT_BASE+i))" \
    --peer-urls="http://127.0.0.1:$((PD_PEER_BASE+i))" \
    --advertise-peer-urls="http://127.0.0.1:$((PD_PEER_BASE+i))" \
    --initial-cluster="$INITIAL" --data-dir=/data/pd --log-file=/data/pd.log
done
```

Wait until PD health passes; repeat this check if the processes are still
starting. Do not proceed on a failed check:

```bash
curl --fail --max-time 10 \
  "http://127.0.0.1:$((PD_CLIENT_BASE+1))/pd/api/v1/health" \
  > "$RUN_ROOT/pd-health.json"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert len(d)==3 and all(x["health"] for x in d)' \
  "$RUN_ROOT/pd-health.json"
```

Start the three storage processes:

```bash
for i in 1 2 3; do
  mkdir "$CLUSTER_ROOT/tikv$i"
  sudo -n docker run -d --name "$CONTAINER_PREFIX-tikv$i" --network host \
    -v "$CLUSTER_ROOT/tikv$i:/data" \
    -v "$CLUSTER_ROOT/tikv.toml:/config.toml:ro" "$TIKV_IMAGE" \
    --pd-endpoints="$PD_ENDPOINTS" \
    --addr="127.0.0.1:$((TIKV_BASE+i))" \
    --advertise-addr="127.0.0.1:$((TIKV_BASE+i))" \
    --status-addr="127.0.0.1:$((TIKV_STATUS_BASE+i))" \
    --data-dir=/data/tikv --log-file=/data/tikv.log --config=/config.toml
done
curl --fail --max-time 10 \
  "http://127.0.0.1:$((PD_CLIENT_BASE+1))/pd/api/v1/stores" \
  > "$RUN_ROOT/stores-before.json"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["count"]==3 and all(x["store"]["state_name"]=="Up" for x in d["stores"])' \
  "$RUN_ROOT/stores-before.json"
curl --fail --max-time 10 \
  "http://127.0.0.1:$((PD_CLIENT_BASE+1))/pd/api/v1/config/replicate" \
  > "$RUN_ROOT/replication.json"
python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["max-replicas"]==3' \
  "$RUN_ROOT/replication.json"
```

Repeat the store check until all three stores are Up. PD health alone is not
proof that posting reads or updates work. The image digests are the observed
experiment pins, not a compatibility promise for arbitrary TiKV releases.
If an image is unavailable, obtain that exact image or explicitly declare
a different environment; do not silently replace it.

## 7. Run TiKV with the same binary

```bash
sha256sum -c "$RUN_ROOT/binary.sha256"
(
  cd "$TIKV_RUN"
  test ! -e output.json
  set +e
  BENCHMARK_CONFIG="$PWD/benchmark.ini" BENCHMARK_OUTPUT="$PWD/output.json" \
    timeout --signal=TERM --kill-after=30s 2h "$BENCH_BIN" \
    --run_test=SPFreshTest/BenchmarkFromConfig --log_level=message \
    > run.log 2>&1
  status=$?
  printf '%s\n' "$status" > exit-code.txt
  exit "$status"
)
```

TiKV update runs have encountered a checkpoint-statistics wait and a throughput
anomaly. A timeout is an incomplete run, even if the last index save succeeded.
Preserve `run.log`, `output.json`, configs, container logs and PD store state.
Do not switch to `Rebuild=false` and assume a safe read-only resume.

## 8. Decide whether a full run actually completed

For each backend, set `RUN_DIR` and run the same check:

```bash
export RUN_DIR="$FILEIO_RUN"  # Repeat with "$TIKV_RUN".
python3 - <<'PY'
import configparser
import json
import os
from pathlib import Path

run = Path(os.environ["RUN_DIR"])
assert (run / "exit-code.txt").read_text().strip() == "0", "Nonzero benchmark exit"
data = json.loads((run / "output.json").read_text())
config, results = data["config"], data["results"]
assert config["baseVectorCount"] == 990000 and config["insertVectorCount"] == 10000
assert config["BatchNum"] == 10 and config["DeleteVectorCount"] == 0
assert config["dimension"] == 128 and config["layers"] == 2
assert config["numQueries"] == 200 and config["topK"] == 5
assert config["numSearchThreads"] == 16 and config["numInsertThreads"] == 4
assert config["ValueType"] == "UInt8" and config["DistMethod"] == "L2"
batches = results["benchmark1_insert"]
assert set(batches) == {f"batch_{i}" for i in range(1, 11)}
rows = [(0, 1, results["benchmark0_query_before_insert"]),
        (0, 2, results["benchmark0b_query_before_insert_round2"])]
for phase in range(1, 11):
    batch = batches[f"batch_{phase}"]
    assert batch["inserted"] == 1000
    assert batch["Load vectorCount"] == 990000 + (phase - 1) * 1000
    assert "save timeSeconds" in batch
    rows.extend([(phase, 1, batch["search"]), (phase, 2, batch["search_round2"])])
assert (run / "index/checkpoint.txt").read_text().strip() == "9"
loader = configparser.ConfigParser()
with (run / "index_9/indexloader.ini").open() as stream:
    loader.read_file(stream)
assert loader.getint("Base", "VectorSize") == 1000000
for phase, round_id, row in rows:
    recall = row["recall"]
    assert row["numQueries"] == recall["numQueries"] == 200
    assert recall["k"] == 5 and 0 <= recall["recallAtK"] <= 1
    print(phase, 990000 + phase * 1000, round_id,
          recall["recallAtK"], row["meanLatency"], row["p95"], row["p99"], row["qps"])
print("Complete: 10 batches, final saved count 1000000, 22 measured query rows")
PY
```

Columns are phase, active vectors, round, Recall@5, mean/P95/P99 ms and QPS.
This checks completion and output shape, not a claim of a universal recall
threshold. Fresh ANN builds have unpinned randomness; measure recall against
the matching truth rather than requiring identical IDs or a previously
observed recall value from a different index.
Any failed assertion or JSON parse means the run has **not** passed this check.
The checkpoint stores zero-based batch indices: `9` means ten completed batches.

For a two-backend run, compare the hashes of these generated files between
the two run directories. **Skip this comparison for FILEIO-only evaluation.**

```bash
for directory in "$FILEIO_RUN" "$TIKV_RUN"; do
  (
    cd "$directory"
    sha256sum perftest_vector.bin.UInt8_990000_128 \
      perftest_addvector.bin.UInt8_10000_128 \
      perftest_query.bin.UInt8_200_128 "$TRUTH_NAME"
  )
done
```

For the exact SIFT inputs, all four pairs must match, including the hashes in
the report's [results.json](results.json). Compare like-for-like phases and
report every measured round, including slow or failed phases.

## 9. Stop only the cluster created by this run

After the benchmark has exited, stop TiKV first, then PD. Keep the data:

```bash
sudo -n docker stop -t 120 \
  "$CONTAINER_PREFIX-tikv1" "$CONTAINER_PREFIX-tikv2" "$CONTAINER_PREFIX-tikv3"
sudo -n docker stop -t 30 \
  "$CONTAINER_PREFIX-pd1" "$CONTAINER_PREFIX-pd2" "$CONTAINER_PREFIX-pd3"
sudo -n docker inspect --format '{{.Name}} running={{.State.Running}} exit={{.State.ExitCode}}' \
  "$CONTAINER_PREFIX-tikv1" "$CONTAINER_PREFIX-tikv2" "$CONTAINER_PREFIX-tikv3" \
  "$CONTAINER_PREFIX-pd1" "$CONTAINER_PREFIX-pd2" "$CONTAINER_PREFIX-pd3" \
  > "$RUN_ROOT/cluster-stopped.txt"
```

To restart these existing containers later, use `docker start` with their exact
names, PD first, then TiKV, and repeat the health checks. Do **not** rerun the
`docker run` blocks over existing names/data. Do not remove containers or data
until the artifacts have been reviewed and intentionally retired.
