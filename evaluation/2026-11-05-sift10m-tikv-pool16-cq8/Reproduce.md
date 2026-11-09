# Reproduce the SIFT10M TiKV 16/8 experiment

[Results and limitations](README.md) | [Portable runner](reproduce.py)

This is a new single-host Linux experiment: fresh 9.9M build, ten 10K inserts,
22 measured native-query rounds and independent final 10M count checking.
It requires no original binary, index, database, patch or prior FILEIO run.
It preserves the workload/deployment, not exact performance or recall.

## 1. Host and checkout

Use an x86-64 AVX2 host with SSD storage, Docker Engine and sufficient RAM
for three 16GB-configured TiKV stores plus build/client/PD memory. The recorded
machine had 96 logical CPUs, AMD EPYC 7V12, and approximately 1.7TiB RAM;
these are context, not measured minimum requirements.
The runner requires at least 500GiB **free before cluster startup** and stops
the benchmark below 250GiB free. Provision substantially more headroom on a
shared host: server reservation, checkpoints and unrelated workloads consume space.
Keep all generated data outside existing experiments.
For `--storage FILEIO`, Docker and server memory are not required: skip Docker
setup and image pulls. The client still uses the same TIKV-enabled build and
its gRPC/Folly runtime dependencies. The 500GiB/250GiB disk guards apply to both
backends.

The tested environment used Ubuntu 22.04-era packages, GCC 11.4.0,
CMake 3.31.12, Protobuf 3.12.4, gRPC 1.30.2 and Folly 0.58.0-dev.
Exact installed package versions are in [software-environment.txt](software-environment.txt).
Do not claim a different package set is byte-identical.

Install missing dependencies on your own host, not blindly on a shared server:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git pkg-config \
  libboost-all-dev libtbb-dev libnuma-dev protobuf-compiler libprotobuf-dev \
  libgrpc++-dev protobuf-compiler-grpc libabsl-dev libgoogle-glog-dev \
  libgflags-dev libevent-dev libdouble-conversion-dev libssl-dev libunwind-dev \
  libfmt-dev python3
```

Install Docker Engine separately and confirm `sudo -n docker info` works.
For unattended runs, obtain appropriate Docker access; cached sudo credentials
can expire during a long benchmark. The wrapper deliberately uses `sudo -n`.
Use CMake 3.31.12 for the recorded setup; the distro CMake package may be older.
The pinned Folly build can fetch dependencies from its pinned manifests if
they are absent, so its first build may require internet access.

Clone this report's branch, then pin the **report publishing commit** for your
reproduction rather than relying on a moving branch tip. Unrelated Git LFS
payloads are not needed. Explicit filter passthrough avoids optional LFS setup:

```bash
git -c filter.lfs.process= -c filter.lfs.smudge=cat -c filter.lfs.required=false \
  clone --branch users/qiazh/pre-merge-tikv-bugfix \
  https://github.com/microsoft/SPTAG.git SPTAG-repro
cd SPTAG-repro
export REPORT_ROOT="$(pwd -P)"
export RECIPE="$REPORT_ROOT/evaluation/2026-11-05-sift10m-tikv-pool16-cq8"
git merge-base --is-ancestor "$(cat "$RECIPE/source-commit.txt")" HEAD
export SPTAG_ROOT="$REPORT_ROOT/benchmark-output/tikv16-measured-source"
mkdir -p "$REPORT_ROOT/benchmark-output"
git -c filter.lfs.process= -c filter.lfs.smudge=cat -c filter.lfs.required=false \
  clone --no-checkout "$REPORT_ROOT" "$SPTAG_ROOT"
git -C "$SPTAG_ROOT" -c filter.lfs.process= -c filter.lfs.smudge=cat \
  -c filter.lfs.required=false checkout --detach "$(cat "$RECIPE/source-commit.txt")"
cd "$SPTAG_ROOT"
git submodule update --init ThirdParty/zstd ThirdParty/folly
test "$(git -C ThirdParty/folly rev-parse HEAD)" = 7088e27f4854c9088bcf35b980e54e89fa966e0f
test "$(git -C ThirdParty/zstd rev-parse HEAD)" = e47e674cd09583ff0503f0f6defd6d23d8b718d3
```

Run subsequent commands from this checkout in the same shell and stop on
any error. Do not switch build variants in its shared `Release` directory.
The branch tip also preserves a separately developed build-memory change.
It is **not** the source of this measurement. The second, fixed checkout above
is mandatory: build there, keep the recipe in `REPORT_ROOT`, and pass
`--source-root "$SPTAG_ROOT"`. The runner checks the exact source commit,
production/test hashes, and absence of application source edits before preparation.

## 2. Build the client, matching protobuf stubs and helpers

```bash
cmake -S ThirdParty/folly -B ThirdParty/folly/_build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF
cmake --build ThirdParty/folly/_build --parallel 4
export LD_LIBRARY_PATH="$SPTAG_ROOT/ThirdParty/folly/_build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

bash ThirdParty/kvproto/generate_cpp.sh
cmake -S . -B build-repro -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DFOLLY=ON -DTBB=ON -DTIKV=ON \
  -DSPDK=OFF -DROCKSDB=OFF -DGPU=OFF -DRABITQ=OFF -DLARGEVID=OFF
cmake --build build-repro --target SPTAGTest --parallel 4
./Release/SPTAGTest --run_test=TiKVIOTest,TestDataGeneratorTest --log_level=message

export HELPERS="$SPTAG_ROOT/benchmark-output/tikv16-helpers"
python3 "$RECIPE/build_helpers.py" --build-dir "$SPTAG_ROOT/build-repro" --output "$HELPERS"
(cd "$RECIPE" && EXACT_TRUTH_BIN="$HELPERS/exact_truth" python3 -m unittest -v test_reproduce)
```

Regenerate stubs for the installed compiler/runtime instead of consuming
an unshipped local tar archive. This changes generated files in your checkout;
retain it as local build provenance, not another source fix.
Build parallelism 4 here is compiler concurrency, unrelated to benchmark
build threads 80. Keep the checkout and Folly library available while running;
the wrapper freezes executables but does not vendor all shared libraries.
Known pre-existing `DedupCandidateTest` failures are not part of the six focused
cases and were not fixed by this work.
The separately published layer profiles use the same shared build workflow.
Their own `Reproduce.md` files supply mandatory storage/layers/presplit settings;
do not use this older report's default unsplit TiKV1 command for those profiles.
Run both tool suites when validating the extended publication tools:

```bash
(cd "$RECIPE" && EXACT_TRUTH_BIN="$HELPERS/exact_truth" \
  python3 -m unittest -v test_reproduce test_publish_report)
```

## 3. Prepare native inputs and phase-specific exact truth

Provide the UInt8 SIFT1B files `bigann_base.u8bin` and `query.10K.u8bin`.
The original location was `/mnt_ssd/data/sift1b`; it is not required.
See the [existing SIFT input acquisition instructions](../2026-10-28-sift1m/Reproduce.md#3-prepare-the-actual-sift-inputs)
if needed, but extract **10M**, not that guide's 1M.
These files are two little-endian uint32 header fields `(rows, 128)` followed
by contiguous UInt8 vectors, not bvecs/fvecs. A matching already-cut 10M base
file is also accepted. Inputs are never synthesized.

```bash
export SIFT_DATA=/absolute/path/to/sift1b
export RUN_TAG="$(date -u +%Y%m%dt%H%M%Sz)"
export RUN_ROOT="$SPTAG_ROOT/benchmark-output/tikv16-full-$RUN_TAG"
python3 "$RECIPE/reproduce.py" prepare --run-root "$RUN_ROOT" \
  --source-root "$SPTAG_ROOT" \
  --base "$SIFT_DATA/bigann_base.u8bin" \
  --queries "$SIFT_DATA/query.10K.u8bin" \
  --truth-generator "$HELPERS/exact_truth" --truth-threads 32 \
  --binary "$SPTAG_ROOT/Release/SPTAGTest" --checker "$HELPERS/check_count" \
  --prefix "sptag16-$RUN_TAG"
```

Use a unique lowercase prefix, such as `sptag16-20261106t030000z`.
The runner rejects invalid/reused run directories rather than falling back.
The default is `--layers 1`, matching this report. To conduct a separate
two-layer comparison, pass `--layers 2` at preparation with a new run directory
and prefix. All other workload and transport settings remain unchanged.
Validation then requires both layer breakdowns and checkpoint records, with
version checks enabled on layer 0 and skipped on layer 1 during search.
For a separate deployment optimization experiment, also pass
`--presplit-regions 128`. Before building, the runner pre-splits the posting
raw-byte keyspace and each layer's zero-padded VersionMap VID range, then
scatters only this run's regions. It verifies all boundaries, three caught-up
voting replicas without pending/down peers, and a leader-count difference
of at most 20% of the bucket count in each key family, with no pending
operators over three consecutive observations. Failure aborts the run rather
than silently benchmarking an unbalanced cluster. This is not the deployment
used by the original layer-1 report; pre-splitting is disabled by default.
For a FILEIO comparison, use `--storage FILEIO --layers 2` and leave
`--presplit-regions` at zero. The same executable is used with local VersionMaps,
no PD/TiKV containers are started, and outputs reside in `fileio/`.
FILEIO retains version checks on both posting layers: the layer-0-only switch
applies to distributed TiKV queries, not local FILEIO.
In the pinned source, FILEIO performs local `Deleted` checks inline in the
posting scan, included in computation time. `versionCheckCountMean` and
`versionMapMs` record TiKV batch checks only and remain zero for FILEIO;
zero is not evidence that local checks were bypassed. FILEIO validation uses
the pinned source, local-map runtime markers on each layer, saved configuration,
nonempty per-layer scans, checkpoint membership and independent final reload
instead of requiring positive TiKV-only counters.

Preparation extracts exactly the first 10M base rows, hashes the base/query,
generates all 11 exact truth phases, requires the recorded truth SHA256,
renders a new INI with `key=value` and semicolon comments, freezes both
executables, runs six focused tests and captures input/config/binary hashes.
Exact truth generation is substantial CPU work, additional to the benchmark.
It scans the 9.9M base once per query and merges each newly inserted 10K batch.
Equal distances are ordered by source row ID.

Alternatively replace `--truth-generator ... --truth-threads 32` with
`--truth /absolute/path/to/verified-phase-truth`. The file must have SHA256
`2cfad2347ac180705e701d43e9fd5a46cbc9e4cd4bd2695b8fe4959dfc287c1b`;
no original local directory is required. The exact truth format is:
header `(220000, 5)`, then `11*10000*5` int32 IDs, then the same number of
float32 squared distances, total 4,400,008 bytes.
Do not substitute SIFT1B's original ground truth for phase-specific first-10M truth.

## 4. Deploy and run

Fetch the exact server images:

```bash
sudo -n docker pull pingcap/pd@sha256:2a79836fcfe4948103b9eed9f69cc664208c3e6d5bae5e12d0c01e29c3d1598e
sudo -n docker pull pingcap/tikv@sha256:b6893e84e1914cff9fd262daac479ab24ef07ad99b15fd2b33457d02cb9ee38a
python3 "$RECIPE/reproduce.py" run --run-root "$RUN_ROOT"
```

For FILEIO, run only the Python command: no images, Docker daemon or cluster
are used. Its workload files reside in `fileio/`, rather than `tikv/`.
Use a persistent terminal/session manager for the long foreground command.
The wrapper does not daemonize itself. Monitor `status.json` and
`tikv/run.log` under the new run root.

`deployment.py` reproduces the recorded container commands. It requires twelve
free localhost ports: PD clients 33991-33993, PD peers 34001-34003, TiKV
30361-30363 and status 30381-30383. It creates a new 3-PD/3-TiKV cluster,
checks PD health, all three stores Up and replication 3, then starts the
benchmark. It uses host networking, localhost advertised addresses and
unauthenticated local RPCs: this is a trusted-host benchmark, not a secure
production deployment. Do not expose it on an untrusted host.

No existing container/database is deleted or restarted. Name collisions fail.
Only containers created by this invocation, with its matching ownership label,
are stopped, TiKV before PD. Data and stopped containers remain for inspection;
no automatic recursive cleanup is performed. SIGINT/SIGTERM trigger cleanup;
SIGKILL, host failure or loss of Docker privileges can prevent it.
If necessary, inspect the six names and their `sptag.run` labels and stop
those exact names manually. Never use global Docker prune or process-name kills.

The server TOML is unchanged from the measurement: 16GB per-store configured
memory limit, 4GB block cache, server gRPC concurrency 8, three replicas.
Client 16 channels/store and 8 CQ are different settings and compiled into
the committed source. Each client/store can establish up to 16 connections;
multiple client instances multiply that resource use.

## 5. Acceptance and interpretation

After benchmark exit 0, the runner verifies runtime TIKVIO/distributed
VersionMap, native inputs, all ten batches and counts, saved effective settings,
22 query/recall outputs, then invokes the included independent count checker
while the cluster is still available. It verifies exact `[0,10M)` membership
and all 100K inserted metadata entries before stopping the cluster.
Normal query-triggered maintenance and version checks remain enabled.
For FILEIO, acceptance instead requires local VersionMap runtime markers on
each posting layer and the pinned inline local-check implementation. Its TiKV
batch-check metrics remain zero. The same final10M membership and100K metadata
checker runs without any TiKV service.

Expected outputs include `tikv/output.json`, `tikv/run.log`, `tikv/resources.txt`,
`query-rounds.csv`, `insertions.csv`, `summary.json`, `final-count-check.log`,
`cluster-stopped.json`, and `status.json` with `phase=complete`.
Re-check retained files offline without restarting the cluster:

```bash
python3 "$RECIPE/reproduce.py" validate --run-root "$RUN_ROOT"
```

Offline validation checks the prior count-check success log; it does not perform
another live enumeration of a stopped database. It is not a substitute for the
live check performed by `run`. Preserve the original outputs and hashes.

Compare all 22 rounds, build/insert timings, P95/P99, and recall, not just QPS.
The observed final round was 1852.3547 QPS, mean 17.2516ms, P99 24.543ms,
Recall@5 95.53%. These are observations, not exact acceptance targets.
The preceding full 8/4 run had 95.74% recall; this decrease is unresolved.
No fixed head-selection seed, CPU isolation, or cache flush was imposed.
Do not mask a recall change by declaring count acceptance to be quality equivalence.
