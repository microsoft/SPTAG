# Reproduce SIFT10M TiKV, posting layers=2

[Measured report](README.md) | [Shared source/build guide](../2026-11-05-sift10m-tikv-pool16-cq8/Reproduce.md)

## 1. Fixed source and dependencies

Follow sections1-2 of the shared guide to install dependencies and create **two**
checkouts: one containing this publishing commit's reports/tools, and one detached
at measured source `95bcb20f48f7864072f9443ec91027875382b722`. Pin the publishing commit
returned with this report; do not silently use a moving branch or source tip.
Build Release SPTAGTest with FOLLY/TBB/TIKV ON even for FILEIO, regenerate protobuf,
build pinned Folly/zstd and build the included truth/count helpers.
Use CMake3.31.12 and the recorded toolchain for the closest source-level reproduction.
The shared guide records exact commands, package versions and known test limitations.
Docker access is required for this TiKV profile.

After the build, keep the guide's `REPORT_ROOT`, `SPTAG_ROOT`, `RECIPE`, `HELPERS`
and `LD_LIBRARY_PATH` variables in the same shell. `RECIPE` is the shared
`evaluation/2026-11-05-sift10m-tikv-pool16-cq8` tools directory, **not**
this report or `recorded-recipe`. Runtime libraries must remain available.
The archive records the original run only; the corrected shared runner is the
supported execution path.

## 2. Real input and exact truth

Acquire/convert the SIFT input as described in section3 of the shared guide.
Set `SIFT_DATA` to your own data directory. Native UInt8 files have two little-endian
uint32 header fields `(rows,128)` followed by row-major bytes; do not rename bvecs.
Prepare extracts exactly first10M, checks base/query hashes and generates exact
Top5 for initial9.9M plus ten10K prefixes. The truth header is `(220000,5)`,
followed by11x10000x5 int32 source IDs, then equally many float32 squared distances.
The22-round header counts two passes/phase; only11 truth blocks are stored.
Tie order is distance then source row ID.

- Base SHA256: `3f92342f582ebe3aa50751be105791e11894c4caac5b9d3321999457c56b054d`
- Query SHA256: `eca755831fc9a8004e14886df48b81109a8ed3bfc6b632509c93c0460a30d552`
- Truth SHA256: `2cfad2347ac180705e701d43e9fd5a46cbc9e4cd4bd2695b8fe4959dfc287c1b`

Instead of generation, provide `--truth /absolute/path/to/verified-phase-truth`;
it must match the same hash. No local truth archive or original experiment is required.
Truth generation is a first-time cost outside the benchmark timer.

## 3. Prepare this exact profile and run

Use a fresh unique output directory/prefix. At least500GiB must be free before
startup; the runtime stops below250GiB and after24hours. TiKV needs extra
headroom for reservations/replicas; do not delete someone else's data.

```bash
set -euo pipefail
export SIFT_DATA=/absolute/path/to/sift1b
export RUN_TAG="$(date -u +%Y%m%dt%H%M%Sz)"
export RUN_ROOT="$SPTAG_ROOT/benchmark-output/tikv-layer2-$RUN_TAG"
python3 "$RECIPE/reproduce.py" prepare --run-root "$RUN_ROOT" \
  --source-root "$SPTAG_ROOT" --storage TIKVIO --layers 2 \
  --presplit-regions 128 \
  --base "$SIFT_DATA/bigann_base.u8bin" --queries "$SIFT_DATA/query.10K.u8bin" \
  --truth-generator "$HELPERS/exact_truth" --truth-threads 32 \
  --binary "$SPTAG_ROOT/Release/SPTAGTest" --checker "$HELPERS/check_count" \
  --prefix "tikv-l2-$RUN_TAG"
sudo -n docker pull pingcap/pd@sha256:2a79836fcfe4948103b9eed9f69cc664208c3e6d5bae5e12d0c01e29c3d1598e
sudo -n docker pull pingcap/tikv@sha256:b6893e84e1914cff9fd262daac479ab24ef07ad99b15fd2b33457d02cb9ee38a
python3 "$RECIPE/reproduce.py" run --run-root "$RUN_ROOT"
python3 "$RECIPE/reproduce.py" validate --run-root "$RUN_ROOT"
```

The [profile INI](benchmark.ini) contains the same settings rendered by this
command. CLI storage/layers/presplit are essential: defaults alone reproduce the
older unsplit TiKV1 profile, not this measurement.
Compare your generated INI with the profile after substituting run paths/prefix.
No manual deployment, prefix pre-split command or index reuse is needed.
Run the long command in a persistent terminal/session manager and do not start
another TiKV profile until the previous owned cluster stops (ports are shared).
No cleanup is automatic; stopped containers/data are preserved.

The runner starts three PD/three TiKV nodes from pinned digests, checks replica3, splits128 buckets per active family and scatters before workload. Require three caught-up voters/no pending or down peers, every boundary, balanced family leaders and no active operators across three observations. The shared deployment and presplit code are the exact runnable implementation. Capacity/memory/cache options must not be changed when claiming this profile.

## 4. Acceptance, metrics and optional report export

Expected `status.json`: `phase=complete`; benchmark exit0;22 measured rounds;
ten10K insertions; final reload enumerates exactly10M IDs and100K metadata entries.
Outputs: `tikv/output.json`, `run.log`, `resources.txt`,
`query-rounds.csv`, `insertions.csv`, `summary.json`, `final-count-check.log`
and `cluster-stopped.json`.
Offline `validate` checks the retained live-count success, not a new database scan.
Keep original inputs/config/binaries/recipe hashes and raw files.

To regenerate the same report format from a successfully verified matched-binary run:

```bash
python3 "$RECIPE/publish_report.py" publish --run-root "$RUN_ROOT" \
  --report-root /absolute/path/to/new-report-directory
python3 "$RECIPE/publish_report.py" verify \
  --report-root /absolute/path/to/new-report-directory
```

The exporter requires the archived measured executable hash; a locally rebuilt
binary is allowed by the benchmark runner but is a **new** provenance, not this
matched-binary publication. Do not relabel it or weaken the exporter check.
To verify the committed report without datasets, executables, Docker or local run paths:

```bash
python3 "$RECIPE/publish_report.py" verify \
  --report-root "$REPORT_ROOT/evaluation/2026-11-06-sift10m-tikv-layer2-presplit128"
```

This recomputes metrics and checks all22/ten shapes, effective configuration,
artifact hashes and prior live-count evidence plus split/replica/leader/clean-shutdown evidence.
Use report sections4-5 for latency/QPS/insert throughput definitions and all batches.
These measurements are observations, not promised reproduction thresholds.
