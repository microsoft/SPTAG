# Normalized SIFT: RaBitQ4 L2/IP and OPQ64 IP

[Evaluation index](../README.md) | [Measured report](Report.md)

RaBitQ4 L2 and repaired IP completed 1M-5M with zero missing current source IDs.
IP's recall decline was 0.52 percentage points, versus L2's 0.58.
Repaired OPQ64 IP reached 95.79% at 1M; its 2M-5M stages were not run.
All measured performance, recall and before/after findings are in [Report.md](Report.md).

## Reproduction

The commands reproduce the workload and core implementations, not byte-identical
executables, randomized indexes or exact timings. They require the original
frozen model files as external inputs, but no existing checkpoint or archived
build manifest. Model binaries are not committed. All generated source,
builds, caches, data and runs stay under a **new, owned `/data` directory**.
There is one Python entry point and two native helpers: normalization/exact
truth and read-only current-ID checks. No unit-test or diagnostic-script
package is needed to reproduce the native comparison.

## Prerequisites and workspace

The measured host used Ubuntu 24.04, GCC 13.3, CMake 3.28.3, Python 3.12,
Boost 1.83 and TBB, with CPU execution only. Python 3.11 or newer is required.
Use the existing dependencies on a shared machine; install missing system
packages only with the machine owner's approval:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3 curl time lsof \
  libboost-system-dev libboost-thread-dev libboost-serialization-dev \
  libboost-regex-dev libboost-filesystem-dev libboost-test-dev \
  libtbb-dev libeigen3-dev libaio-dev libnuma-dev libzstd-dev
```

Do not repartition storage or delete unrelated data. The runners enforce a
250GiB free-space floor. Budget instructions below add a separate 50GiB
contingency; the full comparison needs substantially more free space than
the final index sizes alone.

```bash
set -euo pipefail
WORK=/data/your-owned-directory/new-normalized-sift-eval
test ! -e "$WORK"
mkdir -p "$WORK"/{tmp,cache,input,runs}
export TMPDIR="$WORK/tmp" TMP="$WORK/tmp" TEMP="$WORK/tmp"
export XDG_CACHE_HOME="$WORK/cache" PYTHONPYCACHEPREFIX="$WORK/cache/python"
SRC="$WORK/SPTAG"
git clone --branch users/zhangt/rabitq_ipd https://github.com/microsoft/SPTAG.git "$SRC"
TOOLS="$SRC/evaluation/2026-11-18-sift10m-rabitq-ip/tools"
MODELS=/data/path/to/frozen-normalized-sift-models
test -r "$MODELS/rabitq4.bin"
test -r "$MODELS/opq64.bin"
findmnt -T "$WORK" -o TARGET,SOURCE,FSTYPE
df -h "$WORK"
```

Set `MODELS` to the directory containing the two frozen model artifacts from
the original experiment. Their required SHA-256 values are listed in
[Report.md](Report.md#reproducibility-and-provenance). They are separate inputs,
like the dataset; neither Git nor the `prepare` command supplies them.

## 1. Build the three measured core variants

The reproduction script makes independent source clones because CMake writes `Release`
inside the **source tree**, not just the build directory. It initializes only
the required pinned zstd and RaBitQ submodules, copies this package's benchmark
harness, and builds the library, `SPTAGTest` and native evaluation helpers.

| Case | Core commit | Implementation |
| --- | --- | --- |
| `rabitq4-l2` | `aac5e14` | Original L2 maintenance |
| `rabitq4-ip` | `9480ddd` | IP-specific head maintenance |
| `opq64-ip` | `145255a` | Head repair plus L2-nearest PQ/OPQ encoding |

All 204 AnnService files are verified against the baseline, allowing exactly
the two head headers or those headers plus `PQQuantizer.h`. Full commit IDs
are pinned in the script. No deleted temporary regression fixture is required.

```bash
for case in rabitq4-l2 rabitq4-ip opq64-ip; do
  python3 "$TOOLS/reproduce.py" build \
    --case "$case" --root "$WORK/build-$case" --jobs 8
done
```

Build flags are Release, `FOLLY=OFF`, `TBB=ON`, `RABITQ=ON`, and
`SPDK/ROCKSDB/TIKV/GPU=OFF`. Each build records commands, source/harness and
binary hashes, compiler versions, submodules and linked libraries in its
`build-manifest.json`. A different compiler/build location can change binary
hashes; the newly generated manifest, not a copied historical one, is used
by the runner.

## 2. Download only the native SIFT10M prefix and original 10K queries

The public base object contains 1B rows. The range request downloads only its
first 10M-row payload; the extraction step rewrites the count header.
HTTP 206 and the exact byte limit are required to avoid downloading the full
128GB object.

```bash
code=$(curl --fail --location --show-error --max-time 1800 \
  --range 0-1280000007 --max-filesize 1280000008 \
  --output "$WORK/input/base-prefix.download" --write-out '%{http_code}' \
  https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin)
test "$code" = 206
test "$(stat -c %s "$WORK/input/base-prefix.download")" = 1280000008
curl --fail --location --show-error --max-filesize 1280008 \
  --output "$WORK/input/query.10K.u8bin" \
  https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin
python3 "$TOOLS/reproduce.py" extract \
  --prefix "$WORK/input/base-prefix.download" \
  --output "$WORK/input/sift-first-10m.u8bin"
```

These are native UInt8 files with `(uint32 rows, uint32 dimension)` headers,
not renamed bvecs. Only the first 5M rows are used in the evaluation.
The raw/normalized input and truth hashes are pinned in
[reproduce.py](tools/reproduce.py), with no separate reference JSON.

## 3. Normalize and compute exact truth

```bash
python3 "$TOOLS/reproduce.py" prepare \
  --root "$WORK/prepared" \
  --base "$WORK/input/sift-first-10m.u8bin" \
  --query "$WORK/input/query.10K.u8bin" \
  --helper "$WORK/build-rabitq4-ip/bin/normalized_prepare"
```

This normalizes the base/query using FP64 accumulation/division and stores
Float32. Source order and zero-row IDs are preserved. It creates all five
prefixes and computes **independent squared-L2 and `1-dot` truths** from the
stored Float32 values, with FP64 accumulation and source-ID tie breaking.
All ten truth hashes and all normalized input hashes must match the recorded
references. Exact truth preparation is expensive and is excluded from ANN
build/query timings.

The externally supplied `rabitq4.bin` and `opq64.bin` are frozen
initial-1M-trained models, shared across every prefix. Pass the appropriate
file with `run --model`; its hash is checked before creating a run, and a
private copy is kept with that run. A differently trained model is rejected.
No growth-stage retraining, raw-vector reranking or reconstruction
renormalization occurs.

## 4. Review the space budget and run

Run cases **sequentially**, without overlapping builds, insertions or queries
from another reproduction. Other users' shared-host activity may still vary.

The budget command takes your explicit peak *additional* work allowance
(checkpoints, private predecessor copies, encoded/generated caches and metadata)
and adds 50GiB contingency above the 250GiB floor. It checks current free space,
but it is **not an automatic storage estimator or permission to reclaim data**.

The 300GiB work allowance below is a conservative example for each RaBitQ
growth run, not a measured worst-case bound. It requires at least 600GiB free
at that case's launch; account for the other case's retained outputs as well.
Review or change the allowance for your machine rather than lowering the floor.
The measured repaired IP 1M/5M checkpoints used approximately 11.1/91.4GiB;
copies and temporary caches make peak usage larger.

```bash
for case in rabitq4-l2 rabitq4-ip; do
  python3 "$TOOLS/reproduce.py" budget \
    --case "$case" --root "$WORK/runs/$case" \
    --output "$WORK/$case-budget.json" --work-gib 300 \
    --stop-after-prefix 5 --retain-checkpoints initial-and-latest
  python3 -u "$TOOLS/reproduce.py" run \
    --case "$case" --root "$WORK/runs/$case" \
    --build "$WORK/build-$case" --prepared "$WORK/prepared" \
    --model "$MODELS/rabitq4.bin" \
    --budget "$WORK/$case-budget.json" \
    --stop-after-prefix 5 --retain-checkpoints initial-and-latest
done
```

`initial-and-latest` prunes only this new run's verified obsolete checkpoints
and generated input caches, after a successful successor, save/reload and
current-ID audit, with hash/ownership/open-handle checks. Initial/latest
checkpoints, all failed stages and all result evidence remain. Use `all`
instead if retaining every milestone, and review a correspondingly larger budget.

Reproduce the **measured first-1M OPQ case**, not an unmeasured 5M result.
Its recorded allowance was 64GiB checkpoint + 4GiB caches + 1GiB metadata,
plus the separate 50GiB contingency:

```bash
python3 "$TOOLS/reproduce.py" budget \
  --case opq64-ip --root "$WORK/runs/opq64-ip" \
  --output "$WORK/opq64-ip-budget.json" --work-gib 69 \
  --stop-after-prefix 1 --retain-checkpoints all
python3 -u "$TOOLS/reproduce.py" run \
  --case opq64-ip --root "$WORK/runs/opq64-ip" \
  --build "$WORK/build-opq64-ip" --prepared "$WORK/prepared" \
  --model "$MODELS/opq64.bin" \
  --budget "$WORK/opq64-ip-budget.json" \
  --stop-after-prefix 1 --retain-checkpoints all
```

Each milestone prints its recall/coverage result after primary query rounds,
save, fresh-process reload and physical audit. The manifest retains all five
planned stages; the OPQ example leaves 2M-5M explicitly `not_run`.
The runner records failures and preserves the scene rather than reporting
missing measurements as zero. IP-repaired variants stop on failed integrity.
Refill or pending maintenance without useful progress is bounded at 3,600s;
job-completion counters alone do not reset the guard.

```bash
python3 "$TOOLS/reproduce.py" report \
  --run "rabitq4-l2=$WORK/runs/rabitq4-l2" \
  --run "rabitq4-ip=$WORK/runs/rabitq4-ip" \
  --run "opq64-ip=$WORK/runs/opq64-ip" \
  --output "$WORK/Report.md"
```

The report contains primary round-2 recall/QPS, build/insertion and included
drain times, latency, fresh-reload results and current-ID coverage.
Unrun measurements are `N/A`, never zero recall or zero missing IDs.
Run-local JSON manifests and logs support the lifecycle and audits, but are
not committed as result files; the measured results are summarized in
[Report.md](Report.md).
