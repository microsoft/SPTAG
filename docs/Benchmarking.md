# Benchmark methodology and reproduction

[Documentation index](README.md) | [Evaluation index](../evaluation/README.md) |
[SIFT1M benchmark results](../evaluation/2026-10-28-sift1m/README.md)

This guide defines how to run and report the current SPFresh benchmark harness.
It is not a promise to reproduce historical numbers. The historical reports
have incomplete revision/image provenance and must be interpreted accordingly.

**To set up and run the FILEIO/TiKV experiment, start directly with the
[SIFT1M setup and run guide](../evaluation/2026-10-28-sift1m/Reproduce.md).**
It includes its own dependencies, data download/preparation, build and cluster
instructions; the SDK quick start is not a prerequisite. This page is a
reference for methodology and parameter semantics, not an additional setup
sequence to execute first.

The [SIFT1M report](../evaluation/2026-10-28-sift1m/README.md) describes
complete ten-batch FILEIO and TiKV update runs with `4/4/4`, including their
source/binary differences. It is a single-host evaluation, not a multi-host
or matched-binary backend comparison.
Earlier [partial measurements](../evaluation/2026-10-23-sift1m/README.md) and
older [1M](../evaluation/historical/TiKV-1M.md) and
[10M](../evaluation/historical/TiKV-10M.md) observations are archived separately.

## 1. Pin the complete environment

Record before running:

- SPTAG commit, dirty diff, and `git submodule status --recursive`.
- Compiler, CMake, Boost, optional libraries, ID width, and configure command.
- TiKV source commit or immutable image digest, PD image digest, and all
  cluster configuration files when using TiKV.
- Hardware, resource limits, affinity, network topology, replication factor,
  cache sizes, and storage type.
- Dataset, query set, truth definition, and checksums.

Do not use a personal branch name, `<compatible-tikv-branch>`, or a moving
`nightly` tag as the only build identity in a reproducibility claim. A custom
TiKV fork must be identified by a commit; this repository does not certify an
arbitrary TiKV/PD release pair.

Three processes on one host are **not** a three-host distributed evaluation.
A single-replica performance run is **not** equivalent to three-way replicated
storage. Disclose both topology and replication settings.

## 2. Build the selected backend

Use the [Linux baseline](LinuxInstallation.md) and
[build-option reference](BuildOptions.md). The baseline builds FILEIO,
not TiKV. For TiKV, follow [the integration guide](TiKV-Integration.md)
and build with `TIKV=ON` in a separate build directory.

On Linux, the test binary is `Release/SPTAGTest` relative to the source root,
not `build/Release/SPTAGTest`. Build outputs of the same type share that source
directory, even when multiple CMake build directories exist: rebuild the desired
configuration and record the binary checksum before running.

## 3. Prepare explicit inputs and an isolated work directory

Use absolute paths in the benchmark configuration. Vector/query inputs consumed
by the current generator use SPTAG DEFAULT files: count and dimension headers
followed by vector values, **not** headerless raw coordinates. Check value type,
dimension, row count, ID width, and endianness against the reader.

The harness creates derived `perftest_*` files and can reuse existing generated
files. Use a new work directory for each independent run; otherwise old data or
truth can contaminate the result.

Before running, verify the configured input files really exist. Do not infer
that a successful process used the intended dataset. The generator's fallback
behavior is not an acceptable substitute for a named benchmark corpus.

### Extract a SIFT1B prefix safely

For `bigann_base.u8bin`, a DEFAULT UInt8 file with two little-endian 32-bit
header fields, extract the first 1M rows and **rewrite the row-count header**.
Do not truncate the file while retaining its original billion-row header.
This recipe is for `LARGEVID=OFF`, dimension 128:

```bash
export SIFT_BASE=/absolute/path/to/bigann_base.u8bin
export SIFT_1M=/absolute/path/to/sift-first-1m.u8bin
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
sha256sum "$SIFT_1M"
```

The output must be 128,000,008 bytes. Keep the original input unchanged.
Use the first 200 rows of the query file consistently, and preserve its checksum.

## 4. Configure the current harness

Example FILEIO configuration, saved as `benchmark.ini` in your run directory:

```ini
[Benchmark]
VectorPath=/absolute/path/to/base-and-insert-vectors.bin
QueryPath=/absolute/path/to/queries.bin
TruthPath=perftest_batchtruth.L2.UInt8_990000_128_200_5_1000_0_10
IndexPath=index
ValueType=UInt8
Dimension=128
BaseVectorCount=990000
InsertVectorCount=10000
DeleteVectorCount=0
BatchNum=10
TopK=5
NumSearchThreads=16
NumInsertThreads=4
NumSearchDuringInsertThreads=0
NumQueries=200
DistMethod=L2
Rebuild=true
Resume=-1
Layers=2

[SelectHead]
NumberOfThreads=8

[BuildHead]
NumberOfThreads=8

[BuildSSDIndex]
Storage=FILEIO
NumberOfThreads=8
PostingPageLimit=4
SearchPostingPageLimit=4
BufferLength=4
SearchInternalResultNum=64
LatencyLimit=50
```

Replace the input paths with existing files. The base/insert file must contain
at least `BaseVectorCount + InsertVectorCount` vectors. `InsertVectorCount` is
the **total**, divided across `BatchNum`; choose a divisible total to avoid
integer-truncated batch counts.

With the nonempty `TruthPath` above, the generator computes exact batch truth
if its automatically named file is absent. For this insertion-only example,
phase `p` contains IDs `[0, 990000 + p * 1000)`, for `p=0..10`.
An existing configured truth file overrides the generated filename when loading;
check that its phase layout matches the workload. `TruthPath=none` is appropriate
only for throughput-only runs, never recall validation.

Important section and key distinctions:

| Setting | Correct location / meaning |
| --- | --- |
| `NumSearchThreads` | `[Benchmark]`; `NumThreads` is not the current harness key |
| `NumInsertThreads` | `[Benchmark]`; insertion-driver thread count |
| `NumSearchDuringInsertThreads` | `[Benchmark]`; concurrent search during insertion |
| `Storage`, `TiKVPDAddresses`, `TiKVKeyPrefix` | `[BuildSSDIndex]`, not `[Benchmark]` |
| `NumberOfThreads` | Individual build sections; not the same as query-driver concurrency |
| `TruthPath=none` | Disables ground-truth evaluation; resulting recall values are not valid accuracy evidence |
| `QuantizerFilePath` in `[Benchmark]` | Current harness PQ path; not the RaBitQ CLI training workflow |

For TiKV, replace/add the backend section:

```ini
[BuildSSDIndex]
Storage=TIKVIO
TiKVPDAddresses=127.0.0.1:2379
TiKVKeyPrefix=unique_run_identifier
NumberOfThreads=8
PostingPageLimit=4
SearchPostingPageLimit=4
BufferLength=4
SearchInternalResultNum=64
LatencyLimit=50
```

Use a fresh key prefix **and** a fresh local index/run directory for an
independent run. Do not assume a new local directory clears old TiKV keys.

### Suggested settings for SIFT1M (128-dimensional UInt8)

SIFT1M uses 128-dimensional **UInt8** vectors. The examples above and the
[SIFT1M configuration template](../evaluation/2026-10-28-sift1m/benchmark.ini)
use the following recommended starting settings for new runs on either backend:

```ini
[BuildSSDIndex]
PostingPageLimit=4
SearchPostingPageLimit=4
BufferLength=4
```

Keep `ValueType=UInt8` and `Dimension=128` in `[Benchmark]`; these settings
do not change the dataset representation or phase-specific truth.

The current complete SIFT1M runs used `4/4/4`. The 2026-10-23 partial runs
used `12`, `12`, and `8`, respectively; their historical configurations and
results remain unchanged. Do not mix results across these parameter sets.
Evaluate recall, latency and insertion behavior in a fresh run.

## 5. Run and confirm it actually ran

Run from a dedicated working directory, with `SPTAG_ROOT` set to your absolute
source checkout path:

```bash
mkdir -p benchmark-output/run-001
cd benchmark-output/run-001
test -f benchmark.ini
BENCHMARK_CONFIG="$PWD/benchmark.ini" \
BENCHMARK_OUTPUT="$PWD/output.json" \
  "$SPTAG_ROOT/Release/SPTAGTest" \
  --run_test=SPFreshTest/BenchmarkFromConfig --log_level=message > run.log 2>&1
```

Create/copy the configuration before the `test -f` command. Preserve the exit
code and inspect `run.log` plus the JSON output. The test reports a skip and
returns if `BENCHMARK_CONFIG` is unset; zero exit status alone is not evidence
of a completed benchmark. Verify the logged backend, paths, counts, and phase
outputs. Do not hide failed operations or missing output files.

For TiKV append-only workloads, check the layer-0 `GetNumSamples()` and
contained-ID enumeration against the initial count plus completed insertions,
both before and after saving/reloading. Saved `Base.VectorSize` and metadata
alone do not prove count consistency. Upper-layer counts use global VID bounds
and need not equal their sparse membership sizes.

The insertion path must register a new VID beyond the VersionMap count even
when a missing layer-0 version key is implicitly alive. The
[`InsertedVIDsGrowCountAcrossReloads` regression](../Test/src/TiKVIOTest.cpp)
exercises actual posting insertion, count growth, checkpoint/reload, enumeration,
deleted-VID restoration, and preservation of an existing non-default version.
The bound check applies only to TiKV VersionMaps: a local sparse map counts
members, not the global VID bound. `LocalSparseVIDKeepsExistingVersion` checks
that inserting an existing sparse VID does not reset its version.
Fresh-build initialization must select the actual VersionMap type at runtime:
a TiKV-enabled binary can still use FILEIO and a local VersionMap. Enabling
TiKV at compile time must not compile out local initialization.
`FileIOBuildInitializesLocalVersionsWithTiKVEnabled` builds real one- and
two-layer FILEIO indexes, checks dense bottom and sparse upper membership,
and verifies exact-vector queries before and after saving/reloading.
TiKV's stub pool must also preserve physical connection fan-out: gRPC channels
with identical arguments otherwise share a global subchannel, so multiple
stubs alone do not create independent connections. The pool uses 16 stubs
with local subchannel pools per client and store, independently of maintenance
thread counts, and eight completion queues with one pump thread each. These
transport defaults target concurrent reads; they do not increase search,
insert, or maintenance worker counts. Same-index warm-cache crossover tests
at 32 search threads favored 16 connections over eight and eight completion
queues over four; single-thread latency was slightly higher. These read-only
tests disable query-triggered maintenance and are not a full insertion rerun.
The [full 16-channel/8-CQ experiment](../evaluation/2026-11-05-sift10m-tikv-pool16-cq8/README.md)
also completed a fresh build and all ten insertion batches with count/reload
acceptance. Its standalone reproduction guide uses committed source, pinned
deployment images and exact phase truth; it also records the unresolved
recall decrease between independently rebuilt indexes.
`StubPoolUsesIndependentConnections` verifies the actual
server-side peer count, while `AsyncReadsSurviveClientAddressReuse` checks pool
lifetime across client destruction/recreation.
Run these checks with a TiKV-enabled build:

```bash
./Release/SPTAGTest \
  --run_test=TiKVIOTest --log_level=message
```

`Rebuild=false` does not automatically mean query-only: configured insert/delete
work may still execute. Set operation counts deliberately. Resume behavior
depends on available checkpoint state; validate it with the specific revision
before relying on it after an interrupted experiment.

For an interrupted TiKV run, local checkpoint directories can share a mutable
key prefix. An older local snapshot does not imply older postings in TiKV.
Use the actual saved corpus size and matching truth phase, and do not assume
the checkpoint marker includes the last successfully saved index.
Do not use the update harness as a read-only query tool: it can save indexes
and execute configured mutations even when rebuilding is disabled.

For independent TiKV tuning runs, preserve the baseline configuration and use
a new `IndexPath` and `TiKVKeyPrefix` with `Rebuild=true` and `Resume=-1`.
Keep the same input vectors, queries, and phase-specific truth. Place the truth
under its auto-generated filename in the new working directory as well, so
data preparation does not regenerate it before loading `TruthPath`.
Use `;` for INI comments, not `#`, and write `key=value` without spaces around
the delimiter: the repository parser does not accept `#` comments and retains
leading spaces in string values.

`Benchmark.NumInsertThreads` controls foreground insertion callers.
`BuildSSDIndex.AppendThreadNum` controls each posting layer's shared background
pool for split, merge, reassignment, and asynchronous append work. The harness
defaults the latter to the former; override it explicitly to tune them
independently. `ReassignThreadNum=0` does not disable reassignment in this path.
Editing an INI does not resize an already-running pool.

Set search `MaxCheck` explicitly in `BuildHead` and `BuildSSDIndex` when
comparing exploration budgets. It is distinct from `MaxCheckForRefineGraph`,
which controls graph refinement. Check saved effective settings after building
and reloading. The dynamic-searcher constructor also sets
`SearchPostingPageLimit` to the effective `PostingPageLimit`; do not assume these
two page limits can be tuned independently on this path. Storage-layout changes
such as `UseMultiChunkPosting` require a separate rebuild, not a flag change
against existing single-key postings.

## 6. Measure and report

Use the [report template](../evaluation/ReportTemplate.md). At minimum:

- Keep cold and warmed query rounds separate; define what "cold" means.
- Report all independent runs, repetition count, and variation.
- Report recall with latency/QPS, the exact active corpus per update phase,
  query count, TopK, and search parameters.
- Define latency units, timed region, percentiles, throughput formula, and
  concurrent client count. Do not derive throughput from a mean latency alone
  without accounting for concurrency.
- Record retries, timeouts, failed operations, memory, disk use, build time,
  and update throughput.
- Ground truth must match the same vectors, metric, IDs, and update phase.
  Changing storage backend does not by itself justify changing ground truth.
- Distinguish ID-only recall from the harness's distance-tolerant tie matching.
  Report the denominator; 200 queries at TopK=5 have 1,000 truth neighbors.
- Mark partial runs and log-extracted rows explicitly. Preserve incomplete raw
  JSON unchanged instead of silently presenting it as a successful full run.
- If the current harness cannot fix a random seed, say so and preserve the
  generated inputs/checksums.

Do not use unverified projected latency ranges as expected acceptance results.
The [1M](../evaluation/historical/TiKV-1M.md) and
[10M](../evaluation/historical/TiKV-10M.md) historical
reports are observations under their original conditions, not thresholds.

## 7. Troubleshooting without destroying data

- **Wrong backend:** put backend parameters in `[BuildSSDIndex]`, confirm the
  binary was built with the integration enabled, and inspect startup logs.
- **Unexpected data:** use a clean run directory and verify input headers and
  checksums; the generator caches derived files.
- **Recall missing or zero:** check whether truth was disabled and whether IDs,
  dimensions, corpus phase, and metric match. Do not relabel zero as valid recall.
- **TiKV unavailable:** inspect PD stores/Region routing and service logs;
  successful PD health alone does not prove posting reads/updates work.
- **Memory pressure:** record per-process resource limits and lower concurrency
  or cache sizes; do not silently remove failed runs from a comparison.

The [dated experiments](../evaluation/README.md) preserve prior cluster recipes.
Any format/wipe/rebind operation belongs to a dedicated-machine procedure,
never to the baseline installation or an ordinary benchmark retry.
