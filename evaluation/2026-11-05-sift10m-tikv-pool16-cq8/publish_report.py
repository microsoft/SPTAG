import argparse
import configparser
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil

import deployment
import presplit
import reproduce


RECIPE = Path(__file__).resolve().parent


def measurements(data, rows, insertions):
    results = data["results"]
    reproduce.require(math.isfinite(results["build timeSeconds"]) and results["build timeSeconds"] > 0,
                      "Invalid initial build duration")
    queries = [results["benchmark0_query_before_insert"],
               results["benchmark0b_query_before_insert_round2"]]
    batches = [results["benchmark1_insert"][f"batch_{i}"] for i in range(1, 11)]
    for batch in batches:
        queries.extend([batch["search"], batch["search_round2"]])
    enriched = []
    for row, query in zip(rows, queries):
        record = dict(row)
        for output, key in (("p50_ms", "p50"), ("p90_ms", "p90"),
                            ("min_ms", "minLatency"), ("max_ms", "maxLatency")):
            record[output] = query[key]
        values = [record[key] for key in
                  ("min_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms")]
        reproduce.require(all(math.isfinite(value) for value in values) and
                          values == sorted(values) and values[0] >= 0,
                          "Invalid query latency percentiles")
        reproduce.require(math.isfinite(record["qps"]) and record["qps"] > 0 and
                          math.isfinite(record["mean_ms"]) and
                          record["mean_ms"] > 0 and
                          values[0] <= record["mean_ms"] <= values[-1],
                          "Invalid query performance")
        record["latency_breakdown"] = query["latencyBreakdown"]
        enriched.append(record)
    updates = []
    for row, batch in zip(insertions, batches):
        record = dict(row, barrier_wait_seconds=batch["batch barrier waitSeconds"])
        reproduce.require(all(math.isfinite(record[key]) and record[key] >= 0 for key in
                              ("insert_seconds", "insert_vectors_per_second", "barrier_wait_seconds",
                               "clone_seconds", "load_seconds", "save_seconds")) and
                          record["insert_seconds"] > 0 and record["insert_vectors_per_second"] > 0,
                          "Invalid insertion or checkpoint performance")
        reproduce.require(0 <= record["barrier_wait_seconds"] <= record["insert_seconds"] + 0.001,
                          "Barrier is outside insertion timer")
        expected = record["inserted"] / record["insert_seconds"]
        reproduce.require(math.isclose(expected, record["insert_vectors_per_second"], rel_tol=0.001),
                          "Insertion throughput differs from count / duration")
        updates.append(record)
    seconds = sum(row["insert_seconds"] for row in updates)
    return {
        "query_rounds": enriched,
        "insertions": updates,
        "build_seconds": results["build timeSeconds"],
        "total_inserted": sum(row["inserted"] for row in updates),
        "total_insert_seconds": seconds,
        "aggregate_insert_vectors_per_second": sum(row["inserted"] for row in updates) / seconds,
        "barrier_wait_seconds": sum(row["barrier_wait_seconds"] for row in updates),
        "clone_seconds": sum(row["clone_seconds"] for row in updates),
        "load_seconds": sum(row["load_seconds"] for row in updates),
        "save_seconds": sum(row["save_seconds"] for row in updates),
    }


def read_measurements(root):
    manifest = json.loads((root / "run-manifest.json").read_text())
    storage = manifest.get("storage", "TIKVIO")
    layers = manifest.get("layers", 1)
    saved = configparser.ConfigParser()
    saved.read(root / "saved-indexloader.ini")
    data = json.loads((root / "benchmark-output.json").read_text())
    rows, updates = reproduce.validate_measurements(
        data, (root / "run.log").read_text(), saved, manifest["prefix"], layers, storage)
    return manifest, measurements(data, rows, updates)


def verify_presplit(root, manifest):
    evidence = json.loads((root / "presplit-validation.json").read_text())
    snapshot = json.loads((root / "regions-before-workload.json").read_text())
    reproduce.require(snapshot["count"] == len(snapshot["regions"]), "Incomplete region snapshot")
    stores = {row["store"]["id"] for row in evidence["stores_after"]["stores"]}
    reproduce.require(len(stores) == 3, "Wrong store count")
    lower, upper, keys, samples = presplit.plan(
        manifest["prefix"], manifest.get("layers", 1), 10000000, 128)
    state = presplit.inspect_plan(snapshot["regions"], stores, lower, upper, keys, samples)
    reproduce.require(evidence["status"] == "verified" and
                      evidence["split_keys"] == [key.hex() for key in keys] and
                      not state["missing_boundaries"] and not state["unready_region_ids"] and
                      state["balanced"], "Presplit/scatter acceptance failed")
    for key in ("leader_counts", "groups", "owned_region_ids"):
        reproduce.require(state[key] == evidence[key], "Presplit evidence differs: " + key)
    reproduce.require(all(response["processed-percentage"] == 100
                          for response in evidence["split_responses"]), "Incomplete splits")
    reproduce.require(all(response["processed-percentage"] == 100
                          for response in evidence["scatter_responses"].values()), "Incomplete scatter")
    stopped = json.loads((root / "cluster-stopped.json").read_text())
    expected = {f"/{manifest['prefix']}-{role}{i}" for role in ("pd", "tikv") for i in (1, 2, 3)}
    reproduce.require({row["Name"] for row in stopped} == expected, "Wrong stopped container set")
    for row in stopped:
        state = row["State"]
        role = row["Name"].rsplit("-", 1)[-1]
        image = deployment.PD_IMAGE if role.startswith("pd") else deployment.TIKV_IMAGE
        reproduce.require(not state["Running"] and not state["OOMKilled"] and state["ExitCode"] == 0
                          and row["Config"]["Image"] == image, "Unclean or wrong-image server")


def verify(root):
    for line in (root / "artifacts.sha256").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        path = root / name
        reproduce.require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                          "Artifact path escapes report")
        reproduce.require(reproduce.digest(path) == checksum, "Published artifact changed: " + name)
    manifest, metrics = read_measurements(root)
    reproduce.require(json.loads((root / "results.json").read_text()) == metrics,
                      "Published metrics differ from raw outputs")
    provenance = json.loads((root / "binary-provenance.json").read_text())
    source = json.loads((RECIPE / "publication-provenance.json").read_text())
    reproduce.require(manifest["source_checkout"] == provenance["source_commit"] ==
                      source["published_source_commit"], "Unmeasured source")
    reproduce.require(provenance["binary_sha256"] == manifest["binary_sha256"] ==
                      source["measured_binary_sha256"] and
                      provenance["checker_sha256"] == manifest["count_checker_sha256"] ==
                      source["measured_count_checker_sha256"],
                      "Wrong measured executable")
    reproduce.require(provenance["source_sha256"] == source["source_sha256"],
                      "Source fingerprints differ")
    for name, checksum in manifest["recipe_sha256"].items():
        reproduce.require(reproduce.digest(root / "recorded-recipe" / name) == checksum,
                          "Archived measured recipe changed: " + name)
    inputs = json.loads((root / "input-checksums.json").read_text())
    reproduce.require(inputs["native_base_sha256"] == reproduce.HASHES["input/sift-first-10m.u8bin"] and
                      inputs["native_queries_sha256"] == reproduce.HASHES["input/query.10K.u8bin"] and
                      inputs["phase_truth_sha256"] == reproduce.HASHES["tikv/" + reproduce.TRUTH],
                      "Wrong input identities")
    recorded = configparser.ConfigParser()
    recorded.read(root / "recorded.ini")
    portable = configparser.ConfigParser()
    portable.read_string((root / "benchmark.ini").read_text().replace(
        "@RUN_ROOT@", str(Path(recorded["Benchmark"]["VectorPath"]).parent.parent)).replace(
        "@PREFIX@", manifest["prefix"]))
    reproduce.require(dict(portable) == dict(recorded), "Portable profile differs from recorded INI")
    reproduce.require((root / "benchmark-exit-code.txt").read_text().strip() == "0",
                      "Benchmark did not exit successfully")
    log = (root / "final-count-check.log").read_text()
    reproduce.require("VERIFIED: saved index reload reports 10000000 samples; contained IDs exactly "
                      "[0,10000000); all 100000 inserted metadata entries map bijectively within their batches."
                      in log, "Missing live final enumeration and metadata verification")
    if manifest.get("storage", "TIKVIO") == "TIKVIO":
        reproduce.require(manifest["presplit_regions"] == 128, "Wrong split policy")
        verify_presplit(root, manifest)
    for name, records in (("query-rounds.csv", metrics["query_rounds"]),
                          ("insertions.csv", metrics["insertions"])):
        with (root / name).open() as stream:
            actual = list(csv.DictReader(stream))
        expected = [{key: str(value) for key, value in row.items() if key != "latency_breakdown"}
                    for row in records]
        reproduce.require(actual == expected, "CSV differs from raw results: " + name)
    print("VERIFIED published report:", root.name, "22 rounds, ten batches, final10M evidence")


def table(headers, records):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in records]])


def documents(root, manifest, metrics):
    storage = manifest.get("storage", "TIKVIO")
    layers = manifest.get("layers", 1)
    tikv = storage == "TIKVIO"
    backend = "TiKV" if tikv else "FILEIO"
    shared = Path(os.path.relpath(RECIPE, root)).as_posix()
    final = metrics["query_rounds"][-1]
    resources = (root / "resources.txt").read_text()
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)", resources).group(1)
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", resources).group(1))
    second = [row for row in metrics["query_rounds"] if row["round"] == 2]
    query_table = table(
        ["Batch", "Vectors", "Recall@5", "QPS", "Mean ms", "P50 ms", "P95 ms", "P99 ms"],
        [[row["phase"], row["active_vectors"], f"{row['recall_at_5'] * 100:.2f}%",
          f"{row['qps']:.4f}", f"{row['mean_ms']:.4f}", f"{row['p50_ms']:.3f}",
          f"{row['p95_ms']:.3f}", f"{row['p99_ms']:.3f}"] for row in second])
    insert_table = table(
        ["Batch", "Inserted", "Insert s", "Vectors/s", "Barrier s", "Clone s", "Load s", "Save s"],
        [[row["phase"], row["inserted"], *[f"{row[key]:.4f}" for key in
          ("insert_seconds", "insert_vectors_per_second", "barrier_wait_seconds",
           "clone_seconds", "load_seconds", "save_seconds")]] for row in metrics["insertions"]])
    breakdown = table(
        ["Posting layer", "Read mean ms", "Comp mean ms", "Version mean ms", "Layer total mean ms",
         "Postings/query", "TiKV batch checks/query"],
        [[layer, *[f"{values[key]['mean']:.4f}" for key in
                   ("postingReadMs", "compMs", "versionMapMs", "totalMs")],
          f"{values['postingCountMean']:.4f}", f"{values['versionCheckCountMean']:.4f}"]
         for layer, values in sorted(final["latency_breakdown"]["layers"].items())])
    region_text = "No PD or TiKV processes, network clients or distributed VersionMaps are used."
    if tikv:
        evidence = json.loads((root / "presplit-validation.json").read_text())
        leaders = table(["Key family", "Leaders by store ID"],
                        [[family, ", ".join(f"{store}: {count}" for store, count in counts.items())]
                         for family, counts in evidence["leader_counts"].items()])
        region_text = f"""Three PD and three TiKV containers share this host, with three voting replicas.
Client transport uses 16 independent channels per client/store and 8 completion queues
per client. These constants are compiled into the measured source, not INI tunables.

Before the workload: {len(evidence['split_keys'])} exact split boundaries,
{len(evidence['owned_region_ids'])} owned regions, 128 buckets per active key family.
Posting keys split on the low byte of little-endian DBKey (`0x02,0x04,...,0xfe`);
VersionMap keys split on equal VID ranges in `prefix_v:layer:VID-padded-to-10-digits`.
Layer 1 uses posting/version0 families; layer 2 adds version1. Outer isolation
boundaries account for the extra regions. Store IDs are recorded identities, not
fixed reproduction requirements.

{leaders}

Readiness requires every boundary, three voters, no pending/down owned peers,
no active PD operators, and per-family leader count difference <=26 over three
consecutive observations. Store `Up` alone is insufficient. The retained
[region snapshot](regions-before-workload.json) and [split/scatter evidence](presplit-validation.json)
are independently rechecked by the publication verifier.

The [server TOML](tikv.toml) fixes 16GB memory limit/store, 4GB block cache,
gRPC concurrency 8, apply/store pools 2, RocksDB background jobs 4, and
512MB/384MB maximum/split region sizes. PD ports: 33991-33993 and peers
34001-34003; TiKV ports: 30361-30363 and status 30381-30383.
Immutable server digests and commands are in [deployment](deployment.json)
and the [shared deployment code]({shared}/deployment.py).
This is trusted localhost host-network deployment without RPC authentication.
It is not a multi-host or production-security evaluation.

The TOML requests `reserve-space=1GB`, but each observed placeholder reserved
~71GiB (1% of this ~7TiB filesystem). Capacity/cache settings were not changed.
Only specifically approved reservation files were removed **after** prior
clusters stopped; no databases or accepted results were deleted. There is no
automatic cleanup or equivalent deletion step in the reproduction recipe."""
    version_text = ("Distributed version checks remain enabled on layer0; upper TiKV search "
                    "skips batch checks when layer0-only=true." if tikv else
                    "FILEIO uses local VersionMaps on every posting layer. Local `Deleted` checks "
                    "run inline in the posting scan and are included in computation time. "
                    "The layer0-only switch is TiKV-only. The version timing/count columns "
                    "measure TiKV batch checks, so zero for FILEIO is expected, not a bypass.")
    correction = ""
    if (root / "validation-recovery.json").exists():
        correction = """
The original runner incorrectly required positive TiKV-only counters for FILEIO
and marked validation failed even though the benchmark exited 0. The shared
validator was corrected after inspecting the pinned inline local check, then
all 22 rounds and the independent live final reload were verified.
[Recovery evidence](validation-recovery.json) and the original failed status
are retained. The workload and raw JSON/log were not rerun or modified.
Use the corrected shared runner, not the archived historical validator.
"""
    provenance = json.loads((root / "binary-provenance.json").read_text())
    readme = f"""# SIFT10M {backend}: {layers} posting layer(s), matched measured binary

[Evaluation index](../README.md) | [Reproduce](Reproduce.md) |
[All query rounds](query-rounds.csv) | [Insertions](insertions.csv) |
[Machine-readable results](results.json)

**Status: full workload and independent final10M membership/metadata verified.**
Recorded preparation UTC: `{manifest['created_utc']}`.
This is one independent fresh build, not a same-index read-only retest.

## 1. Provenance and environment

- Measured source: `{provenance['source_commit']}`, already published remotely.
- Executable SHA256: `{provenance['binary_sha256']}`.
- Checker SHA256: `{provenance['checker_sha256']}`.
- This run shares the exact executable, checker, runtime library bytes, native
  inputs and truth with the other three matched November 6/9 reports.
  The older November 3 FILEIO1 run is historical and is not this comparison's baseline.
- Release; FOLLY/TBB/TIKV ON, GPU/SPDK/ROCKSDB/RABITQ/LARGEVID OFF.
  Folly `7088e27f4854c9088bcf35b980e54e89fa966e0f`;
  zstd `e47e674cd09583ff0503f0f6defd6d23d8b718d3`.
- GCC11.4, CMake3.31.12, Protobuf3.12.4, gRPC1.30.2; detailed
  [recorded build environment]({shared}/software-environment.txt).
- Shared AMD EPYC7V12 host, 96 logical CPUs, two sockets/four NUMA nodes,
  ~1.7TiB RAM and ~7TiB SSD filesystem. No fixed affinity/CPU isolation,
  cache flush or build seed; disk throughput and network RTT were not recorded.
- [Binary/library provenance](binary-provenance.json), [input hashes](input-checksums.json),
  [original manifest](run-manifest.json), [recorded recipe](recorded-recipe/)
  and [artifact hashes](artifacts.sha256) identify the actual measurement.
  Historical absolute paths are evidence only, not reproduction prerequisites.
  The historical Python recipe was an uncommitted snapshot on the original
  publishing checkout; its archived files/hashes, not `recipe_checkout` alone,
  identify it. The measured C++ source remains the fixed committed revision.

## 2. Workload and configuration

| Setting | Value |
| --- | --- |
| Runtime backend / posting layers | {storage} / {layers}, plus final in-memory BKT head |
| Dataset / metric | First10M SIFT, UInt8,128D, squared L2, no quantizer |
| Initial / insert / delete | 9,900,000 / ten10,000 batches / 0 |
| Queries / TopK | 10,000 native SIFT queries / 5 |
| Passes | One excluded warmup before each measured pass; 22 measured passes |
| Build threads | SelectHead80 / BuildHead80 / BuildSSDIndex80 |
| Insert / maintenance / query threads | 16 / 48 / 32 |
| Search during insertion | 0; sequential query and foreground-insert phases |
| MaxCheck | 8192 in BuildHead and BuildSSDIndex |
| Posting/Search pages / BufferLength | 4 / 4 / 4 |
| SearchInternalResultNum / LatencyLimit | 64 / 50 |
| Fresh run | Rebuild=true, Resume=-1 |
| Version policy | {version_text} |

[Portable configuration](benchmark.ini) fixes this profile's storage/layers.
[Recorded input INI](recorded.ini) and [saved effective INI](saved-indexloader.ini)
retain exact measured paths/settings. [Raw JSON](benchmark-output.json) and
[raw log](run.log) are byte-preserved.

## 3. Deployment

{region_text}

## 4. Measurement definitions

- Query latency is the elapsed `SearchIndex` call per query, in milliseconds.
  QPS is `10000 / query-pass wall seconds` across32 worker threads, including
  worker creation/join; result serialization and recall computation are outside
  this timer. It is not `1000 / mean latency`.
- Per-query percentiles sort10000 latencies and select zero-based
  `floor(N*p)` (P50=5000, P95=9500, P99=9900), without interpolation.
  Head traversal, setup and posting-layer work contribute to total latency;
  marginal component percentiles must not be added as if correlated.
- Every phase uses squared-L2 exact Top5 over that phase's active prefix.
  Ground-truth ties sort by source row ID. Concurrent inserted VIDs can differ
  from source IDs; metadata maps them back. Recall uses native10000 queries
  and five neighbors per query. Truth layout/hashes are in [reproduction](Reproduce.md).
- Each insert timer covers `InsertVectors`, including the background-work
  completion barrier. Per-batch vectors/s =10000/insert seconds. Aggregate
  vectors/s =100000/sum(insert seconds), **not** an average of batch rates.
  Build, query/warmup, clone, load and save are excluded. Therefore insertion
  throughput is not end-to-end pipeline throughput or concurrent-update QPS.
- Warmups and checkpoint work still contribute to process wall time.
  Historical cold/warm round labels do not establish a controlled cold-cache
  measurement: the harness executes a warmup before each measured pass.
  RSS in [resources](resources.txt) measures the benchmark client and excludes
  separately running PD/TiKV servers. No claim about server total RSS is made.

## 5. Results

| Metric | Value |
| --- | --- |
| Initial build seconds | {metrics['build_seconds']:.4f} |
| Benchmark process wall time | {wall} |
| Benchmark maximum RSS KiB | {rss:,} |
| Final10M round2 Recall@5 | {final['recall_at_5'] * 100:.2f}% |
| Final10M round2 QPS | {final['qps']:.4f} |
| Final10M round2 mean / P95 / P99 ms | {final['mean_ms']:.4f} / {final['p95_ms']:.3f} / {final['p99_ms']:.3f} |
| 100K insertion timer total seconds | {metrics['total_insert_seconds']:.4f} |
| Aggregate insertion vectors/s | {metrics['aggregate_insert_vectors_per_second']:.4f} |
| Included barrier wait seconds | {metrics['barrier_wait_seconds']:.4f} |
| Separate clone / load / save totals seconds | {metrics['clone_seconds']:.4f} / {metrics['load_seconds']:.4f} / {metrics['save_seconds']:.4f} |

### Queries: second measured pass at every phase

Round2 is selected consistently, not the faster pass. All22 passes, including
P50/P90/P95/P99/min/max and all per-layer breakdowns, remain in the CSV/JSON.

{query_table}

### Final10M round2 posting-layer latency breakdown

{breakdown}

{version_text}

### All ten insertion batches

{insert_table}

## 6. Acceptance and reproducibility limits

All22 measured rounds, ten ordered batches, exact final `[0,10000000)`
membership and100K inserted-metadata batch bijections passed.
[Independent live reload log](final-count-check.log) validates logical membership
and metadata, **not** every physical posting payload. Upper sparse VersionMap
counts need not equal10M. No new recall, latency or throughput release threshold
was imposed. TiKV's six owned containers stopped exit0 without OOM.
{correction}
Published: source/tools, full configuration, raw aggregate JSON/log, metrics,
provenance and acceptance evidence. Not shipped: executables/shared libraries,
SIFT vectors, truth payload, index snapshots or TiKV databases.
The [reproduction guide](Reproduce.md) builds fixed remotely available source
and creates a new workload without the original host paths or local patch.
An isolated source build and six focused regressions were previously verified;
the full measured runs reuse the archived binary. A clean-host dependency install
and a full10M rerun using a newly rebuilt binary have **not** been verified.
Reproduction preserves controls and acceptance, not exact binary hashes,
random heads, recall or timings. The publication verifier checks retained
evidence offline; it cannot independently enumerate an unshipped database.

The four profiles each have one build, on a shared host at different times;
two query passes are not independent builds. Do not attribute a backend/layer
difference solely to one setting, or infer that pre-splitting accelerates query
performance: the unsplit TiKV2 attempt was aborted and is not an accepted control.
Known pre-existing DedupCandidateTest failures remain outside focused coverage.
"""
    if not tikv:
        readme = readme.replace("TiKV's six owned containers stopped exit0 without OOM.",
                               "No TiKV containers were constructed or started.")
    (root / "README.md").write_text(readme)
    pulls = ""
    if tikv:
        pulls = f"""sudo -n docker pull {deployment.PD_IMAGE}
sudo -n docker pull {deployment.TIKV_IMAGE}
"""
    reproduction = f"""# Reproduce SIFT10M {backend}, posting layers={layers}

[Measured report](README.md) | [Shared source/build guide]({shared}/Reproduce.md)

## 1. Fixed source and dependencies

Follow sections1-2 of the shared guide to install dependencies and create **two**
checkouts: one containing this publishing commit's reports/tools, and one detached
at measured source `{provenance['source_commit']}`. Pin the publishing commit
returned with this report; do not silently use a moving branch or source tip.
Build Release SPTAGTest with FOLLY/TBB/TIKV ON even for FILEIO, regenerate protobuf,
build pinned Folly/zstd and build the included truth/count helpers.
Use CMake3.31.12 and the recorded toolchain for the closest source-level reproduction.
The shared guide records exact commands, package versions and known test limitations.
{"Docker access is required for this TiKV profile." if tikv else "Docker is not required or invoked for this FILEIO profile; skip Docker setup and image pulls."}

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

- Base SHA256: `{reproduce.HASHES['input/sift-first-10m.u8bin']}`
- Query SHA256: `{reproduce.HASHES['input/query.10K.u8bin']}`
- Truth SHA256: `{reproduce.HASHES['tikv/' + reproduce.TRUTH]}`

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
export RUN_ROOT="$SPTAG_ROOT/benchmark-output/{backend.lower()}-layer{layers}-$RUN_TAG"
python3 "$RECIPE/reproduce.py" prepare --run-root "$RUN_ROOT" \\
  --source-root "$SPTAG_ROOT" --storage {storage} --layers {layers} \\
  --presplit-regions {128 if tikv else 0} \\
  --base "$SIFT_DATA/bigann_base.u8bin" --queries "$SIFT_DATA/query.10K.u8bin" \\
  --truth-generator "$HELPERS/exact_truth" --truth-threads 32 \\
  --binary "$SPTAG_ROOT/Release/SPTAGTest" --checker "$HELPERS/check_count" \\
  --prefix "{backend.lower()}-l{layers}-$RUN_TAG"
{pulls}python3 "$RECIPE/reproduce.py" run --run-root "$RUN_ROOT"
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

{"The runner starts three PD/three TiKV nodes from pinned digests, checks replica3, splits128 buckets per active family and scatters before workload. Require three caught-up voters/no pending or down peers, every boundary, balanced family leaders and no active operators across three observations. The shared deployment and presplit code are the exact runnable implementation. Capacity/memory/cache options must not be changed when claiming this profile." if tikv else "The runner neither constructs a Cluster nor starts PD/TiKV. Local VersionMaps check Deleted inline on all posting layers. Do not interpret zero TiKV-only version counters as disabled FILEIO checks."}

## 4. Acceptance, metrics and optional report export

Expected `status.json`: `phase=complete`; benchmark exit0;22 measured rounds;
ten10K insertions; final reload enumerates exactly10M IDs and100K metadata entries.
Outputs: `{reproduce.directory_name(storage)}/output.json`, `run.log`, `resources.txt`,
`query-rounds.csv`, `insertions.csv`, `summary.json`, `final-count-check.log`
{"and `cluster-stopped.json`." if tikv else "and no cluster directory."}
Offline `validate` checks the retained live-count success, not a new database scan.
Keep original inputs/config/binaries/recipe hashes and raw files.

To regenerate the same report format from a successfully verified matched-binary run:

```bash
python3 "$RECIPE/publish_report.py" publish --run-root "$RUN_ROOT" \\
  --report-root /absolute/path/to/new-report-directory
python3 "$RECIPE/publish_report.py" verify \\
  --report-root /absolute/path/to/new-report-directory
```

The exporter requires the archived measured executable hash; a locally rebuilt
binary is allowed by the benchmark runner but is a **new** provenance, not this
matched-binary publication. Do not relabel it or weaken the exporter check.
To verify the committed report without datasets, executables, Docker or local run paths:

```bash
python3 "$RECIPE/publish_report.py" verify \\
  --report-root "$REPORT_ROOT/evaluation/{root.name}"
```

This recomputes metrics and checks all22/ten shapes, effective configuration,
artifact hashes and prior live-count evidence{" plus split/replica/leader/clean-shutdown evidence" if tikv else ""}.
Use report sections4-5 for latency/QPS/insert throughput definitions and all batches.
These measurements are observations, not promised reproduction thresholds.
"""
    (root / "Reproduce.md").write_text(reproduction)


def publish(run, root):
    reproduce.require(not root.exists(), "Refusing to overwrite an existing published report")
    manifest = json.loads((run / "run-manifest.json").read_text())
    storage = manifest.get("storage", "TIKVIO")
    directory = run / reproduce.directory_name(storage)
    source = json.loads((RECIPE / "publication-provenance.json").read_text())
    reproduce.require(manifest["source_checkout"] == source["published_source_commit"],
                      "Unmeasured source checkout")
    for name, checksum in json.loads((run / "checksums.json").read_text()).items():
        reproduce.require(reproduce.digest(run / name) == checksum, "Frozen artifact changed: " + name)
    for name, expected in (("bin/SPTAGTest", source["measured_binary_sha256"]),
                           ("bin/check_count", source["measured_count_checker_sha256"])):
        reproduce.require(reproduce.digest(run / name) == expected, "Wrong matched executable: " + name)
    reproduce.require((directory / "index/checkpoint.txt").read_text().strip() == "9",
                      "Wrong final checkpoint")
    status = json.loads((run / "status.json").read_text())
    if status["phase"] != "complete":
        recovery = json.loads((run / "validation-recovery.json").read_text())
        reproduce.require(storage == "FILEIO" and
                          recovery["status"] == "benchmark_and_counts_verified_after_validator_correction",
                          "Unaccepted workload status")
    root.mkdir(parents=True)
    for name, original in {
        "run-manifest.json": run / "run-manifest.json",
        "original-run-status.json": run / "status.json",
        "recorded-checksums.json": run / "checksums.json",
        "recorded.ini": directory / "benchmark.ini",
        "saved-indexloader.ini": directory / "index_9/indexloader.ini",
        "benchmark-output.json": directory / "output.json",
        "run.log": directory / "run.log",
        "resources.txt": directory / "resources.txt",
        "benchmark-exit-code.txt": directory / "exit-code.txt",
        "final-count-check.log": run / "final-count-check.log",
        "runtime-ldd.txt": run / "runtime-ldd.txt",
        "focused-tests.log": run / "focused-tests.log",
    }.items():
        shutil.copy2(original, root / name)
    for name in ("validation-recovery.json", "recovery-validator-tests.log", "recipe-tests.log",
                 "presplit-validation.json", "regions-before-workload.json",
                 "cluster-stopped.json", "reservation-cleanup.json"):
        if (run / name).exists():
            shutil.copy2(run / name, root / name)
    recorded = root / "recorded-recipe"
    recorded.mkdir()
    recipe = Path(manifest["recipe_root"])
    for name in ("reproduce.py", "deployment.py", "presplit.py", "benchmark.ini",
                 "tikv.toml", "publication-provenance.json", "source-commit.txt"):
        shutil.copy2(recipe / name, recorded / name)
    if storage == "TIKVIO":
        shutil.copy2(run / "tikv.toml", root / "tikv.toml")
        reproduce.write_json(root / "deployment.json", {
            "pd_image": deployment.PD_IMAGE, "tikv_image": deployment.TIKV_IMAGE,
            "pd_nodes": 3, "tikv_nodes": 3, "replicas": 3,
            "host_network": True, "pd_addresses": deployment.ENDPOINTS,
            "ports": deployment.PORTS, "buckets_per_active_key_family": 128,
            "client_channels_per_store": 16, "completion_queues_per_client": 8,
            "commands": "../2026-11-05-sift10m-tikv-pool16-cq8/deployment.py",
        })
    libraries = {path.name: reproduce.digest(path) for path in (run / "lib").iterdir() if path.is_file()}
    reproduce.write_json(root / "binary-provenance.json", {
        "source_commit": manifest["source_checkout"], "source_sha256": source["source_sha256"],
        "binary_sha256": reproduce.digest(run / "bin/SPTAGTest"),
        "checker_sha256": reproduce.digest(run / "bin/check_count"),
        "runtime_library_sha256": libraries, "dependencies": source["dependencies"],
        "newly_rebuilt_binary_measured": False,
    })
    reproduce.write_json(root / "input-checksums.json", {
        "base_shape": [10000000, 128], "query_shape": [10000, 128],
        "native_base_sha256": reproduce.HASHES["input/sift-first-10m.u8bin"],
        "native_queries_sha256": reproduce.HASHES["input/query.10K.u8bin"],
        "phase_truth_sha256": reproduce.HASHES["tikv/" + reproduce.TRUTH],
    })
    template = (RECIPE / "benchmark.ini").read_text()
    template = template.replace("@LAYERS@", str(manifest.get("layers", 1)))
    template = template.replace("@STORAGE@", storage).replace("@WORKDIR@", reproduce.directory_name(storage))
    template = template.replace("@DISTRIBUTED_VERSION_MAP@", str(storage == "TIKVIO").lower())
    if storage == "FILEIO":
        template = "".join(line for line in template.splitlines(keepends=True)
                           if not line.startswith(("TiKVPDAddresses=", "TiKVKeyPrefix=")))
    (root / "benchmark.ini").write_text(template)
    manifest, metrics = read_measurements(root)
    reproduce.write_json(root / "results.json", metrics)
    for name, rows in (("query-rounds.csv", metrics["query_rounds"]),
                       ("insertions.csv", metrics["insertions"])):
        records = [{key: value for key, value in row.items() if key != "latency_breakdown"} for row in rows]
        with (root / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    documents(root, manifest, metrics)
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    (root / "artifacts.sha256").write_text(
        "".join(f"{reproduce.digest(path)}  {path.relative_to(root).as_posix()}\n" for path in paths))
    verify(root)


def main():
    parser = argparse.ArgumentParser(description="Publish or verify an independently measured SIFT10M profile.")
    parser.add_argument("command", choices=("publish", "verify"))
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    if args.command == "publish":
        if args.run_root is None:
            parser.error("publish requires --run-root")
        publish(args.run_root.resolve(), args.report_root.resolve())
    else:
        verify(args.report_root.resolve())


if __name__ == "__main__":
    main()
