import argparse
import configparser
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import time

from deployment import Cluster
from presplit import prepare_cluster


RECIPE = Path(__file__).resolve().parent
GIB = 1024 ** 3
TRUTH = "perftest_batchtruth.L2.UInt8_9900000_128_10000_5_10000_0_10"
HASHES = {
    "input/sift-first-10m.u8bin": "3f92342f582ebe3aa50751be105791e11894c4caac5b9d3321999457c56b054d",
    "input/query.10K.u8bin": "eca755831fc9a8004e14886df48b81109a8ed3bfc6b632509c93c0460a30d552",
    "tikv/" + TRUTH: "2cfad2347ac180705e701d43e9fd5a46cbc9e4cd4bd2695b8fe4959dfc287c1b",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def directory_name(storage):
    return {"TIKVIO": "tikv", "FILEIO": "fileio"}[storage]


def extract_vectors(source, output, count=10000000, dimension=128):
    with source.open("rb") as stream, output.open("xb") as target:
        rows, dim = struct.unpack("<II", stream.read(8))
        require(dim == dimension and rows >= count and
                source.stat().st_size == 8 + rows * dim, "Unexpected source vector shape/length")
        target.write(struct.pack("<II", count, dimension))
        remaining = count * dimension
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            require(bool(block), "Source ended before requested vector count")
            target.write(block)
            remaining -= len(block)


def prepare(args):
    require(args.presplit_regions == 0 or 3 <= args.presplit_regions <= 256,
            "Presplit requires 3..256 buckets, or 0 to disable")
    require(args.storage == "TIKVIO" or args.presplit_regions == 0, "Presplit is TiKV-only")
    require(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.prefix), "Use a fresh lowercase prefix, at most 64 characters")
    root = args.run_root.resolve()
    workdir = directory_name(args.storage)
    hashes = {name: value for name, value in HASHES.items() if not name.startswith("tikv/")}
    hashes[workdir + "/" + TRUTH] = HASHES["tikv/" + TRUTH]
    provenance = json.loads((RECIPE / "publication-provenance.json").read_text())
    source_root = args.source_root.resolve()
    source_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    require(source_commit == provenance["published_source_commit"],
            "Build from the fixed measured source commit; pass its checkout with --source-root")
    for name, checksum in provenance["source_sha256"].items():
        require(digest(source_root / name) == checksum,
                "Checkout source differs from measured implementation: " + name)
    subprocess.run(["git", "-C", str(source_root), "diff", "--exit-code", "HEAD", "--",
                    "AnnService", "Test", "CMakeLists.txt", "ThirdParty/kvproto/proto"],
                   check=True, stdout=subprocess.DEVNULL)
    root.mkdir(parents=True, exist_ok=False)
    for name in ("input", workdir, "bin"):
        (root / name).mkdir()
    extract_vectors(args.base.resolve(), root / "input/sift-first-10m.u8bin")
    for source, target in ((args.queries, root / "input/query.10K.u8bin"),
                           (args.binary, root / "bin/SPTAGTest"),
                           (args.checker, root / "bin/check_count")):
        shutil.copy2(source.resolve(), target)
    for name, checksum in HASHES.items():
        if name.startswith("tikv/"):
            continue
        require(digest(root / name) == checksum, "Input/truth checksum mismatch: " + name)
    if args.truth is not None:
        shutil.copy2(args.truth.resolve(), root / workdir / TRUTH)
    else:
        require(args.truth_threads > 0, "Truth generation requires positive thread count")
        subprocess.run([str(args.truth_generator.resolve()), str(root / "input/sift-first-10m.u8bin"),
                        str(root / "input/query.10K.u8bin"), "9900000", "10000", "10",
                        str(args.truth_threads), str(root / workdir / TRUTH)], check=True)
    require(digest(root / workdir / TRUTH) == HASHES["tikv/" + TRUTH], "Exact truth checksum mismatch")
    with (root / "focused-tests.log").open("x") as stream:
        subprocess.run([str(root / "bin/SPTAGTest"), "--run_test=TiKVIOTest,TestDataGeneratorTest",
                        "--log_level=message"], stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180)
    ldd = subprocess.check_output(["ldd", str(root / "bin/SPTAGTest")], text=True)
    require("not found" not in ldd, "Missing runtime library; inspect ldd")
    (root / "runtime-ldd.txt").write_text(ldd)
    template = (RECIPE / "benchmark.ini").read_text()
    rendered = template.replace("@RUN_ROOT@", str(root)).replace("@PREFIX@", args.prefix)
    rendered = rendered.replace("@LAYERS@", str(args.layers))
    rendered = rendered.replace("@WORKDIR@", workdir).replace("@STORAGE@", args.storage)
    rendered = rendered.replace("@DISTRIBUTED_VERSION_MAP@", str(args.storage == "TIKVIO").lower())
    if args.storage == "FILEIO":
        rendered = "".join(line for line in rendered.splitlines(keepends=True)
                           if not line.startswith(("TiKVPDAddresses=", "TiKVKeyPrefix=")))
    require("@" not in rendered, "Unresolved configuration placeholder")
    (root / workdir / "benchmark.ini").write_text(rendered)
    shutil.copy2(RECIPE / "tikv.toml", root / "tikv.toml")
    recipe_commit = subprocess.check_output(
        ["git", "-C", str(RECIPE), "rev-parse", "HEAD"], text=True).strip()
    files = [*hashes, "bin/SPTAGTest", "bin/check_count", workdir + "/benchmark.ini", "tikv.toml",
             "focused-tests.log", "runtime-ldd.txt"]
    write_json(root / "checksums.json", {name: digest(root / name) for name in files})
    write_json(root / "run-manifest.json", {
        "prefix": args.prefix, "source_checkout": source_commit,
        "layers": args.layers,
        "storage": args.storage,
        "presplit_regions": args.presplit_regions,
        "recipe_checkout": recipe_commit,
        "source_root": str(source_root), "recipe_root": str(RECIPE),
        "recipe_source_commit": (RECIPE / "source-commit.txt").read_text().strip(),
        "recipe_sha256": {name: digest(RECIPE / name) for name in
                          ("reproduce.py", "deployment.py", "presplit.py", "benchmark.ini", "tikv.toml")},
        "binary_sha256": digest(root / "bin/SPTAGTest"),
        "count_checker_sha256": digest(root / "bin/check_count"),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print("Prepared exact inputs and fresh configuration:", root, flush=True)


def validate_measurements(data, log, saved, prefix, layers=1, storage="TIKVIO"):
    directory_name(storage)
    tikv = storage == "TIKVIO"
    for key, value in {
        "ValueType": "UInt8", "dimension": 128, "baseVectorCount": 9900000,
        "insertVectorCount": 100000, "DeleteVectorCount": 0, "BatchNum": 10,
        "numQueries": 10000, "topK": 5, "numSearchThreads": 32, "numInsertThreads": 16,
        "layers": layers, "DistMethod": "L2",
    }.items():
        require(data["config"][key] == value, "Wrong output configuration: " + key)
    runtime_markers = (("SPANNIndex:UseTiKV", "Using distributed TiKV VersionMap (layer=0")
                       if tikv else ("SPANNIndex:UseFileIO", "Using local in-memory VersionMap (layer=0"))
    for marker in (*runtime_markers, "initialize p_localToGlobal(0) and versionMap (9900000)"):
        require(marker in log, "Missing runtime evidence: " + marker)
    forbidden = (("SPANNIndex:UseFileIO",) if tikv else
                 ("SPANNIndex:UseTiKV", "Using distributed TiKV VersionMap"))
    for marker in (*forbidden, "Using random generation!", "AddIndex failed", "giving up",
                   "read postings fail", "BatchGetVersions: MultiGet failed"):
        require(marker not in log, "Benchmark failure marker: " + marker)
    results = data["results"]
    batches = results["benchmark1_insert"]
    require(set(batches) == {f"batch_{i}" for i in range(1, 11)}, "Incomplete insert batches")
    queries = [(0, 1, results["benchmark0_query_before_insert"]),
               (0, 2, results["benchmark0b_query_before_insert_round2"])]
    insertions = []
    for phase in range(1, 11):
        batch = batches[f"batch_{phase}"]
        require(batch["inserted"] == 10000 and batch["Load vectorCount"] == 9900000 + (phase - 1) * 10000,
                "Wrong inserted/reloaded count")
        require(batch["insert timeSeconds"] > 0, "Insertion time must be positive")
        queries.extend([(phase, 1, batch["search"]), (phase, 2, batch["search_round2"])])
        insertions.append({"phase": phase, "inserted": batch["inserted"],
                           "insert_seconds": batch["insert timeSeconds"],
                           "insert_vectors_per_second": batch["insert throughput"],
                           "load_seconds": batch["Load timeSeconds"],
                           "clone_seconds": batch["Clone timeSeconds"],
                           "save_seconds": batch["save timeSeconds"]})
    counts = re.findall(r"Checkpoint layer stats: batch (\d+)/10 layer=0 samples=(\d+) deleted=(\d+) headMapping=(\d+)", log)
    require([int(row[0]) for row in counts] == list(range(1, 11)), "Missing checkpoint counts")
    for phase, count, deleted, mapping in counts:
        require(int(count) == int(mapping) == 9900000 + int(phase) * 10000 and int(deleted) == 0,
                "Checkpoint logical membership mismatch")
    if layers == 2:
        marker = ("Using distributed TiKV VersionMap (layer=1" if tikv else
                  "Using local in-memory VersionMap (layer=1")
        require(marker in log, "Missing upper-layer VersionMap")
        upper_counts = re.findall(
            r"Checkpoint layer stats: batch (\d+)/10 layer=1 samples=(\d+) deleted=(\d+) headMapping=(\d+)",
            log)
        require([int(row[0]) for row in upper_counts] == list(range(1, 11)),
                "Missing upper-layer checkpoint counts")
    metadata = [int(value) for value in re.findall(r"Save MetaIndex\((\d+)\)", log)]
    require(metadata and metadata[-1] == 10000000 and
            set(metadata) == {9900000 + i * 10000 for i in range(11)}, "Metadata phase counts differ")
    require(saved.getint("Base", "VectorSize") == 10000000, "Wrong final saved vector count")
    parameters = {"Storage": storage, "Layers": str(layers), "AppendThreadNum": "48", "MaxCheck": "8192",
                  "PostingPageLimit": "4", "SearchPostingPageLimit": "4", "BufferLength": "4"}
    if tikv:
        parameters["TiKVKeyPrefix"] = prefix
    for key, expected in parameters.items():
        require(saved.get("BuildSSDIndex", key) == expected, "Wrong saved parameter: " + key)
    require(saved.getboolean("BuildSSDIndex", "DistributedVersionMap") == tikv and
            saved.getboolean("BuildSSDIndex", "SearchCheckVersionMapOnlyLayer0"),
            "Version checking configuration changed")
    for section in ("SelectHead", "BuildHead", "BuildSSDIndex"):
        require(saved.getint(section, "NumberOfThreads") == 80, "Wrong saved build threads")
    require(saved.getint("BuildHead", "MaxCheck") == 8192, "Wrong saved head exploration budget")
    rows = []
    for phase, round_id, query in queries:
        recall = query["recall"]
        require(query["numQueries"] == recall["numQueries"] == 10000 and recall["k"] == 5,
                "Wrong query/recall shape")
        require(0 <= recall["recallAtK"] <= 1, "Invalid recall value")
        breakdown = query["latencyBreakdown"]["layers"]
        require(set(breakdown) == {str(i) for i in range(layers)}, "Wrong measured posting-layer breakdown")
        if not tikv:
            # Pinned FILEIO checks Deleted inline; these counters measure TiKV batch checks only.
            require(all(layer["postingCountMean"] > 0 and layer["listElementsMean"] > 0 and
                        layer["versionCheckCountMean"] == 0 and layer["versionMapMs"]["mean"] == 0
                        for layer in breakdown.values()), "Unexpected local FILEIO layer telemetry")
        else:
            require(breakdown["0"]["versionCheckCountMean"] > 0, "Layer-0 version checks were disabled")
            if layers == 2:
                require(breakdown["1"]["postingCountMean"] > 0 and
                        breakdown["1"]["versionCheckCountMean"] == 0 and
                        breakdown["1"]["versionMapMs"]["mean"] == 0,
                        "Upper-layer search did not retain layer-0-only version checking")
        rows.append({"phase": phase, "round": round_id, "active_vectors": 9900000 + phase * 10000,
                     "recall_at_5": recall["recallAtK"], "qps": query["qps"],
                     "mean_ms": query["meanLatency"], "p95_ms": query["p95"], "p99_ms": query["p99"]})
    return rows, insertions


def validate(root, live=False):
    manifest = json.loads((root / "run-manifest.json").read_text())
    storage = manifest.get("storage", "TIKVIO")
    directory = root / directory_name(storage)
    saved = configparser.ConfigParser()
    saved.read(directory / "index_9/indexloader.ini")
    require((directory / "index/checkpoint.txt").read_text().strip() == "9", "Wrong checkpoint marker")
    data = json.loads((directory / "output.json").read_text())
    layers = manifest.get("layers", 1)
    rows, insertions = validate_measurements(
        data, (directory / "run.log").read_text(), saved, manifest["prefix"], layers=layers, storage=storage)
    if live:
        with (root / "final-count-check.log").open("x") as stream:
            subprocess.run([str(root / "bin/check_count"), str(directory / "index_9")],
                           cwd=directory, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=900)
    require("VERIFIED: saved index reload reports 10000000 samples; contained IDs exactly "
            in (root / "final-count-check.log").read_text(), "Missing independent final count-check success")
    for name, records in (("query-rounds.csv", rows), ("insertions.csv", insertions)):
        with (root / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    write_json(root / "summary.json", {
        "status": "benchmark_and_counts_verified", "query_rounds": rows, "insertions": insertions,
        "source_commit": manifest["recipe_source_commit"], "storage": storage,
        "connections_per_client_per_store": 16 if storage == "TIKVIO" else None,
        "completion_queues_per_client": 8 if storage == "TIKVIO" else None, "final_reloaded_samples": 10000000,
        "posting_layers": layers,
        "contained_ids": 10000000, "inserted_metadata_bijection": 100000,
        "count_validation_scope": "Logical membership and metadata, not exhaustive physical payload scan",
    })
    print("VERIFIED: 22 rounds, ten counts, final reload/enumeration and inserted metadata", flush=True)


def interrupted(signum, frame):
    raise InterruptedError("Reproduction received signal " + str(signum))


def run(root):
    manifest = json.loads((root / "run-manifest.json").read_text())
    storage = manifest.get("storage", "TIKVIO")
    directory = root / directory_name(storage)
    require(storage == "TIKVIO" or not manifest.get("presplit_regions", 0), "Presplit is TiKV-only")
    for name, checksum in json.loads((root / "checksums.json").read_text()).items():
        require(digest(root / name) == checksum, "Frozen input/binary/config changed: " + name)
    for name, checksum in manifest["recipe_sha256"].items():
        require(digest(RECIPE / name) == checksum, "Recipe changed after preparation: " + name)
    require(shutil.disk_usage(root).free >= 500 * GIB, "Need at least 500 GiB free before experiment startup")
    require(not (root / "cluster").exists() and not (directory / "run.log").exists(),
            "Refusing to reuse a previous run")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    cluster = Cluster(root, manifest["prefix"]) if storage == "TIKVIO" else None
    try:
        if cluster is not None:
            cluster.start()
        if manifest.get("presplit_regions", 0):
            prepare_cluster(root, manifest["prefix"], manifest.get("layers", 1),
                            10000000, manifest["presplit_regions"])
        env = dict(os.environ, BENCHMARK_CONFIG=str(directory / "benchmark.ini"),
                   BENCHMARK_OUTPUT=str(directory / "output.json"))
        with (directory / "run.log").open("x") as stream:
            process = subprocess.Popen(
                ["/usr/bin/time", "-v", "-o", str(directory / "resources.txt"),
                 str(root / "bin/SPTAGTest"), "--run_test=SPFreshTest/BenchmarkFromConfig", "--log_level=message"],
                cwd=directory, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            started = time.monotonic()
            try:
                while process.poll() is None:
                    write_json(root / "status.json", {"phase": "benchmark", "runner_pid": os.getpid(),
                                                     "benchmark_process_group": process.pid,
                                                     "elapsed_seconds": time.monotonic() - started})
                    require(shutil.disk_usage(root).free >= 250 * GIB, "Disk safety stop below 250 GiB")
                    require(time.monotonic() - started < 24 * 3600, "24-hour benchmark safety stop")
                    time.sleep(15)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                (directory / "exit-code.txt").write_text(str(process.returncode) + "\n")
        require(process.returncode == 0, "Benchmark failed; inspect " + str(directory / "run.log"))
        write_json(root / "status.json", {"phase": "validating"})
        validate(root, live=True)
    except BaseException as error:
        write_json(root / "status.json", {"phase": "failed", "error": repr(error)})
        raise
    finally:
        if cluster is not None:
            cluster.close()
    write_json(root / "status.json", {"phase": "complete"})


def main():
    parser = argparse.ArgumentParser(description="Run the full SIFT10M FILEIO or TiKV experiment.")
    parser.add_argument("command", choices=("prepare", "run", "validate"))
    parser.add_argument("--run-root", type=Path, required=True)
    for name in ("base", "queries", "truth", "binary", "checker"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--prefix")
    parser.add_argument("--source-root", type=Path, default=RECIPE.parent.parent)
    parser.add_argument("--layers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--storage", choices=("TIKVIO", "FILEIO"), default="TIKVIO")
    parser.add_argument("--presplit-regions", type=int, default=0)
    parser.add_argument("--truth-generator", type=Path)
    parser.add_argument("--truth-threads", type=int, default=32)
    args = parser.parse_args()
    if args.command == "prepare":
        if any(getattr(args, name) is None for name in ("base", "queries", "binary", "checker", "prefix")):
            parser.error("prepare requires --base --queries --binary --checker --prefix")
        if (args.truth is None) == (args.truth_generator is None):
            parser.error("prepare requires exactly one of --truth or --truth-generator")
        prepare(args)
    elif args.command == "run":
        run(args.run_root.resolve())
    else:
        validate(args.run_root.resolve())


if __name__ == "__main__":
    main()
