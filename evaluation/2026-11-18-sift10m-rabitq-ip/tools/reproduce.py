"""Reproduce normalized SIFT 1M-5M RaBitQ/OPQ comparisons on Linux."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import configparser
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import tarfile
import time


SOURCE = Path(__file__).resolve().parents[3]


RECIPE = Path(__file__).resolve().parent


PROFILE = RECIPE.parent / "benchmark.ini"


RAW_BASE_SHA = "3f92342f582ebe3aa50751be105791e11894c4caac5b9d3321999457c56b054d"


RAW_QUERY_SHA = "eca755831fc9a8004e14886df48b81109a8ed3bfc6b632509c93c0460a30d552"


SEARCH_K = 20


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def extract(source, target, count, truncated=False):
    with source.open("rb") as stream, target.open("xb") as output:
        rows, dim = struct.unpack("<II", stream.read(8))
        require(dim == 128 and rows >= count, "Invalid native SIFT header")
        require(source.stat().st_size == 8 + (count if truncated else rows) * dim,
                "Unexpected input length")
        output.write(struct.pack("<II", count, dim))
        remaining = count * dim
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            require(bool(block), "Premature input EOF")
            output.write(block)
            remaining -= len(block)


def execute_process(command, cwd, name, env=None, timeout=86400):
    require(shutil.disk_usage(cwd).free >= 250 * 1024**3, "Less than 250GiB free disk")
    started = time.monotonic()
    with (cwd / (name + ".log")).open("x") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                require(time.monotonic() - started < timeout, "Process exceeded time guard")
                require(shutil.disk_usage(cwd).free >= 250 * 1024**3, "Disk guard reached")
                write_json(cwd / (name + "-status.json"),
                           {"status": "running", "pid": process.pid,
                            "elapsed_seconds": time.monotonic() - started, "command": command})
                time.sleep(5)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            write_json(cwd / (name + "-status.json"), {"status": "interrupted", "returncode": process.returncode})
            raise
    write_json(cwd / (name + "-status.json"),
               {"status": "exited", "returncode": process.returncode,
                "elapsed_seconds": time.monotonic() - started, "command": command})
    require(process.returncode == 0, f"{name} exited {process.returncode}; see {cwd / (name + '.log')}")


def model_info(path):
    data = path.read_bytes()
    offset = data.index(struct.pack("<I", 0x32464252))
    require(struct.unpack_from("<IIiiiI", data, offset)[1:] == (6, 128, 128, 4, 0),
                   "Expected a new global version6 RaBitQ4 model")
    return {"sha256": sha256_file(path), "version": 6, "code_bytes": 88, "training": "all normalized initial rows"}


def stage_zero_insert_fixture(directory, count, queries, truth, metric="InnerProduct"):
    # The retained generator defaults a=0 to N and unconditionally prepares an addset.
    # Its cache permits an explicit empty dataset for this strictly zero-insertion run.
    add = directory / f"perftest_addvector.bin.Float_{count}_128"
    meta = directory / f"perftest_addmeta.bin.{count}_{count}"
    offsets = directory / f"perftest_addmetaidx.bin.{count}_{count}"
    cached_truth = directory / f"perftest_batchtruth.{metric}.Float_{count}_128_{queries}_5_0_0_0"
    for path in (add, meta, offsets, cached_truth):
        require(not path.exists(), "Never replace an existing harness fixture")
    add.write_bytes(struct.pack("<ii", 0, 128))
    meta.write_bytes(b"")
    offsets.write_bytes(struct.pack("<iQ", 0, 0))
    shutil.copy2(truth, cached_truth)
    write_json(directory / "zero-insertion-fixture.json", {
        "unused_addset_rows": 0, "unused_metadata_rows": 0,
        "purpose": "Avoid generator a=0 -> N fallback for unused insertion data; no synthesized vectors",
        "core_or_harness_code_changed": False, "truth_cache_sha256": sha256_file(cached_truth),
        "files": [p.name for p in (add, meta, offsets, cached_truth)]})


ERROR = re.compile(r"\[[45]\] \d{4}-\d{2}-\d{2} ")


WARNING = re.compile(r"\[3\] \d{4}-\d{2}-\d{2} ")


# WorkSpace::_CheckAndSet grows the table and retries after this LL_Error message.
HASH_TABLE_RESIZE = re.compile(
    r"\[4\] \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} "
    r"Hash table is full! Set HashTableExponent to larger value \(default is 2\)\. "
    r"NewHashTableExponent=[1-9]\d* NewPoolSize=[1-9]\d*"
)


PENDING = re.compile(r"layer (\d+) pending queue:(\d+) split:(\d+) merge:(\d+) append:(\d+) reassign:(\d+) running:(\d+)")


def file_inventory(root):
    return {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino]
            for p in root.rglob("*") if p.is_file()}


def private_copy(source, target, record):
    require(not target.exists() and not any(p.is_symlink() for p in source.rglob("*")),
                   "Checkpoint must be an independent directory")
    before = file_inventory(source)
    needed = 250 * 1024**3 + 2 * sum(value[0] for value in before.values())
    require(shutil.disk_usage(target.parent).free >= needed, "Insufficient retained-checkpoint disk budget")
    subprocess.run(["cp", "-a", "--reflink=auto", "--sparse=always", str(source), str(target)], check=True)
    hashes = {}
    for name, (size, _, inode) in before.items():
        original, copy = source / name, target / name
        require(copy.stat().st_ino != inode and copy.stat().st_size == size, "Non-independent copy")
        hashes[name] = sha256_file(original)
        require(sha256_file(copy) == hashes[name], "Checkpoint copy hash mismatch")
    loader = target / "indexloader.ini"
    text = loader.read_text()
    require("IndexDirectory=" + str(source) in text, "Unexpected saved checkpoint path")
    loader.write_text(text.replace("IndexDirectory=" + str(source), "IndexDirectory=" + str(target)))
    require(file_inventory(source) == before, "Accepted checkpoint changed")
    write_json(record, {"source": str(source), "copy": str(target), "files": before,
                              "sha256": hashes, "original_unchanged": True, "full_hash_verification": True})
    return before


def round_result(row, queries, vectors, role):
    recall = row["recall"]
    require(row["numQueries"] == recall["numQueries"] == queries and
                   recall["k"] == 5 and recall["searchK"] == 20, "Wrong strict recall shape")
    require(math.isfinite(recall["sourceRecallAtK"]) and 0 <= recall["sourceRecallAtK"] <= 1,
                   "Nonfinite/invalid strict recall")
    performance = {key: row[key] for key in ("qps", "meanLatency", "p50", "p95", "p99")}
    require(all(math.isfinite(v) and v > 0 for v in performance.values()), "Invalid performance")
    return {"vectors": vectors, "role": role, "recall5_at20": recall["sourceRecallAtK"], **performance}


def slice_hash(path, start, count):
    with path.open("rb") as source:
        rows, dim = struct.unpack("<ii", source.read(8))
        require(dim == 128 and 0 <= start < start + count <= rows, "Invalid Float32 slice")
        source.seek(8 + start * dim * 4)
        digest = hashlib.sha256(struct.pack("<ii", count, dim))
        remaining = count * dim * 4
        while remaining:
            block = source.read(min(remaining, 8 * 1024**2))
            require(bool(block), "Incomplete Float32 source")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


RABITQ_MODEL_SHA = "feaebba9a7d116a72b8535f29f5077d8b786bfb0dc897c449fd7d28d01f338d1"


INITIAL_SHA = "7c6895e18be50f0d11cd11d2cf2ced9b15ff9132c4a267edf9fc6e4bf6b60484"


NORMALIZED_QUERY_SHA = "ca5a0cdc7997684f97bb432933ba2c9a6f5f4095179622e6ec4f6385e6e50180"


INITIAL_TRUTH_SHA = "be3e7cbe7371a3635f9feb0fa2f360d6559d6d3296e4522e26b7754fc9bcdbab"


REFERENCE = {
    "base_sha256": {
        "1": INITIAL_SHA,
        "2": "397f64b2a4531bf6f87511b6788d6e766526ffa81d01add453c76677ef45e736",
        "3": "405ec63f311019d6fd06671383cc80878b22ea78b6eda007414e1b12e69b3682",
        "4": "48a4a3e5872972adb9912f556735bc016a8f848f61ae6adb38a94f54ffb64a75",
        "5": "9951abb035570b9204dca3ad8fec98ad3c152c9a9378739dbf397839ada04c7e",
    },
    "truth_sha256": {
        "L2": {
            "1": "0453330e2e1d2ef92bd9ef4dec9cedfb9453c80fdee4f86adb840489db1f456e",
            "2": "99ede9633b2a3460eb3654cf3cbfd8144cc0c4005571e9d7e966531dd2f1cd43",
            "3": "bb11bcc1c55f54fbcd320fad383238c7d040ad87a9e23fa33269f5ff634887b3",
            "4": "121ed8f8696c40bea3198a3545f8db0e1bf35b0d7f0c74fc31ab6777faad4b01",
            "5": "6d3ea045d62c54ea32463b72ce297d06fe55eca298f4fdfb3aa66ff15075eb70",
        },
        "InnerProduct": {
            "1": INITIAL_TRUTH_SHA,
            "2": "075f31d1ad7ff38b47cacdc431066749a30c8d4883ee8947e95d474f6128686d",
            "3": "0be427941b37c7285e7ba1756b77e9a9a68cca95ecf01715679f28503b4ec140",
            "4": "dbdc806c05ddc3f05d15e3e9d67498cbd5c7f385b02c7d31cbc4c2cf861c1b1b",
            "5": "e967655063d033afd2e1826f181fb1897523e7cbea75269a78f5e4c12fbd3990",
        },
    },
}


def prefix_file(source, target, rows):
    with source.open("rb") as stream:
        count, dim = struct.unpack("<ii", stream.read(8))
        require(dim == 128 and 0 < rows <= count and
                       source.stat().st_size == 8 + count * dim * 4, "Invalid stored Float32 prefix")
        with target.open("xb") as output:
            output.write(struct.pack("<ii", rows, dim))
            remaining = rows * dim * 4
            while remaining:
                block = stream.read(min(remaining, 8 * 1024**2))
                require(bool(block), "Incomplete Float32 payload")
                output.write(block)
                remaining -= len(block)


def norm_audit(path):
    low, high, zeros, nonfinite, actual = math.inf, 0.0, [], 0, 0
    with path.open("rb") as stream:
        count, dim = struct.unpack("<ii", stream.read(8))
        require(dim == 128, "Unexpected dimension")
        while chunk := stream.read(4096 * 128 * 4):
            for row in struct.iter_unpack("<128f", chunk):
                nonfinite += sum(not math.isfinite(v) for v in row)
                norm = math.sqrt(math.fsum(v * v for v in row))
                if norm == 0:
                    zeros.append(actual)
                else:
                    require(math.isfinite(norm) and abs(norm - 1) <= 1e-6, "Invalid stored norm")
                    low, high = min(low, norm), max(high, norm)
                actual += 1
    require(actual == count and nonfinite == 0, "Invalid stored row count/nonfinite coordinates")
    return {"rows": actual, "zero_source_ids": zeros, "zero_rows": len(zeros),
            "nonfinite_coordinates": nonfinite, "minimum_nonzero_norm": low if actual > len(zeros) else 0,
            "maximum_nonzero_norm": high, "tolerance": 1e-6,
            "method": "Independent streaming reread; math.fsum of stored Float32 coordinate squares"}


MIN_FREE = 250 * 1024**3


MILLION = 1000000


def configuration(case, base, query, truth, prefix, models, rebuild=False, index=None, no_insert=False,
                  metric="InnerProduct", model_file=None, initial_count=MILLION, queries=10000):
    phase = 0 if no_insert else prefix - 1
    count = initial_count if phase else prefix * initial_count
    (case / "tmpdir").mkdir()
    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str
    with PROFILE.open() as stream:
        config.read_file(stream)
    config["Benchmark"].update({
        "VectorPath": str(base), "QueryPath": str(query), "TruthPath": str(truth),
        "IndexPath": str(index or case / "index"), "ValueType": "Float", "Layers": "1",
        "BaseVectorCount": str(count), "InsertVectorCount": str(phase * initial_count),
        "BatchNum": str(phase), "NumQueries": str(queries), "DistMethod": metric,
        "SearchK": str(SEARCH_K), "NumBuildThreads": "32", "Rebuild": str(rebuild).lower(),
        "Resume": str(phase - 2 if phase else -1),
        "QuantizerFilePath": str(model_file or models / "rabitq4.bin"),
    })
    for section in ("SelectHead", "BuildHead", "BuildSSDIndex"):
        config[section]["NumberOfThreads"] = "32"
    config["SelectHead"]["Ratio"] = "0.2"
    config["BuildSSDIndex"].update(Layers="1", AppendThreadNum="24", PostingVectorLimit="0")
    with (case / "benchmark.ini").open("x") as stream:
        config.write(stream, space_around_delimiters=False)
    if phase == 0:
        stage_zero_insert_fixture(case, count, queries, truth, metric)


class MaintenanceProgress:
    """Bound persistent undrained work by population/queue reduction, not job churn."""

    def __init__(self, vectors, limit):
        self.vectors, self.limit = vectors, limit
        self.since = None
        self.low_water = {}
        self.samples = 0
        self.recent = []
        self.refill_since = None
        self.refill_processed = 0

    def observe(self, line, now):
        if "zero-replica refill workers:" in line:
            self.refill_since, self.refill_processed = now, 0
        refill = re.search(r"zero-replica refill progress (\d+)/", line)
        if refill and int(refill[1]) > self.refill_processed:
            self.refill_since, self.refill_processed = now, int(refill[1])
        if "zero-replica refill done," in line:
            self.refill_since = None
        count = re.search(r"Current Vector Num:\s*(\d+)", line)
        if count and int(count[1]) > self.vectors:
            self.vectors = int(count[1])
            self.low_water = {}
            if self.since is not None:
                self.since = now
        if "ALL DONE" in line:
            self.since, self.low_water, self.samples = None, {}, 0
        queue = PENDING.search(line)
        if queue:
            outstanding = int(queue[2]) + int(queue[7])
            if outstanding == 0:
                self.since, self.low_water, self.samples = None, {}, 0
                return
            if self.since is None or outstanding < self.low_water.get(queue[1], math.inf):
                self.since = now
                if self.refill_since is not None:
                    self.refill_since = now
            self.low_water[queue[1]] = min(outstanding, self.low_water.get(queue[1], math.inf))
            self.samples += 1
            self.recent = (self.recent + [line])[-12:]

    def state(self, now):
        return {"population_last_sample": self.vectors, "queue_low_water": self.low_water,
                "seconds_without_population_or_queue_reduction": 0 if self.since is None else now - self.since,
                "pending_observations": self.samples, "limit_seconds": self.limit, "recent_queues": self.recent,
                "refill_processed_last_sample": self.refill_processed,
                "refill_seconds_without_observed_progress": 0 if self.refill_since is None else now - self.refill_since}

    def check(self, now):
        require(self.refill_since is None or now - self.refill_since < self.limit,
                       f"Bounded initial zero-replica refill: no observed progress/completion for{self.limit}s")
        require(self.since is None or self.samples < 2 or now - self.since < self.limit,
                       f"Bounded maintenance nonconvergence: no population/queue reduction for{self.limit}s")


def monitor(case, binary, vectors, timeout=86400, maintenance_stall_seconds=None):
    command = ["/usr/bin/time", "-v", "-o", str(case / "resources.txt"), str(binary),
               "--run_test=SPFreshTest/BenchmarkFromConfig", "--log_level=message"]
    env = dict(os.environ, BENCHMARK_CONFIG=str(case / "benchmark.ini"),
               BENCHMARK_OUTPUT=str(case / "output.json"), TMPDIR=str(case / "tmpdir"))
    started = progress = time.monotonic()
    offset, tail, low_water, completed = 0, "", {}, ""
    maintenance = MaintenanceProgress(vectors, maintenance_stall_seconds) if maintenance_stall_seconds else None
    with (case / "benchmark.log").open("x") as log:
        process = subprocess.Popen(command, cwd=case, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                now = time.monotonic()
                with (case / "benchmark.log").open(errors="replace") as stream:
                    stream.seek(offset)
                    text = tail + stream.read()
                    offset = stream.tell()
                lines, _, tail = text.rpartition("\n")
                if vectors == 0 and lines:
                    progress = now
                for line in lines.splitlines():
                    if maintenance:
                        maintenance.observe(line, now)
                    count = re.search(r"Current Vector Num:\s*(\d+)", line)
                    if count and int(count[1]) > vectors:
                        vectors, progress, low_water = int(count[1]), now, {}
                    queue = PENDING.search(line)
                    if queue:
                        outstanding = int(queue[2]) + int(queue[7])
                        if outstanding < low_water.get(queue[1], math.inf):
                            low_water[queue[1]], progress = outstanding, now
                    if maintenance is None and "total_completed" in line and line.split("total_completed", 1)[1] != completed:
                        completed, progress = line.split("total_completed", 1)[1], now
                    if "Checkpoint saved:" in line:
                        progress = now
                state = {"status": "running", "pid": process.pid, "pgid": process.pid,
                         "command": command, "seconds": now - started, "observed_vectors": vectors,
                         "stationary_seconds": now - progress, "queue_low_water": low_water}
                if maintenance:
                    state["maintenance_progress"] = maintenance.state(now)
                if process.poll() is not None:
                    state.update(status="exited", returncode=process.returncode)
                write_json(case / "benchmark-status.json", state)
                if process.poll() is not None:
                    require(process.returncode == 0, f"Benchmark hard failure: exit{process.returncode}")
                    return
                require(now - started < timeout, f"Process safety limit ({timeout}s)")
                if maintenance:
                    maintenance.check(now)
                require(now - progress < 7200, "No observable progress for2h")
                require(shutil.disk_usage(case).free >= MIN_FREE, "250GiB disk safety floor")
                time.sleep(20)
        except BaseException as error:
            actions = []
            if process.poll() is None:
                for sig, wait in ((signal.SIGINT, 60), (signal.SIGTERM, 20), (signal.SIGKILL, 10)):
                    try:
                        os.killpg(process.pid, sig)
                        actions.append(sig.name)
                        process.wait(timeout=wait)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                    except ProcessLookupError:
                        break
            write_json(case / "hard-failure.json",
                              {"error": str(error), "pid": process.pid, "signals": actions,
                               "returncode": process.poll()})
            raise


def joined_truth(paths, queries=10000):
    require(2 <= len(paths) <= 5, "Diagnostic scope is2M through5M")
    rows = [p.read_bytes() for p in paths]
    block = queries * 5 * 4
    require(all(len(row) == 8 + 2 * block and struct.unpack_from("<ii", row) == (2 * queries, 5)
                       for row in rows), "Wrong normalized exact truth layout")
    rows[0] = rows[-2]
    return (struct.pack("<ii", len(rows) * 2 * queries, 5) +
            b"".join(row[8:8 + block] for row in rows) + b"".join(row[8 + block:] for row in rows))


def physical_audit(index, expected, output, previous_missing=(), record_bytes=93):
    """Read-only one-layer traversal; retired heads alone are not data loss."""
    require(isinstance(record_bytes, int) and record_bytes > 5, "Invalid native posting record width")
    output.mkdir()
    versions = bytearray([254]) * expected
    members = bytearray(expected)
    raw = (index / "DeletedIDs.bin_0").read_bytes()
    count, = struct.unpack_from("<i", raw)
    require(len(raw) == 4 + 5 * count, "Bad version dictionary")
    invalid_members = duplicates = 0
    for vid, version in struct.iter_unpack("<iB", raw[4:]):
        if not 0 <= vid < expected:
            invalid_members += 1
        else:
            duplicates += members[vid]
            members[vid], versions[vid] = 1, version
    head_data = (index / "SPTAGHeadVectorIDs.bin").read_bytes()
    heads, columns = struct.unpack_from("<ii", head_data)
    labels = (index / "HeadIndex/deletes.bin").read_bytes()
    deleted, label_rows, label_columns = struct.unpack_from("<iii", labels)
    require(columns == label_columns == 1 and heads == label_rows and
                   len(head_data) == 8 + 4 * heads and len(labels) == 12 + heads,
                   "Unexpected top BKT mapping/deletion layout")
    require(labels[12:].count(1) == deleted, "Top head deletion count mismatch")
    active = {vid for (vid,), label in zip(struct.iter_unpack("<i", head_data[8:]), labels[12:])
              if label != 1}
    mapping = (index / "ssdmapping_0").read_bytes()
    rows, columns = struct.unpack_from("<ii", mapping)
    require(columns == 10 and len(mapping) == 8 + rows * 80, "Unexpected FILEIO map layout")
    seen, seen_any_map = bytearray(expected), bytearray(expected)
    stats = Counter()
    mapped = set()
    before = {name: [(index / name).stat().st_size, (index / name).stat().st_mtime_ns]
              for name in ("ssdmapping_0", "ssdmapping_0_postings", "DeletedIDs.bin_0")}
    fd = os.open(index / "ssdmapping_0_postings", os.O_RDONLY)
    try:
        for row in struct.iter_unpack("<10q", mapping[8:]):
            key, length = row[0], row[1] & 0xFFFFFFFF
            mapped.add(key)
            reachable = key in active
            stats["active_postings" if reachable else "inactive_postings"] += 1
            if not 0 <= length <= 32768 or length % record_bytes:
                stats["malformed_postings"] += 1
                continue
            payload = b"".join(os.pread(fd, 4096, page * 4096)
                               for page in row[2:2 + (length + 4095) // 4096])[:length]
            if len(payload) != length:
                stats["short_reads"] += 1
                continue
            stats["max_records"] = max(stats["max_records"], length // record_bytes)
            current, per_posting = 0, set()
            for offset in range(0, length, record_bytes):
                vid, version = struct.unpack_from("<iB", payload, offset)
                stats["records"] += 1
                if not 0 <= vid < expected:
                    stats["invalid_id_records"] += 1
                elif not members[vid]:
                    stats["absent_member_records"] += 1
                elif versions[vid] != version:
                    stats["stale_records"] += 1
                else:
                    stats["current_records"] += 1
                    stats["duplicate_current_records"] += vid in per_posting
                    per_posting.add(vid)
                    current += 1
                    seen_any_map[vid] = 1
                    if reachable:
                        seen[vid] = 1
            if reachable and not current:
                stats["active_postings_without_current_records"] += 1
    finally:
        os.close(fd)
    after = {name: [(index / name).stat().st_size, (index / name).stat().st_mtime_ns] for name in before}
    require(before == after, "Checkpoint changed during read-only coverage scan")
    offsets = (index / "metadataIndex.bin").read_bytes()
    metadata = (index / "metadata.bin").read_bytes()
    metadata_rows, = struct.unpack_from("<i", offsets)
    require(metadata_rows == expected and len(offsets) == 4 + (expected + 1) * 8,
                   "Unexpected source metadata layout")
    missing_sources = set()
    with (output / "missing-source-ids.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("index_vid", "source_id", "version", "logical_member", "present_in_inactive_posting"))
        for vid in range(expected):
            if seen[vid]:
                continue
            start, end = struct.unpack_from("<QQ", offsets, 4 + vid * 8)
            source = int(metadata[start:end])
            missing_sources.add(source)
            writer.writerow((vid, source, versions[vid] if members[vid] else -1,
                             members[vid], seen_any_map[vid]))
    previous_missing = set(previous_missing)
    newly, recovered = missing_sources - previous_missing, previous_missing - missing_sources
    for name, ids in (("newly-missing.csv", newly), ("recovered.csv", recovered)):
        (output / name).write_text("source_id\n" + "".join(f"{vid}\n" for vid in sorted(ids)))
    result = {"layers": 1, "expected_source_ids": expected, "logical_members": sum(members),
              "unique_current_reachable_ids": sum(seen), "unique_current_all_mapped_ids": sum(seen_any_map),
              "missing_source_count": len(missing_sources), "newly_missing_count": len(newly),
              "recovered_since_previous_count": len(recovered), "live_top_heads": len(active),
              "retired_top_heads": deleted, "active_heads_without_posting": len(active - mapped),
              "invalid_version_members": invalid_members, "duplicate_version_members": duplicates,
              "checkpoint_size_mtime_unchanged": True, **dict(stats)}
    result["coverage_complete"] = not (missing_sources or active - mapped or invalid_members or duplicates or
                                       stats["malformed_postings"] or stats["short_reads"])
    write_json(output / "coverage.json", result)
    return result, missing_sources


def log_review(case):
    warnings, errors, resizes, queues, barriers = [], [], [], {}, []
    for line in (case / "benchmark.log").read_text(errors="replace").splitlines():
        if HASH_TABLE_RESIZE.fullmatch(line):
            resizes.append(line)
        elif ERROR.search(line):
            errors.append(line)
        if WARNING.search(line):
            warnings.append(line)
        found = re.search(r"layer (\d+) (?:ALL DONE \||pending queue:)", line)
        if found:
            queues[found[1]] = line
        found = re.search(r"BatchBarrierWait seconds=([0-9.]+)", line)
        if found:
            barriers.append(float(found[1]))
    drained = bool(queues) and all("ALL DONE" in line or (PENDING.search(line) and
                  all(int(value) == 0 for value in PENDING.search(line).groups()[1:]))
                  for line in queues.values())
    result = {"error_messages": errors, "warnings": warnings, "last_queues": queues,
              "handled_hash_table_resizes": resizes,
              "observed_queues_drained": drained, "barrier_seconds": barriers,
              "warning_counts": dict(Counter("async-read-timeout" if "timeout, continue" in line else
                                              "head-lock-retry" if "is being locked" in line else "other"
                                              for line in warnings))}
    write_json(case / "log-review.json", result)
    return result


def evaluate(args, prefix, base, truth, prior_missing, layout=None, initial_count=MILLION, queries=10000):
    layout = layout or {"code_bytes": 88, "model_sha256": RABITQ_MODEL_SHA,
                        "initial_sha256": INITIAL_SHA, "query_sha256": NORMALIZED_QUERY_SHA}
    record_bytes = layout["code_bytes"] + 5
    case = args.root / f"prefix-{prefix}"
    index = case / ("index" if prefix == 1 else f"index_{prefix - 2}")
    expected = prefix * initial_count
    saved = configparser.ConfigParser()
    saved.read(index / "indexloader.ini")
    require(saved.getint("BuildSSDIndex", "Layers") == 1 and
                   saved.get("BuildSSDIndex", "Storage") == "FILEIO" and
                   saved.getint("Base", "Dim") == layout["code_bytes"] and
                   sha256_file(index / "quantizer.bin") == layout["model_sha256"],
                   "Wrong layer/type/model; cannot interpret this experiment")
    for key, value in {"PostingPageLimit": "4", "SearchPostingPageLimit": "4", "BufferLength": "4",
                       "AppendThreadNum": "24", "MaxCheck": "8192", "PostingVectorLimit": "0",
                       "SearchInternalResultNum": "64"}.items():
        require(saved.get("BuildSSDIndex", key) == value, "Profile changed: " + key)
    require(saved.getfloat("BuildSSDIndex", "LatencyLimit") == 50 and
                   not saved.getboolean("BuildSSDIndex", "AsyncMergeInSearch"), "Search profile changed")
    data = json.loads((case / "output.json").read_text())
    require(data["config"]["layers"] == 1 and data["config"]["ValueType"] == "Float",
                   "Wrong measured layer/input type")
    require(data["config"]["DistMethod"] == args.metric, "Wrong measured distance metric")
    generated = {
        f"perftest_vector.bin.Float_{initial_count}_128": layout["initial_sha256"],
        f"perftest_query.bin.Float_{queries}_128": layout["query_sha256"],
    }
    if prefix > 1:
        generated[f"perftest_addvector.bin.Float_{(prefix - 1) * initial_count}_128"] = slice_hash(
            base, initial_count, (prefix - 1) * initial_count)
    require(all(sha256_file(case / name) == sha for name, sha in generated.items()),
                   "Actual harness input bytes changed")
    write_json(case / "generated-data-verification.json", generated)
    results = data["results"]
    measured = []
    metric_issues = []
    before = (prefix - 1) * initial_count if prefix > 1 else initial_count
    for name in ("benchmark0_query_before_insert", "benchmark0b_query_before_insert_round2"):
        try:
            measured.append(round_result(results[name], queries, before, "prelude"))
        except Exception as error:
            metric_issues.append(str(error))
    entry = None
    if prefix > 1:
        entry = results["benchmark1_insert"][f"batch_{prefix - 1}"]
        for name in ("search", "search_round2"):
            try:
                measured.append(round_result(entry[name], queries, expected, "current-prefix"))
            except Exception as error:
                metric_issues.append(str(error))
    review = log_review(case)
    command = [str(args.checker), str(index), str(initial_count), str(initial_count), str(prefix - 1)]
    with (case / "count-check.log").open("x") as stream:
        counted = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=3600)
    reload = case / "reload"
    reload.mkdir()
    configuration(reload, base, args.prepared / "query.f32bin", truth, prefix,
                  args.root / "models", index=index, no_insert=True, metric=args.metric,
                  model_file=layout.get("model_path"), initial_count=initial_count, queries=queries)
    monitor(reload, args.binary, expected, timeout=7200)
    reload_data = json.loads((reload / "output.json").read_text())
    require(reload_data["config"]["layers"] == 1 and
                   reload_data["config"]["baseVectorCount"] == expected and
                   reload_data["config"]["BatchNum"] == 0, "Wrong reload phase")
    reloaded = []
    for name in ("benchmark0_query_before_insert", "benchmark0b_query_before_insert_round2"):
        try:
            reloaded.append(round_result(reload_data["results"][name], queries, expected, "fresh-reload"))
        except Exception as error:
            metric_issues.append("reload: " + str(error))
    reload_review = log_review(reload)
    try:
        coverage, missing = physical_audit(index, expected, case / "physical-audit", prior_missing, record_bytes)
    except Exception as error:
        coverage, missing = {"coverage_complete": False, "audit_error": str(error),
                             "missing_source_count": None, "history_carried_forward_not_current_measurement": True}, set(prior_missing)
        write_json(case / "physical-audit/coverage.json", coverage)
    integrity = coverage["coverage_complete"] and counted.returncode == 0 and not review["error_messages"]
    integrity = integrity and (prefix == 1 or review["observed_queues_drained"]) and not reload_review["error_messages"]
    result = {"status": "integrity-accepted" if integrity else "NOT-integrity-accepted-diagnostic-continuation",
              "vectors": expected, "checkpoint": str(index), "loadable": True,
              "diagnostic_continuation_authorized": True, "query_rounds": measured,
              "reload_query_rounds": reloaded, "metric_issues": metric_issues, "insertion": entry,
              "raw_build_fields": {key: value for key, value in results.items() if "build" in key.lower()},
              "logical_check_returncode": counted.returncode, "maintenance": review,
              "reload_log_review": reload_review, "physical_coverage": coverage,
              "truth_sha256": sha256_file(truth), "model_sha256": sha256_file(index / "quantizer.bin"),
              "posting_base_limit": 4 * 4096 // record_bytes, "posting_hard_limit": 8 * 4096 // record_bytes}
    write_json(case / "diagnostic-result.json", result)
    return result, missing, index


def prune_completed_seed(root, prefix, checkpoint):
    root = root.resolve()
    case = root / f"prefix-{prefix}"
    seed = case / "index"
    expected = case / f"index_{prefix - 2}"
    source = root / f"prefix-{prefix - 1}" / ("index" if prefix == 2 else f"index_{prefix - 3}")
    require(2 <= prefix <= 5 and checkpoint.resolve() == expected and expected.is_dir(),
                   "Only an obsolete private seed may be pruned")
    require(seed.is_dir() and not seed.is_symlink() and seed.resolve() == seed and
                   all(not p.is_symlink() and (not p.is_file() or p.stat().st_nlink == 1)
                       for p in seed.rglob("*")), "Seed must be an independent owned directory")
    proof = json.loads((case / "checkpoint-copy.json").read_text())
    require(Path(proof["copy"]) == seed and Path(proof["source"]) == source and
                   proof["full_hash_verification"] and file_inventory(source) == proof["files"],
                   "Preserved predecessor or copy provenance changed")
    result = json.loads((case / "diagnostic-result.json").read_text())
    require(result["loadable"] and Path(result["checkpoint"]) == expected and
                   all(json.loads((folder / "benchmark-status.json").read_text())["returncode"] == 0
                       for folder in (case, case / "reload")), "Save and fresh reload must complete")
    inventory = file_inventory(seed)
    require(set(inventory) == set(proof["files"]) | {"checkpoint.txt"}, "Unexpected seed files")
    for name, metadata in proof["files"].items():
        if name == "indexloader.ini":
            continue
        path = seed / name
        if metadata[0] < 256 * 1024**2:
            require(sha256_file(path) == proof["sha256"][name], "Seed contains changed state: " + name)
        else:
            require(inventory[name][:2] == metadata[:2], "Seed posting changed after verified copy")
    handles = subprocess.run(["lsof", "-t", "+D", str(seed)], capture_output=True, text=True)
    require(handles.returncode == 1 and not handles.stdout.strip(), "Private seed has open handles")
    record = {"status": "verified-before-pruning", "path": str(seed), "canonical_predecessor": str(source),
              "canonical_current": str(expected), "inventory": inventory,
              "allocated_bytes": sum(p.stat().st_blocks * 512 for p in (seed, *seed.rglob("*"))),
              "small_control_files": {name: (seed / name).read_text()
                                      for name in ("indexloader.ini", "checkpoint.txt")},
              "copy_proof": str(case / "checkpoint-copy.json"), "lsof_stderr": handles.stderr,
              "free_before": shutil.disk_usage(root).free}
    output = case / "private-seed-pruning.json"
    require(not output.exists(), "Never replace pruning evidence")
    write_json(output, record)
    shutil.rmtree(seed)
    require(not seed.exists() and source.is_dir() and expected.is_dir(), "Pruning invariant failed")
    record.update(status="pruned-obsolete-private-seed", free_after=shutil.disk_usage(root).free)
    write_json(output, record)
    return str(output)


def accepted_stage(root, prefix):
    case = root / f"prefix-{prefix}"
    index = case / ("index" if prefix == 1 else f"index_{prefix - 2}")
    result = json.loads((case / "diagnostic-result.json").read_text())
    require(result["status"] == "integrity-accepted" and result["loadable"] and
                   Path(result["checkpoint"]) == index and not result["metric_issues"] and
                   result["physical_coverage"]["coverage_complete"] and
                   result["physical_coverage"]["missing_source_count"] == 0 and
                   all(json.loads((folder / "benchmark-status.json").read_text())["returncode"] == 0
                       for folder in (case, case / "reload")),
                   "Never prune a failed or unaudited stage")
    return index


def prune_completed_predecessor(root, prefix, checkpoint):
    root = root.resolve()
    require(3 <= prefix <= 5, "Never prune the initial checkpoint")
    case = root / f"prefix-{prefix}"
    current = accepted_stage(root, prefix)
    source = accepted_stage(root, prefix - 1)
    initial = root / "prefix-1/index"
    require(checkpoint.resolve() == current and current.is_dir() and initial.is_dir() and
                   source.is_dir() and source.resolve() == source and not source.is_symlink() and
                   all(not p.is_symlink() and (not p.is_file() or p.stat().st_nlink == 1)
                       for p in source.rglob("*")), "Only an independent obsolete checkpoint may be pruned")
    proof = json.loads((case / "checkpoint-copy.json").read_text())
    seed = json.loads((case / "private-seed-pruning.json").read_text())
    inventory = file_inventory(source)
    require(Path(proof["source"]) == source and proof["full_hash_verification"] and
                   proof["original_unchanged"] and inventory == proof["files"] and
                   set(proof["sha256"]) == set(inventory) and
                   seed["status"] == "pruned-obsolete-private-seed" and
                   Path(seed["canonical_predecessor"]) == source and
                   Path(seed["canonical_current"]) == current,
                   "Completed successor/copy/pruning provenance changed")
    handles = subprocess.run(["lsof", "-t", "+D", str(source)], capture_output=True, text=True)
    require(handles.returncode == 1 and not handles.stdout.strip(), "Predecessor has open handles")
    output = case / "predecessor-pruning.json"
    require(not output.exists(), "Never replace predecessor pruning evidence")
    record = {"status": "verified-before-pruning", "path": str(source), "retained_initial": str(initial),
              "retained_current": str(current), "inventory": inventory, "sha256": proof["sha256"],
              "verification": "Full hashes at private copy; inode, size and mtime unchanged since that copy",
              "copy_proof": str(case / "checkpoint-copy.json"),
              "allocated_bytes": sum(p.stat().st_blocks * 512 for p in (source, *source.rglob("*"))),
              "lsof_stderr": handles.stderr, "free_before": shutil.disk_usage(root).free}
    write_json(output, record)
    shutil.rmtree(source)
    require(not source.exists() and initial.is_dir() and current.is_dir(), "Retention invariant failed")
    record.update(status="pruned-verified-predecessor", free_after=shutil.disk_usage(root).free)
    write_json(output, record)
    return str(output)


def prune_generated_caches(root, prefix, base, query, initial_count=MILLION, queries=10000):
    root = root.resolve()
    require(1 <= prefix <= 5, "Invalid cache-pruning stage")
    checkpoint = accepted_stage(root, prefix)
    require(checkpoint.is_dir(), "Current checkpoint must be retained")
    case = root / f"prefix-{prefix}"
    proof = json.loads((case / "generated-data-verification.json").read_text())
    names = {f"perftest_vector.bin.Float_{initial_count}_128",
             f"perftest_query.bin.Float_{queries}_128"}
    if prefix > 1:
        names.add(f"perftest_addvector.bin.Float_{(prefix - 1) * initial_count}_128")
    require(set(proof) == names, "Unexpected generated input inventory")
    files = {case / name: sha for name, sha in proof.items()}
    files[case / "reload" / f"perftest_vector.bin.Float_{prefix * initial_count}_128"] = sha256_file(base)
    files[case / "reload" / f"perftest_query.bin.Float_{queries}_128"] = sha256_file(query)
    for path, sha in files.items():
        require(path.is_file() and not path.is_symlink() and path.resolve() == path and
                       path.stat().st_nlink == 1 and sha256_file(path) == sha,
                       "Refusing to prune changed or shared cache: " + str(path))
    handles = subprocess.run(["lsof", "-t", "--", *(str(path) for path in files)],
                             capture_output=True, text=True)
    require(handles.returncode == 1 and not handles.stdout.strip(), "Generated caches have open handles")
    output = case / "generated-cache-pruning.json"
    require(not output.exists(), "Never replace cache pruning evidence")
    record = {"status": "verified-before-pruning", "retained_checkpoint": str(checkpoint),
              "files": {str(path): {"sha256": sha, "bytes": path.stat().st_size} for path, sha in files.items()},
              "lsof_stderr": handles.stderr, "free_before": shutil.disk_usage(root).free}
    write_json(output, record)
    for path in files:
        path.unlink()
    require(checkpoint.is_dir() and all(not path.exists() for path in files), "Cache retention invariant failed")
    record.update(status="pruned-verified-generated-caches", free_after=shutil.disk_usage(root).free)
    write_json(output, record)
    return str(output)


CASES = {
    "rabitq4-l2": {"metric": "L2", "quantizer": "RaBitQ4", "model_file": "rabitq4.bin",
                   "model_sha256": RABITQ_MODEL_SHA},
    "rabitq4-ip": {"metric": "InnerProduct", "quantizer": "RaBitQ4", "model_file": "rabitq4.bin",
                   "model_sha256": RABITQ_MODEL_SHA},
    "opq64-ip": {"metric": "InnerProduct", "quantizer": "OPQ64x8", "model_file": "opq64.bin",
                 "model_sha256": "8080a514d5e5988cd6cf445003df4c25b575b1b876690359b807dd09e668712d"},
}


SCHEDULE = (1000000, 2000000, 3000000, 4000000, 5000000)


MAINTENANCE_STALL_SECONDS = 3600


IP_HEAD_FILES = {
    "AnnService/inc/Core/Common/BKTree.h",
    "AnnService/inc/Core/SPANN/ExtraDynamicSearcher.h",
}


IMPLEMENTATIONS = {
    "original-maintenance": ("clean-harness-build-complete", set()),
    "ip-head-fix": ("ip-head-build-complete", IP_HEAD_FILES),
    "ip-head-opq-encoding-fix": ("ip-head-opq-encoding-build-complete", IP_HEAD_FILES | {
        "AnnService/inc/Core/Common/PQQuantizer.h",
    }),
}


def require_data_paths(paths):
    require(all(str(path.resolve()).startswith("/data/") for path in paths),
                   "All experiment paths must resolve under/data")


def workload(args):
    case = CASES[args.case]
    data = json.loads((args.prepared / "manifest.json").read_text())
    unit, queries, prefixes = MILLION, 10000, 5
    require(data["metric"] == case["metric"] and data.get("initial_count", unit) == unit and
                   data["queries"] == queries, "Wrong prepared case metric or schedule")
    data["base_sha256"] = data.get("base_sha256", data.get("prefix_base_sha256"))
    data["truth_sha256"] = data.get("truth_sha256", data.get("truth"))
    require(all(str(n) in data["base_sha256"] and str(n) in data["truth_sha256"]
                       for n in range(1, prefixes + 1)), "Missing exact prefix data/truth")
    require(data["base_sha256"]["1"] == INITIAL_SHA and
            data["query_sha256"] == NORMALIZED_QUERY_SHA,
            "Current cases require the frozen normalized1M/10K-query workload")
    return case, data, unit, queries, prefixes


def case_layout(case, model):
    if case["quantizer"] == "OPQ64x8":
        layout = model_layout(model)
    else:
        native = model_info(model)
        layout = {"quantizer": "RaBitQ4", "code_bytes": native["code_bytes"],
                  "model_sha256": native["sha256"]}
    require(layout["model_sha256"] == case["model_sha256"], "Wrong frozen initial1M-trained model")
    width = layout["code_bytes"] + 5
    layout.update(record_bytes=width, posting_base_limit=4 * 4096 // width,
                  posting_hard_limit=8 * 4096 // width)
    return layout


def model_layout(path):
    raw = path.read_bytes()
    require(len(raw) >= 15, "Truncated OPQ model")
    kind, dtype, subvectors, centers, subdimension = struct.unpack_from("<BBiii", raw)
    require((kind, dtype, subvectors, centers, subdimension) == (2, 3, 64, 256, 2),
                   "Requires native Float OPQ64/PQ64x8")
    metric_offset = 14 + subvectors * centers * subdimension * 4
    expected = metric_offset + 1 + (subvectors * subdimension) ** 2 * 4
    require(len(raw) == expected and raw[metric_offset] == 2,
                   "OPQ must include native InnerProduct metric before its rotation")
    width = subvectors + 5
    return {"quantizer": "OPQ", "code_bytes": subvectors, "record_bytes": width,
            "model_sha256": sha256_file(path), "posting_base_limit": 4 * 4096 // width,
            "posting_hard_limit": 8 * 4096 // width}


def identity(args):
    build = json.loads(args.build_manifest.read_text())
    core_path = args.build_manifest.parent / "core-before.json"
    core = json.loads(core_path.read_text())["annservice_files"]
    variant = getattr(args, "implementation", "original-maintenance")
    status, allowed_changes = IMPLEMENTATIONS[variant]
    require(build["status"] == status and
                   build.get("implementation_variant", "original-maintenance") == variant and
                   sha256_file(args.binary) == build["binary_sha256"], "Wrong measured implementation")
    source = Path(build["source_root"])
    require(len(core) == 204 and all(sha256_file(Path(build["source_root"]) / name) == digest
                                            for name, digest in core.items()), "Frozen core changed")
    baseline_sha = None
    if allowed_changes:
        baseline_path = args.build_manifest.parent / "baseline-core.json"
        baseline_sha = sha256_file(baseline_path)
        baseline = json.loads(baseline_path.read_text())["annservice_files"]
        actual_files = {p.relative_to(source).as_posix()
                        for p in (source / "AnnService").rglob("*") if p.is_file()}
        require(baseline_sha == build["baseline_core_sha256"] and
                       set(baseline) == set(core) == actual_files and
                       {name for name in core if core[name] != baseline[name]} == allowed_changes and
                       set(build["intentional_core_changes"]) == allowed_changes,
                       "Patched build differs from the reviewed implementation changes")
        require(sha256_file(source / "Test/src/SPFreshTest.cpp") == build["harness_sha256"],
                       "Measured benchmark harness changed")
    return {"implementation_variant": variant, "baseline_core_sha256": baseline_sha,
            "binary_sha256": build["binary_sha256"], "harness_sha256": build["harness_sha256"],
            "core_files_verified": len(core), "core_reference_sha256": sha256_file(core_path),
            "runtime_sha256": {name: sha256_file(Path(name)) for name in
                              re.findall(r"(?:=> )?(/[^\s]+)", build["link_dependencies"])}}


def run_case(args):
    args.case = getattr(args, "case", "opq64-ip")
    case_definition, data, unit, queries, prefixes = workload(args)
    stop_after = getattr(args, "stop_after_prefix", None)
    if stop_after is None:
        stop_after = prefixes
    require(type(stop_after) is int and 1 <= stop_after <= prefixes,
                   "Stop prefix must be within the configured schedule")
    args.metric = case_definition["metric"]
    retention = getattr(args, "retain_checkpoints", "all")
    implementation = getattr(args, "implementation", "original-maintenance")
    model = args.model.resolve()
    require_data_paths((args.root, args.prepared, args.binary, args.checker, args.build_manifest, model))
    require(model.is_file(), f"Frozen model file not found: {model}")
    layout = case_layout(case_definition, model)
    require(args.budget is not None, "A reviewed budget for the planned run is required")
    budget = json.loads(args.budget.read_text())
    require(budget.get("stop_after_prefix", prefixes) == stop_after,
            "Budget stop prefix does not match the planned run")
    require(budget["status"] == "feasible" and budget["floor_bytes"] == MIN_FREE and
            budget.get("case", args.case) == args.case and
            shutil.disk_usage(args.root.parent).free >= budget["floor_bytes"] +
            budget["projected_additional_bytes"], "Planned-run resource budget unavailable")
    if retention == "initial-and-latest":
        require(budget.get("retention_mode") == retention,
                "Reduced retention needs an explicit budget for this new run")
    if stop_after < prefixes or retention == "initial-and-latest":
        require(Path(budget["output_root"]).resolve() == args.root.resolve(),
                "Bounded or reduced-retention runs need a budget for this exact output root")
    args.root.mkdir(exist_ok=False)
    (args.root / "models").mkdir()
    shutil.copy2(model, args.root / "models" / case_definition["model_file"])
    layout.update(model_path=args.root / "models" / case_definition["model_file"],
                  initial_sha256=data["base_sha256"]["1"], query_sha256=data["query_sha256"])
    frozen = identity(args)
    archive = args.root / "recipe"
    archive.mkdir()
    for path in (Path(__file__), PROFILE):
        shutil.copy2(path, archive / path.name)
    manifest = dict(frozen, schema_version=1, case=args.case, status="starting",
                    driver_pid=os.getpid(), quantizer=case_definition["quantizer"],
                    metric=args.metric, model_sha256=layout["model_sha256"],
                    code_bytes=layout["code_bytes"], record_bytes=layout["record_bytes"],
                    posting_base_limit=layout["posting_base_limit"],
                    posting_hard_limit=layout["posting_hard_limit"], layers=1, head_ratio=0.2,
                    maximum_vectors=unit * prefixes, prepared=str(args.prepared), milestones=[],
                    stop_after_prefix=stop_after, run_limit_vectors=unit * stop_after,
                    schedule_vectors=[n * unit for n in range(1, prefixes + 1)],
                    stages=[{"prefix": n, "vectors": n * unit, "status": "not_run"}
                            for n in range(1, prefixes + 1)],
                    maintenance_stall_seconds=MAINTENANCE_STALL_SECONDS,
                    tiny_success_required=False,
                    diagnostic_continuation_authorized=implementation == "original-maintenance", raw_rerank=False,
                    storage=subprocess.check_output(
                        ["findmnt", "-T", str(args.root), "-n", "-o", "SOURCE,TARGET,FSTYPE"], text=True).strip(),
                    comparison_caveat="OPQ69B versus RaBitQ93B records changes4+4 capacity/topology; "
                                      "shared-host load, cache state and independent randomized builds differ",
                    retention_mode=retention,
                    retention_policy=("all canonical milestones; prune only obsolete private seeds"
                                      if retention == "all" else
                                      "initial and latest accepted checkpoints; preserve all failures and evidence"))
    missing, ever_missing, previous = set(), set(), None
    truths = []
    try:
        for prefix in range(1, stop_after + 1):
            manifest.update(status="preparing-stage", current_prefix=prefix, operation="preparation")
            manifest["stages"][prefix - 1]["status"] = "in_progress"
            write_json(args.root / "manifest.json", manifest)
            require(identity(args) == frozen, "Measured implementation changed")
            base, truth = args.prepared / f"base{prefix}m.f32bin", args.prepared / f"truth/truth{prefix}m.bin"
            require_data_paths((base, truth, args.prepared / "query.f32bin"))
            require(sha256_file(base) == data["base_sha256"][str(prefix)] and
                           sha256_file(truth) == data["truth_sha256"][str(prefix)] and
                           sha256_file(args.prepared / "query.f32bin") == layout["query_sha256"],
                           "Prepared normalized case input/truth bytes changed")
            truths.append(truth)
            case = args.root / f"prefix-{prefix}"
            case.mkdir()
            before = None
            if previous is not None:
                before = private_copy(previous, case / "index", case / "checkpoint-copy.json")
                if prefix > 2:
                    (case / f"index_{prefix - 3}").symlink_to("index", target_is_directory=True)
                joined = case / f"perftest_batchtruth.{args.metric}.Float_{unit}_128_{queries}_5_{unit}_0_{prefix - 1}"
                joined.write_bytes(joined_truth(truths, queries))
            else:
                joined = truth
            configuration(case, base, args.prepared / "query.f32bin", joined, prefix,
                                 args.root / "models", rebuild=prefix == 1, model_file=layout["model_path"],
                                 initial_count=unit, queries=queries, metric=args.metric)
            manifest.update(status="running-benchmark",
                            operation="initial-build-and-query" if prefix == 1 else "increment-and-query")
            write_json(args.root / "manifest.json", manifest)
            try:
                monitor(case, args.binary, 0 if prefix == 1 else (prefix - 1) * unit,
                        timeout=86400, maintenance_stall_seconds=MAINTENANCE_STALL_SECONDS)
            finally:
                if before is not None:
                    require(file_inventory(previous) == before, "Preserved predecessor changed")
            manifest.update(operation="save-reload-logical-and-coverage-audit")
            write_json(args.root / "manifest.json", manifest)
            result, missing, previous = evaluate(
                args, prefix, base, truth, missing, layout, unit, queries)
            require(not result["metric_issues"], "Invalid query metrics; preserve this stage")
            ever_missing.update(missing)
            primary = [row for row in result["query_rounds"] if row["vectors"] == prefix * unit]
            require(len(primary) == 2, "Exactly two current-prefix measured passes required")
            result.update(schema_version=1, case=args.case, metric=args.metric, prefix=prefix, layers=1,
                          implementation_variant=implementation,
                          diagnostic_continuation_authorized=implementation == "original-maintenance",
                          quantizer=case_definition["quantizer"], code_bytes=layout["code_bytes"],
                          record_bytes=layout["record_bytes"],
                          primary_query_rounds=primary,
                          build_or_insert_seconds=(result["insertion"]["insert timeSeconds"] if prefix > 1 else
                                                   result["raw_build_fields"]["build timeSeconds"]),
                          maintenance_wait_seconds=sum(result["maintenance"]["barrier_seconds"]),
                          cumulative_distinct_missing_source_ids=len(ever_missing))
            write_json(case / "diagnostic-result.json", result)
            (case / "physical-audit/ever-missing.csv").write_text(
                "source_id\n" + "".join(f"{vid}\n" for vid in sorted(ever_missing)))
            if implementation != "original-maintenance":
                require(result["status"] == "integrity-accepted",
                               "IP repair failed the current-ID/runtime integrity gate; preserve this stage")
            milestone = {"vectors": prefix * unit, "status": result["status"],
                         "missing_source_count": result["physical_coverage"]["missing_source_count"],
                         "result": str(case / "diagnostic-result.json"), "checkpoint": str(previous)}
            if prefix > 1:
                milestone["seed_pruning"] = prune_completed_seed(args.root, prefix, previous)
            if retention == "initial-and-latest":
                if prefix > 2:
                    milestone["predecessor_pruning"] = prune_completed_predecessor(
                        args.root, prefix, previous)
                    manifest["stages"][prefix - 2]["checkpoint_retention"] = "pruned-after-verified-successor"
                milestone["cache_pruning"] = prune_generated_caches(
                    args.root, prefix, base, args.prepared / "query.f32bin", unit, queries)
            manifest["milestones"].append(milestone)
            manifest["stages"][prefix - 1]["status"] = result["status"]
            manifest.update(status="milestone-recorded")
            write_json(args.root / "manifest.json", manifest)
            print(json.dumps({"vectors": prefix * unit, "status": result["status"],
                              "missing": len(missing), "rounds": result["query_rounds"][-2:]}), flush=True)
        manifest.update(status=("completed-through-prefix" if stop_after < prefixes else
                                "completed-diagnostic-single-layer5m"),
                        full_schedule_completed=stop_after == prefixes,
                        cumulative_distinct_missing_source_ids=len(ever_missing))
    except BaseException as error:
        manifest.update(status="hard-failure-or-runner-error", error=str(error))
        if manifest.get("current_prefix"):
            manifest["stages"][manifest["current_prefix"] - 1]["status"] = "failed_hard"
        raise
    finally:
        write_json(args.root / "manifest.json", manifest)


BASELINE_REF = "aac5e1466d5985258e735e1bf30e3313c1407068"


VARIANTS = {
    "original-maintenance": BASELINE_REF,
    "ip-head-fix": "9480ddda68fb06b71f9c8596c6898c0636779cdb",
    "ip-head-opq-encoding-fix": "145255ad1e1eecaeb8f1de68eb63d1d529629c8a",
}


CASE_VARIANTS = {
    "rabitq4-l2": "original-maintenance",
    "rabitq4-ip": "ip-head-fix",
    "opq64-ip": "ip-head-opq-encoding-fix",
}


GIB = 1024**3


CONTINGENCY = 50 * GIB


def environment(root):
    temporary = root / "tmp"
    cache = root / "cache"
    temporary.mkdir()
    cache.mkdir()
    return dict(os.environ, TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary),
                XDG_CACHE_HOME=str(cache), PYTHONPYCACHEPREFIX=str(cache / "python"),
                CMAKE_EXPORT_NO_PACKAGE_REGISTRY="ON")


def git_output(*args, cwd=SOURCE):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def baseline_core():
    archive = subprocess.check_output(
        ["git", "-C", str(SOURCE), "archive", BASELINE_REF, "AnnService"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        return {item.name: hashlib.sha256(stream.extractfile(item).read()).hexdigest()
                for item in stream.getmembers() if item.isfile()}


def build(args):
    root = args.root.resolve()
    require_data_paths((root, SOURCE))
    require(not root.exists(), "Build root must be new; CMake writes Release inside its source tree")
    require(1 <= args.jobs <= 32, "Build jobs must be between 1 and 32")
    implementation = args.implementation or CASE_VARIANTS[args.case]
    ref = git_output("rev-parse", VARIANTS[implementation])
    baseline = baseline_core()
    root.mkdir()
    env = environment(root)
    source, output = root / "source", root / "build"
    status, allowed = IMPLEMENTATIONS[implementation]
    record = {"status": "building", "implementation_variant": implementation,
              "source_commit": ref, "source_root": str(source), "commands": [],
              "harness_source_commit": git_output("rev-parse", "HEAD")}
    try:
        def execute(command, name):
            record["commands"].append(command)
            write_json(root / "build-manifest.json", record)
            execute_process(command, root, name, env, timeout=7200)

        execute(["git", "clone", "--quiet", "--no-hardlinks", "--no-checkout",
                 str(SOURCE), str(source)], "clone")
        execute(["git", "-C", str(source), "checkout", "--quiet", "--detach", ref], "checkout")
        execute(["git", "-C", str(source), "submodule", "update", "--init",
                 "ThirdParty/zstd", "ThirdParty/RaBitQOfficial"], "submodules")
        shutil.copy2(SOURCE / "Test/src/SPFreshTest.cpp", source / "Test/src/SPFreshTest.cpp")
        core = {p.relative_to(source).as_posix(): sha256_file(p)
                for p in (source / "AnnService").rglob("*") if p.is_file()}
        require(len(core) == 204 and set(core) == set(baseline) and
                       {name for name in core if core[name] != baseline[name]} == allowed,
                       "Source variant differs from the measured core changes")
        write_json(root / "core-before.json", {"annservice_files": core})
        write_json(root / "baseline-core.json", {"annservice_files": baseline})
        execute(["cmake", "-S", str(source), "-B", str(output), "-DCMAKE_BUILD_TYPE=Release",
                 "-DFOLLY=OFF", "-DTBB=ON", "-DSPDK=OFF", "-DROCKSDB=OFF",
                 "-DTIKV=OFF", "-DGPU=OFF", "-DRABITQ=ON",
                 "-DCMAKE_EXPORT_PACKAGE_REGISTRY=OFF", "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF"],
                "configure")
        execute(["cmake", "--build", str(output), "--target", "SPTAGLib", "SPTAGTest",
                 "--parallel", str(args.jobs)], "compile")
        binary = source / "Release/SPTAGTest"
        helpers = root / "bin"
        helpers.mkdir()
        for name in ("normalized_prepare", "check_count"):
            command = ["g++", "-std=c++20", "-O3", "-ffp-contract=off", "-DNDEBUG",
                       str(RECIPE / (name + ".cpp")), "-pthread", "-o", str(helpers / name)]
            if name == "check_count":
                command.extend(["-DRABITQ", "-DTBB", "-DNUMA", "-fopenmp",
                                "-I" + str(source / "AnnService"), "-I/usr/include/eigen3",
                                "-I" + str(source / "ThirdParty/RaBitQOfficial/include"),
                                "-L" + str(source / "Release"), "-Wl,-rpath," + str(source / "Release"),
                                "-lSPTAGLib", "-lboost_filesystem", "-ltbb", "-lnuma"])
            execute(command, "compile-" + name)
        require(all(sha256_file(source / name) == sha for name, sha in core.items()),
                       "Core sources changed during the build")
        record.update(status=status, binary_sha256=sha256_file(binary),
                      harness_sha256=sha256_file(source / "Test/src/SPFreshTest.cpp"),
                      baseline_core_sha256=sha256_file(root / "baseline-core.json"),
                      intentional_core_changes=sorted(allowed),
                      link_dependencies=subprocess.check_output(["ldd", str(binary)], text=True),
                      helper_sha256={p.name: sha256_file(p) for p in helpers.iterdir()},
                      helper_source_sha256={
                          name: sha256_file(RECIPE / name) for name in
                          ("normalized_prepare.cpp", "check_count.cpp")},
                      compiler=subprocess.check_output(["g++", "--version"], text=True),
                      cmake=subprocess.check_output(["cmake", "--version"], text=True),
                      submodules=git_output("submodule", "status", cwd=source),
                      completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        record.update(status="build-failed", error=str(error))
        raise
    finally:
        write_json(root / "build-manifest.json", record)


def prepare(args):
    root = args.root.resolve()
    require_data_paths((root, args.base, args.query, args.helper, PROFILE))
    reference = REFERENCE
    require(sha256_file(args.base) == RAW_BASE_SHA and
                   sha256_file(args.query) == RAW_QUERY_SHA, "Wrong original SIFT inputs")
    require(not root.exists(), "Prepared-data root must be new")
    root.mkdir()
    env = environment(root)
    manifest = {"status": "preparing", "initial_count": 1000000, "queries": 10000,
                "maximum_vectors": 5000000, "raw_rerank": False, "model_retrained": False,
                "raw_base_sha256": RAW_BASE_SHA, "raw_query_sha256": RAW_QUERY_SHA,
                "helper_sha256": sha256_file(args.helper), "base_sha256": {}}
    try:
        for label, source, rows in (("base5m", args.base, 5000000), ("query", args.query, 10000)):
            execute_process([str(args.helper.resolve()), "normalize", str(source.resolve()),
                            str(rows), str(root / (label + ".f32bin"))], root,
                           "normalize-" + label, env, timeout=3600)
        for prefix in range(1, 6):
            path = root / f"base{prefix}m.f32bin"
            if prefix < 5:
                prefix_file(root / "base5m.f32bin", path, prefix * 1000000)
            sha = sha256_file(path)
            require(sha == reference["base_sha256"][str(prefix)], "Normalized prefix changed")
            manifest["base_sha256"][str(prefix)] = sha
        query = root / "query.f32bin"
        require(sha256_file(query) == NORMALIZED_QUERY_SHA, "Normalized query bytes changed")
        manifest["query_sha256"] = NORMALIZED_QUERY_SHA
        write_json(root / "stored-norm-audit.json", norm_audit(root / "base5m.f32bin"))
        write_json(root / "query-norm-audit.json", norm_audit(query))
        for metric in ("L2", "InnerProduct"):
            prepared = root / metric
            (prepared / "truth").mkdir(parents=True)
            for name in ["query.f32bin", *(f"base{i}m.f32bin" for i in range(1, 6))]:
                (prepared / name).symlink_to(root / name)
            case = dict(manifest, status="preparing-truth", metric=metric, truth_sha256={})
            write_json(prepared / "manifest.json", case)
            for prefix in range(1, 6):
                truth = prepared / f"truth/truth{prefix}m.bin"
                execute_process([str(args.helper.resolve()), "truth", str(root / f"base{prefix}m.f32bin"),
                                str(query), "32", str(truth), metric], root,
                               f"truth-{metric}-{prefix}m", env, timeout=28800)
                sha = sha256_file(truth)
                require(sha == reference["truth_sha256"][metric][str(prefix)],
                               "Exact stored-Float32 truth differs from the measured reference")
                case["truth_sha256"][str(prefix)] = sha
                write_json(prepared / "manifest.json", case)
            case["status"] = "prepared-normalized-single-layer5m"
            write_json(prepared / "manifest.json", case)
        manifest["status"] = "prepared-normalized-single-layer5m"
    except BaseException as error:
        manifest.update(status="preparation-failed", error=str(error))
        raise
    finally:
        write_json(root / "manifest.json", manifest)


def budget(args):
    require_data_paths((args.root, args.output))
    require(not args.root.exists() and not args.output.exists(),
                   "Budget must name a new run root and a new budget file")
    require(math.isfinite(args.work_gib) and args.work_gib > 0, "Work allowance must be positive")
    additional = math.ceil(args.work_gib * GIB) + CONTINGENCY
    free = shutil.disk_usage(args.root.parent).free
    require(free >= MIN_FREE + additional,
                   "Insufficient space for the work allowance, 50GiB contingency and 250GiB floor")
    write_json(args.output, {
        "status": "feasible", "case": args.case, "output_root": str(args.root.resolve()),
        "stop_after_prefix": args.stop_after_prefix, "retention_mode": args.retain_checkpoints,
        "projected_additional_bytes": additional, "contingency_bytes": CONTINGENCY,
        "floor_bytes": MIN_FREE, "free_bytes_at_review": free,
        "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Caller-specified work allowance, not an automatic peak-space estimate",
    })


def run(args):
    build_root = args.build.resolve()
    build_record = json.loads((build_root / "build-manifest.json").read_text())
    args.binary = Path(build_record["source_root"]) / "Release/SPTAGTest"
    args.checker = build_root / "bin/check_count"
    args.build_manifest = build_root / "build-manifest.json"
    args.implementation = build_record["implementation_variant"]
    args.prepared = args.prepared.resolve() / CASES[args.case]["metric"]
    run_case(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract", help="Convert the downloaded native 10M prefix header")
    extraction.add_argument("--prefix", type=Path, required=True)
    extraction.add_argument("--output", type=Path, required=True)
    builder = commands.add_parser("build", help="Build one isolated, source-pinned implementation")
    builder.add_argument("--root", type=Path, required=True)
    builder.add_argument("--case", choices=tuple(CASES), required=True)
    builder.add_argument("--implementation", choices=tuple(VARIANTS))
    builder.add_argument("--jobs", type=int, default=8)
    data = commands.add_parser("prepare", help="Normalize 5M rows and compute both metrics' five truths")
    for name in ("root", "base", "query", "helper"):
        data.add_argument("--" + name, type=Path, required=True)
    planning = commands.add_parser("budget", help="Record an explicit work allowance and fresh space check")
    planning.add_argument("--output", type=Path, required=True)
    planning.add_argument("--work-gib", type=float, required=True)
    reporting = commands.add_parser("report", help="Summarize local run results as Markdown")
    reporting.add_argument("--run", action="append", required=True, metavar="CASE=DIRECTORY")
    reporting.add_argument("--output", type=Path, required=True)
    runner = commands.add_parser("run", help="Run the shared build/insert/query/reload/audit lifecycle")
    for name in ("build", "prepared", "budget"):
        runner.add_argument("--" + name, type=Path, required=True)
    runner.add_argument("--model", type=Path, required=True,
                        help="External frozen initial-1M model file (SHA-256 checked)")
    for command in (planning, runner):
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--case", choices=tuple(CASES), required=True)
        command.add_argument("--stop-after-prefix", type=int, choices=range(1, 6), required=True)
        command.add_argument("--retain-checkpoints", choices=("all", "initial-and-latest"), default="all")
    args = parser.parse_args()
    if args.command == "extract":
        require_data_paths((args.prefix, args.output))
        extract(args.prefix, args.output, 10000000, truncated=True)
        require(sha256_file(args.output) == RAW_BASE_SHA, "Extracted SIFT prefix hash differs")
    else:
        {"build": build, "prepare": prepare, "budget": budget, "run": run,
         "report": report}[args.command](args)


REPORT_FIELDS = [
    "case", "vectors", "status", "recall5_at20", "qps_round2",
    "build_or_insert_seconds", "included_drain_seconds", "missing_current_ids",
    "mean_ms", "p50_ms", "p95_ms", "p99_ms", "reload_recall5_at20", "reload_qps_round2",
]


def report_rows(name, root):
    require(name in CASES, "Unknown benchmark case")
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["metric"] == CASES[name]["metric"], "Wrong case distance metric")
    stages = {stage["prefix"]: stage["status"] for stage in manifest.get("stages", [])}
    result = []
    for prefix in range(1, 6):
        row = dict.fromkeys(REPORT_FIELDS)
        row.update(case=name, vectors=prefix * 1000000)
        path = root / f"prefix-{prefix}/diagnostic-result.json"
        if not path.exists():
            require(stages.get(prefix) in ("not_run", "failed_hard"),
                           "Missing measured result is not an explicitly unrun or failed stage")
            row["status"] = stages[prefix]
        else:
            data = json.loads(path.read_text())
            require(data["vectors"] == row["vectors"] and not data["metric_issues"] and
                           data["model_sha256"] == CASES[name]["model_sha256"],
                           "Invalid result population, metrics or frozen model")
            primary = [r for r in data["query_rounds"] if r["vectors"] == data["vectors"]]
            reload = data["reload_query_rounds"]
            require(len(primary) == len(reload) == 2, "Expected two primary and two reload rounds")
            latest = primary[1]
            seconds = (data["insertion"]["insert timeSeconds"] if prefix > 1 else
                       data["raw_build_fields"]["build timeSeconds"])
            row.update(status=data["status"], recall5_at20=latest["recall5_at20"],
                       qps_round2=latest["qps"], build_or_insert_seconds=seconds,
                       included_drain_seconds=sum(data["maintenance"]["barrier_seconds"]),
                       missing_current_ids=data["physical_coverage"]["missing_source_count"],
                       mean_ms=latest["meanLatency"], p50_ms=latest["p50"],
                       p95_ms=latest["p95"], p99_ms=latest["p99"],
                       reload_recall5_at20=reload[1]["recall5_at20"], reload_qps_round2=reload[1]["qps"])
        result.append(row)
    return result


def report(args):
    exported, seen = [], set()
    for value in args.run:
        name, separator, directory = value.partition("=")
        require(separator and directory and name not in seen, "Use one CASE=DIRECTORY per run")
        seen.add(name)
        exported.extend(report_rows(name, Path(directory)))

    def cell(row, key):
        value = row[key]
        if value is None or (key == "included_drain_seconds" and row["vectors"] == MILLION):
            return "N/A"
        if key in ("recall5_at20", "reload_recall5_at20"):
            return f"{100 * value:.2f}%"
        if key == "vectors":
            return f"{value // MILLION}M"
        if key not in ("case", "status", "missing_current_ids"):
            return f"{value:,.4f}"
        return str(value).replace("|", "\\|")

    sections = [
        ("Recall, throughput and build/insertion", [
            ("Case", "case"), ("Vectors", "vectors"), ("Status", "status"),
            ("Recall5@20", "recall5_at20"), ("QPS", "qps_round2"),
            ("Build/insert s", "build_or_insert_seconds"), ("Included drain s", "included_drain_seconds"),
            ("Missing IDs", "missing_current_ids"),
        ]),
        ("Query latency and fresh reload", [
            ("Case", "case"), ("Vectors", "vectors"), ("Mean ms", "mean_ms"),
            ("p50 ms", "p50_ms"), ("p95 ms", "p95_ms"), ("p99 ms", "p99_ms"),
            ("Reload recall", "reload_recall5_at20"), ("Reload QPS", "reload_qps_round2"),
        ]),
    ]
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write("# Normalized SIFT evaluation results\n\n")
        stream.write("Query metrics use measured round 2. Build/insert seconds already include "
                     "drain; N/A denotes an unmeasured or inapplicable value.\n")
        for title, columns in sections:
            stream.write(f"\n## {title}\n\n")
            stream.write("| " + " | ".join(label for label, _ in columns) + " |\n")
            stream.write("| " + " | ".join("---" for _ in columns) + " |\n")
            for row in exported:
                stream.write("| " + " | ".join(cell(row, key) for _, key in columns) + " |\n")


if __name__ == "__main__":
    main()
