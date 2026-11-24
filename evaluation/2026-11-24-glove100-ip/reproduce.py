"""GloVe-100 IP at c1 or c1 with L2 OPQ encoding: strict Recall10@10 and Recall10@40."""
import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import time


ROWS, DIMENSION, QUERIES, K = 1183514, 100, 10000, 10
CASES = ("float32-ip", "rabitq4-ip", "opq50-ip")
CHECKS = (4000,)
REPEATS = 5
ANN_REF = "2e081ad32c1eccab72dcb739ad886c310b90f715"
DATA_SHA = "544af1d5e84e112cd4749571dcfd8ca109818a572f850af75a3a09e093a953c4"
CORE_SHA = "41c85b549de9b83d539f6994c07d3ed5cc8ed530b0d3a7289f0fb9fd3d11009c"
OPQ_L2_CORE_SHA = "8fd72392938519fecf5cab33a9836367d1d33246b97c63dda4497496a48619a5"
GIB = 1024**3
ERROR = re.compile(r"\[[45]\] \d{4}-\d{2}-\d{2} ")
WARNING = re.compile(r"\[3\] \d{4}-\d{2}-\d{2} ")
HASH_TABLE_RESIZE = re.compile(
    r"\[4\] \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} "
    r"Hash table is full! Set HashTableExponent to larger value \(default is 2\)\. "
    r"NewHashTableExponent=[1-9]\d* NewPoolSize=[1-9]\d*")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def core_files(source):
    return {p.relative_to(source).as_posix(): sha(p)
            for p in (source / "AnnService").rglob("*") if p.is_file()}


def core_fingerprint(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_core(source):
    files = core_files(source)
    require(len(files) == 204 and core_fingerprint(files) in (CORE_SHA, OPQ_L2_CORE_SHA),
            "Not a reviewed native source snapshot")
    return files


def environment(root, threads=8):
    return dict(os.environ, TMPDIR=str(root / "tmp"), TMP=str(root / "tmp"),
                TEMP=str(root / "tmp"), XDG_CACHE_HOME=str(root / "cache"),
                PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(threads),
                OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS=str(threads),
                PYTHONPATH=os.pathsep.join((str(root / "source/Release"),
                                           str(root / "ann-benchmarks"))))


def budget(root, allocation_gib=224):
    require(isinstance(allocation_gib, int) and allocation_gib > 0, "Invalid workspace budget")
    limit = allocation_gib * GIB
    allocated = int(subprocess.check_output(["du", "-s", "-B1", str(root)], text=True).split()[0])
    free = shutil.disk_usage(root).free
    require(allocated <= limit, f"Workspace exceeds its {allocation_gib}GiB allocation budget")
    require(free >= 300 * GIB + max(0, limit - allocated),
            "Insufficient space above the 250GiB floor and 50GiB contingency")
    return {"allocated_bytes": allocated, "free_bytes": free, "budget_bytes": limit}


def review_log(root, path):
    errors, warnings, resizes = [], [], []
    for line in path.read_text(errors="replace").splitlines():
        if HASH_TABLE_RESIZE.fullmatch(line):
            resizes.append(line)
        elif ERROR.search(line):
            errors.append(line)
        if WARNING.search(line):
            warnings.append(line)
    review = {"errors": errors, "warnings": warnings, "handled_hash_table_resizes": resizes}
    write_json(path.with_suffix(".review.json"), review)
    require(not errors, f"Native errors in {path}; outputs retained")
    return review


def run_step(root, name, command, threads=8, timeout=7200, budget_root=None, budget_gib=224):
    log_path = root / "logs" / (name + ".log")
    status_path = root / "logs" / (name + ".status.json")
    require(not log_path.exists(), f"Refusing to overwrite {log_path}")
    started = time.monotonic()
    record = {"status": "running", "command": command}
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=root, env=environment(root, threads),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record["pid"] = process.pid
        write_json(status_path, record)
        try:
            while True:
                try:
                    process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    require(time.monotonic() - started < timeout, name + " exceeded its time limit")
                    require(shutil.disk_usage(root).free >= 250 * GIB, "Disk safety floor reached")
                    if budget_root is not None:
                        budget(budget_root, budget_gib)
            record.update(status="exited", returncode=process.returncode,
                          elapsed_seconds=time.monotonic() - started)
            require(process.returncode == 0, f"{name} exited {process.returncode}; see {log_path}")
        except BaseException as error:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            record.update(status="failed", returncode=process.returncode, error=str(error))
            raise
        finally:
            write_json(status_path, record)
    review_log(root, log_path)
    print(name, "completed", round(record["elapsed_seconds"], 2), "seconds", flush=True)
    return record["elapsed_seconds"]


def build(root):
    require(not (root / "native-build.json").exists(), "Use a new build workspace")
    source = root / "source"
    core = verify_core(source)
    run_step(root, "native-configure", ["cmake", "-S", str(source), "-B", str(root / "build"),
             "-DCMAKE_BUILD_TYPE=Release", "-DFOLLY=OFF", "-DTBB=ON", "-DRABITQ=ON",
             "-DSPDK=OFF", "-DROCKSDB=OFF", "-DTIKV=OFF", "-DGPU=OFF",
             "-DCMAKE_EXPORT_PACKAGE_REGISTRY=OFF", "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF"])
    run_step(root, "native-compile", ["cmake", "--build", str(root / "build"),
             "--target", "_SPTAG", "quantizer", "--parallel", "8"])
    require(core_files(source) == core, "Core changed during build")
    patch = subprocess.check_output(["git", "-C", str(source), "diff", "--binary", "HEAD"])
    (root / "source.patch").write_bytes(patch)
    write_json(root / "native-build.json", {
        "source_commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "source_patch_sha256": sha(root / "source.patch"), "core_sha256": core,
        "native_sha256": {name: sha(source / "Release" / name)
                         for name in ("_SPTAG.so", "SPTAG.py", "libSPTAGLib.so", "quantizer")},
        "compiler": subprocess.check_output(["g++", "--version"], text=True),
        "submodules": subprocess.check_output(
            ["git", "-C", str(source), "submodule", "status"], text=True),
        "linked_libraries": subprocess.check_output(
            ["ldd", str(source / "Release/_SPTAG.so")], text=True)})


def verify_runtime(root):
    record = json.loads((root / "native-build.json").read_text())
    require(verify_core(root / "source") == record["core_sha256"], "Native sources changed")
    require(all(sha(root / "source/Release" / name) == value
                for name, value in record["native_sha256"].items()), "Native runtime changed")
    ref = subprocess.check_output(
        ["git", "-C", str(root / "ann-benchmarks"), "rev-parse", "HEAD"], text=True).strip()
    require(ref == ANN_REF, "Wrong ANN-Benchmarks reference")


def prepare(root):
    import h5py
    import numpy as np

    require(not (root / "prepared/preparation.json").exists(), "Prepared outputs already exist")
    source = root / "input/glove-100-angular.hdf5"
    require(sha(source) == DATA_SHA, "Wrong official GloVe HDF5")
    record = {"hdf5_sha256": DATA_SHA, "distance": "angular", "dimension": DIMENSION,
              "base_count": ROWS, "query_count": QUERIES, "source_ids": "unchanged HDF5 row order"}
    with h5py.File(source, "r") as dataset:
        require(dataset.attrs["distance"] == "angular", "Wrong official metric")
        for key, count in (("train", ROWS), ("test", QUERIES)):
            values = dataset[key]
            require(values.shape == (count, DIMENSION) and values.dtype == np.float32,
                    "Unexpected official vector layout")
            target = root / "prepared" / (key + ".f32bin")
            minimum, maximum, stored_error = float("inf"), 0.0, 0.0
            with target.open("xb") as output:
                output.write(struct.pack("<ii", count, DIMENSION))
                for start in range(0, count, 65536):
                    block = values[start:start + 65536].astype(np.float64)
                    norms = np.sqrt(np.sum(block * block, axis=1))
                    require(np.isfinite(block).all() and np.isfinite(norms).all() and (norms > 0).all(),
                            "Nonfinite or zero-norm vector; no rows may be silently removed")
                    normalized = (block / norms[:, None]).astype("<f4")
                    stored_norm = np.sqrt(np.sum(normalized.astype(np.float64) ** 2, axis=1))
                    stored_error = max(stored_error, float(np.max(np.abs(stored_norm - 1))))
                    minimum, maximum = min(minimum, float(norms.min())), max(maximum, float(norms.max()))
                    output.write(normalized.tobytes())
            require(stored_error < 1e-6, "Stored normalization exceeds tolerance")
            record[key] = {"sha256": sha(target), "input_norm_min": minimum,
                           "input_norm_max": maximum, "stored_norm_max_error": stored_error,
                           "zero_rows": 0, "nonfinite_rows": 0}
        truth, ids = np.array(dataset["distances"]), np.array(dataset["neighbors"])
        require(truth.shape == ids.shape == (QUERIES, 100), "Wrong official truth shape")
        require(np.isfinite(truth).all() and (np.diff(truth, axis=1) >= -1e-6).all()
                and (ids >= 0).all() and (ids < ROWS).all(), "Malformed official truth")
        sys.path.insert(0, str(root / "ann-benchmarks"))
        from ann_benchmarks.distance import metrics
        maximum_error = 0.0
        for q in range(0, QUERIES, 100):
            query = np.array(dataset["test"][q])
            for rank, vid in enumerate(ids[q, :K]):
                actual = metrics["angular"].distance(query, np.array(dataset["train"][int(vid)]))
                maximum_error = max(maximum_error, abs(float(actual) - float(truth[q, rank])))
        require(maximum_error < 1e-5, "Official truth does not match its raw source vectors")
        record["official_truth_spotcheck_max_error"] = maximum_error
    write_json(root / "prepared/preparation.json", record)
    print(json.dumps(record), flush=True)


def normalized_base(root):
    import numpy as np

    path = root / "prepared/train.f32bin"
    record = json.loads((root / "prepared/preparation.json").read_text())
    require(sha(path) == record["train"]["sha256"], "Prepared training data changed")
    require(path.stat().st_size == 8 + ROWS * DIMENSION * 4, "Wrong normalized base length")
    with path.open("rb") as stream:
        require(struct.unpack("<ii", stream.read(8)) == (ROWS, DIMENSION), "Wrong normalized base shape")
    return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(ROWS, DIMENSION))


def train_opq(root):
    import faiss
    import numpy as np

    values = normalized_base(root)
    faiss.omp_set_num_threads(8)
    index = faiss.index_factory(DIMENSION, "OPQ50,PQ50x8", faiss.METRIC_INNER_PRODUCT)
    rotation = faiss.downcast_VectorTransform(index.chain.at(0))
    pq = faiss.downcast_index(index.index).pq
    record = {"faiss": faiss.__version__, "numpy": np.__version__, "factory": "OPQ50,PQ50x8",
              "faiss_metric": int(index.metric_type),
              "eligible_training_rows": ROWS, "training_metric": "L2", "scoring_metric": "InnerProduct",
              "threads": 8, "opq": {key: getattr(rotation, key)
                  for key in ("niter", "niter_pq", "niter_pq_0", "max_train_points")},
              "pq": {key: getattr(pq.cp, key)
                  for key in ("seed", "niter", "max_points_per_centroid")}}
    started = time.perf_counter()
    index.train(values)
    record["training_seconds"] = time.perf_counter() - started
    matrix = faiss.vector_to_array(rotation.A).reshape(DIMENSION, DIMENSION)
    books = faiss.vector_to_array(pq.centroids).reshape(50, 256, 2)
    require(np.isfinite(matrix).all() and np.isfinite(books).all(), "Nonfinite OPQ model")
    error = float(np.max(np.abs(matrix.astype(np.float64) @ matrix.astype(np.float64).T
                               - np.eye(DIMENSION))))
    require(error < 1e-4, "OPQ rotation is not orthogonal")
    model = root / "models/opq50-ip.bin"
    with model.open("xb") as output:
        output.write(struct.pack("<BBiii", 2, 3, 50, 256, 2))
        output.write(books.astype("<f4").tobytes())
        output.write(struct.pack("<B", 2))
        output.write(matrix.T.astype("<f4").copy().tobytes())
    faiss.write_index(index, str(root / "models/opq50-training.faiss"))
    record.update(model_sha256=sha(model), rotation_orthogonality_max_error=error,
                  native_rotation="FAISS A.T")
    write_json(root / "models/opq50-training.json", record)
    print(json.dumps(record), flush=True)


def train(root, case):
    verify_runtime(root)
    normalized_base(root)
    if case == "float32-ip":
        record = root / "models" / (case + ".json")
        require(not record.exists(), "Float32 preparation record already exists")
        write_json(record, {
            "case": case, "preparation_seconds": 0, "quantization": False,
            "input_sha256": sha(root / "prepared/train.f32bin"),
            "value_type": "Float", "dimension": DIMENSION, "vector_bytes": DIMENSION * 4,
            "record_bytes": DIMENSION * 4 + 5, "training_rows": 0,
            "runtime_metric": "InnerProduct"})
        return
    model, encoded = root / "models" / (case + ".bin"), root / "prepared" / (case + ".u8bin")
    require(not model.exists() and not encoded.exists(), "Model/encoded output already exists")
    started = time.perf_counter()
    if case == "opq50-ip":
        run_step(root, "opq50-training", [sys.executable, str(Path(__file__).resolve()),
                                         "_train-opq", "--root", str(root)])
    kind, width = ("RaBitQQuantizer", 88) if case == "rabitq4-ip" else ("OPQQuantizer", 50)
    run_step(root, case + "-encode", [str(root / "source/Release/quantizer"),
             "-i", str(root / "prepared/train.f32bin"), "-o", str(encoded),
             "-oq", str(model), "-qt", kind, "-qd", "4" if case == "rabitq4-ip" else "50",
             "-ts", str(ROWS), "-v", "Float", "-d", str(DIMENSION),
             "-f", "DEFAULT", "-t", "8", "-norm", "false", "-dd", "InnerProduct"])
    elapsed = time.perf_counter() - started
    with encoded.open("rb") as stream:
        require(struct.unpack("<ii", stream.read(8)) == (ROWS, width), "Wrong stored code shape")
    require(encoded.stat().st_size == 8 + ROWS * width, "Wrong stored code length")
    raw = model.read_bytes()
    if case == "rabitq4-ip":
        offset = raw.index(struct.pack("<I", 0x32464252))
        require(struct.unpack_from("<IIiiiI", raw, offset)[1:] == (6, 100, 128, 4, 0),
                "RaBitQ model must be global 4-bit, 100D input, 128D internal, without renormalization")
    else:
        require(struct.unpack_from("<BBiii", raw) == (2, 3, 50, 256, 2)
                and len(raw) == 142415 and raw[102414] == 2, "Wrong OPQ50 IP model layout")
    write_json(root / "models" / (case + ".json"), {
        "case": case, "preparation_seconds": elapsed, "model_sha256": sha(model),
        "encoded_sha256": sha(encoded), "code_bytes": width, "record_bytes": width + 5,
        "training_rows": ROWS, "threads": 8, "runtime_metric": "InnerProduct"})


def configuration(root, case, width):
    unquantized = case == "float32-ip"
    return {
        "Base": {"DistCalcMethod": "InnerProduct", "IndexAlgoType": "BKT",
                 "ValueType": "Float" if unquantized else "UInt8",
                 "Dim": width, "VectorPath": str(root / "prepared" /
                                               ("train.f32bin" if unquantized else case + ".u8bin")),
                 "IndexDirectory": str(root / "runs" / case / "index"),
                 "QuantizerFilePath": "" if unquantized else "quantizer.bin"},
        "SelectHead": {"isExecute": True, "NumberOfThreads": 32, "SelectHeadType": "BKT",
                       "SelectThreshold": 0, "SplitFactor": 0, "SplitThreshold": 0,
                       "Ratio": 0.2, "ParallelBKTBuild": True},
        "BuildHead": {"isExecute": True, "NumberOfThreads": 32,
                      "AddCountForRebuild": 10000, "MaxCheck": 8192},
        "BuildSSDIndex": {
            "isExecute": True, "BuildSsdIndex": True, "InternalResultNum": 64,
            "SearchInternalResultNum": 64, "NumberOfThreads": 32, "Storage": "FILEIO",
            "PostingPageLimit": 4, "SearchPostingPageLimit": 4, "BufferLength": 4,
            "PostingVectorLimit": 0, "MaxCheck": 8192, "TmpDir": str(root / "tmp" / case),
            "ExcludeHead": False, "ResultNum": K, "SearchThreadNum": 1,
            "Update": True, "SteadyState": True, "InsertThreadNum": 1,
            "AppendThreadNum": 24, "ReassignThreadNum": 0, "DisableReassign": False,
            "ReassignK": 64, "LatencyLimit": 50, "SearchDuringUpdate": True,
            "MergeThreshold": 10, "Sampling": 4, "InPlace": True,
            "StartFileSizeGB": 1, "GrowthFileSizeGB": 1, "MaxFileSizeGB": 64,
            "OneClusterCutMax": True, "ConsistencyCheck": False, "ChecksumCheck": False,
            "ChecksumInRead": False, "AsyncMergeInSearch": False,
            "DeletePercentageForRefine": 0.4, "AsyncAppendQueueSize": 0,
            "AllowZeroReplica": False, "ShareDB": True, "Layers": 1,
            "DistributedVersionMap": False, "SearchCheckVersionMapOnlyLayer0": True}}


def index_build(root, case):
    import SPTAG

    verify_runtime(root)
    record = json.loads((root / "models" / (case + ".json")).read_text())
    unquantized = case == "float32-ip"
    if unquantized:
        require(sha(root / "prepared/train.f32bin") == record["input_sha256"],
                "Frozen Float32 input changed")
    else:
        require(sha(root / "models" / (case + ".bin")) == record["model_sha256"]
                and sha(root / "prepared" / (case + ".u8bin")) == record["encoded_sha256"],
                "Frozen model/codes changed")
    directory = root / "runs" / case
    directory.mkdir()
    (root / "tmp" / case).mkdir()
    width = DIMENSION if unquantized else record["code_bytes"]
    config = configuration(root, case, width)
    write_json(directory / "configuration.json", config)
    index = SPTAG.AnnIndex("SPANN", config["Base"]["ValueType"], width)
    if not unquantized:
        require(index.LoadQuantizer(str(root / "models" / (case + ".bin"))), "Cannot load quantizer")
        index.SetQuantizerADC(False)
    for section, parameters in config.items():
        for key, value in parameters.items():
            index.SetBuildParam(key, str(value).lower() if isinstance(value, bool) else str(value), section)
    metadata = "".join(f"{vid}\n" for vid in range(ROWS)).encode("ascii")
    started = time.perf_counter()
    require(index.BuildSPANNWithMetaData(metadata, ROWS, False, True), "Native SPANN build failed")
    build_seconds = time.perf_counter() - started
    if not unquantized:
        index.SetQuantizerADC(True)
    started = time.perf_counter()
    require(index.Save(str(directory / "index")), "Native save failed")
    save_seconds = time.perf_counter() - started
    del index
    gc.collect()
    write_json(directory / "build.json", {"case": case, "native_build_seconds": build_seconds,
               "save_seconds": save_seconds, "quantization_preparation_seconds": record["preparation_seconds"],
               "native_build_includes_quantizer_training": False, "build_threads": 32,
               "logical_expected_rows": ROWS})


def physical_audit(index, expected, output, record_bytes):
    """Read current records reachable through live heads without modifying the index."""
    require(record_bytes > 5, "Invalid posting record width")
    output.mkdir()
    versions, members = bytearray([254]) * expected, bytearray(expected)
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
    require(columns == label_columns == 1 and heads == label_rows
            and len(head_data) == 8 + 4 * heads and len(labels) == 12 + heads,
            "Unexpected top BKT mapping/deletion layout")
    require(labels[12:].count(1) == deleted, "Top head deletion count mismatch")
    active = {vid for (vid,), label in zip(struct.iter_unpack("<i", head_data[8:]), labels[12:])
              if label != 1}
    mapping = (index / "ssdmapping_0").read_bytes()
    rows, columns = struct.unpack_from("<ii", mapping)
    require(columns == 10 and len(mapping) == 8 + rows * 80, "Unexpected FILEIO map layout")
    seen, seen_any_map = bytearray(expected), bytearray(expected)
    stats, mapped = Counter(), set()
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
    missing = [vid for vid in range(expected) if not seen[vid]]
    with (output / "missing-source-ids.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source_id", "logical_member", "present_in_inactive_posting"))
        writer.writerows((vid, members[vid], seen_any_map[vid]) for vid in missing)
    result = {"expected_source_ids": expected, "logical_members": sum(members),
              "unique_current_reachable_ids": sum(seen), "missing_source_count": len(missing),
              "live_top_heads": len(active), "retired_top_heads": deleted,
              "active_heads_without_posting": len(active - mapped),
              "invalid_version_members": invalid_members, "duplicate_version_members": duplicates,
              "checkpoint_size_mtime_unchanged": True, **dict(stats)}
    result["coverage_complete"] = not (missing or active - mapped or invalid_members or duplicates
                                       or stats["malformed_postings"] or stats["short_reads"])
    write_json(output / "coverage.json", result)
    return result


def audit(root, case, allow_missing_ids=False):
    directory = root / "runs" / case
    index = directory / "index"
    record = json.loads((root / "models" / (case + ".json")).read_text())
    metadata, offsets = (index / "metadata.bin").read_bytes(), (index / "metadataIndex.bin").read_bytes()
    require(struct.unpack_from("<i", offsets)[0] == ROWS and len(offsets) == 4 + (ROWS + 1) * 8,
            "Invalid metadata offset table")
    for vid in range(ROWS):
        start, end = struct.unpack_from("<QQ", offsets, 4 + vid * 8)
        require(metadata[start:end] == f"{vid}\n".encode("ascii"), "Source-ID mapping changed")
    result = physical_audit(
        index, ROWS, directory / "physical-audit", record_bytes=record["record_bytes"])
    require(result["logical_members"] == ROWS and not any(result.get(key, 0) for key in
            ("invalid_id_records", "absent_member_records", "invalid_version_members",
             "duplicate_version_members", "malformed_postings", "short_reads")),
            "Malformed checkpoint; do not benchmark this index")
    if not result["coverage_complete"]:
        print(case, "WARNING: incomplete source-ID coverage:", json.dumps(result), flush=True)
        require(allow_missing_ids, "Coverage incomplete; explicit --allow-missing-ids is required")
    print(case, "current-ID coverage", result["unique_current_reachable_ids"], "/", ROWS, flush=True)


def validate_search(internal_results, checks):
    require(0 < internal_results <= ROWS, "Invalid internal-result budget")
    require(checks and all(0 < check <= ROWS for check in checks)
            and list(checks) == sorted(set(checks)), "MaxCheck points must be positive, unique and increasing")


class SptagIP:
    def __init__(self, root, case, internal_results=64):
        import SPTAG

        validate_search(internal_results, CHECKS)
        self.case = case
        self.index = SPTAG.AnnIndex.Load(str(root / "runs" / case / "index"))
        require(self.index.ReadyToServe(), "Native fresh-process load failed")
        if case != "float32-ip":
            self.index.SetQuantizerADC(True)
        self.index.SetSearchParam("NumberOfThreads", "1", "BuildHead")
        self.index.SetSearchParam("NumberOfThreads", "1", "BuildSSDIndex")
        self.index.SetSearchParam("SearchThreadNum", "1", "BuildSSDIndex")
        self.index.SetSearchParam("SearchInternalResultNum", str(internal_results), "BuildSSDIndex")

    def set_query_arguments(self, check):
        self.index.SetSearchParam("MaxCheck", str(check), "BuildHead")
        self.index.SetSearchParam("MaxCheck", str(check), "BuildSSDIndex")

    def query(self, vector, count):
        import numpy as np
        raw = vector.astype(np.float64)
        query = (raw / np.sqrt(np.sum(raw * raw))).astype(np.float32)
        ids = self.index.Search(query, count)[0]
        require(len(ids) == count and len(set(ids)) == count
                and all(0 <= vid < ROWS for vid in ids), "Invalid or duplicate native result IDs")
        return ids

    def get_additional(self):
        return {}

    def __str__(self):
        return "SPANN-" + self.case


def strict_recall(truth, ids):
    require(len(truth) == len(ids) and len(ids) > 0, "Mismatched or empty query results")
    return sum(len(set(found) & set(expected[:K])) for found, expected in zip(ids, truth)) / (len(ids) * K)


def queries(root, case, reload_only=False, internal_results=64, checks=CHECKS):
    import h5py
    import numpy as np
    from ann_benchmarks.runner import run_individual_query
    validate_search(internal_results, checks)
    verify_runtime(root)
    require(1 in os.sched_getaffinity(0), "CPU 1 is not available")
    os.sched_setaffinity(0, {1})
    with h5py.File(root / "input/glove-100-angular.hdf5", "r") as dataset:
        base, test = np.array(dataset["train"]), np.array(dataset["test"])
        truth_ids = np.array(dataset["neighbors"])[:, :K]
    algorithm = SptagIP(root, case, internal_results)
    directory = root / "runs" / case / ("reload" if reload_only else "queries")
    directory.mkdir()
    points = []
    for check in ((checks[-1],) if reload_only else checks):
        algorithm.set_query_arguments(check)
        native_top10 = None
        for returned in (10, 40):
            passes = []
            for repeat in range(1 if reload_only else REPEATS):
                attrs, results = run_individual_query(algorithm, base, test, "angular", returned, 1, False)
                latencies = np.array([elapsed for elapsed, _ in results])
                ids = np.array([[vid for vid, _ in found] for _, found in results], dtype=np.int32)
                distances = np.array([[distance for _, distance in found] for _, found in results])
                require(ids.shape == distances.shape == (QUERIES, returned) and np.isfinite(distances).all()
                        and np.isfinite(latencies).all() and (latencies > 0).all(), "Malformed measured results")
                if returned == 10:
                    native_top10 = ids.copy()
                else:
                    require(np.array_equal(ids[:, :10], native_top10),
                            "Search40 top-ten prefix differs from Search10")
                result = {"repeat": repeat + 1, "returned_neighbors": returned,
                          "strict_recall_10_at_k": strict_recall(truth_ids, ids),
                          "strict_recall_10_at_10": strict_recall(truth_ids, ids[:, :10]),
                          "qps": 1 / attrs["best_search_time"],
                          "mean_ms": float(latencies.mean() * 1000),
                          "p50_ms": float(np.percentile(latencies, 50) * 1000),
                          "p95_ms": float(np.percentile(latencies, 95) * 1000),
                          "p99_ms": float(np.percentile(latencies, 99) * 1000)}
                with h5py.File(directory / f"maxcheck-{check}-k{returned}-round-{repeat + 1}.hdf5", "x") as output:
                    output.create_dataset("neighbors", data=ids)
                    output.create_dataset("distances", data=distances)
                    output.create_dataset("times", data=latencies)
                    for key, value in result.items():
                        output.attrs[key] = value
                passes.append(result)
                write_json(directory / f"maxcheck-{check}-k{returned}.json",
                           {"maxcheck": check, "returned_neighbors": returned, "passes": passes})
                print(case, "MaxCheck", check, json.dumps(result), flush=True)
            best = max(passes, key=lambda row: row["qps"])
            points.append({"maxcheck": check, "returned_neighbors": returned,
                           "qps": best["qps"], "best_round": best["repeat"],
                           "strict_recall_10_at_k": passes[-1]["strict_recall_10_at_k"],
                           "strict_recall_10_at_10": passes[-1]["strict_recall_10_at_10"],
                           "passes": passes})
    del algorithm
    gc.collect()
    write_json(directory / "summary.json", {"case": case, "cpu_affinity": [1],
        "ann_benchmarks_commit": ANN_REF, "truth_neighbors": K, "returned_neighbors": [10, 40],
        "query_count": QUERIES, "runs_per_return_count": 1 if reload_only else REPEATS,
        "raw_rerank": False, "search_internal_results": internal_results,
        "recall_metric": "Strict top-ten source-ID intersection; denominator 100000",
        "top10_prefix_matches_search10": True,
        "qps_selection": "fastest pass per return count", "recall_selection": "last pass",
        "points": points})


def run(root, selected_case=None, internal_results=64, checks=CHECKS, allow_missing_ids=False):
    require(selected_case is None or selected_case in CASES, "Unknown benchmark case")
    validate_search(internal_results, checks)
    cases = CASES if selected_case is None else (selected_case,)
    allocation_gib = 224 if selected_case is None else 72
    verify_runtime(root)
    require(not (root / "run-status.json").exists(), "Existing run state; use a new workspace")
    script_path = Path(__file__).resolve()
    record = {"status": "running", "pid": os.getpid(), "cases": list(cases), "completed": [],
              "budget": budget(root, allocation_gib), "runner_sha256": sha(script_path),
              "native_core_sha256": core_fingerprint(core_files(root / "source")),
              "allow_missing_ids": allow_missing_ids, "search_internal_results": internal_results,
              "max_checks": list(checks)}
    write_json(root / "run-status.json", record)
    script = str(script_path)
    try:
        for case in cases:
            train(root, case)
            budget(root, allocation_gib)
            run_step(root, case + "-index", [sys.executable, script, "_index",
                     "--root", str(root), "--case", case], threads=32,
                     budget_root=root, budget_gib=allocation_gib)
            audit(root, case, allow_missing_ids)
        for case in cases:
            budget(root, allocation_gib)
            search_args = ["--search-internal-results", str(internal_results),
                           "--max-checks", *(str(check) for check in checks)]
            run_step(root, case + "-queries", [sys.executable, script, "_queries",
                     "--root", str(root), "--case", case, *search_args], threads=1)
            run_step(root, case + "-reload", [sys.executable, script, "_reload",
                     "--root", str(root), "--case", case, *search_args], threads=1)
            record["completed"].append(case)
            write_json(root / "run-status.json", record)
        verify_runtime(root)
        require(sha(script_path) == record["runner_sha256"], "Measured runner changed during execution")
        record.update(status="complete", final_budget=budget(root, allocation_gib))
    except BaseException as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        write_json(root / "run-status.json", record)


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"Interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "prepare", "run", "_train-opq",
                                           "_index", "_queries", "_reload"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--allow-missing-ids", action="store_true",
                        help="Measure the existing truncation policy even if coverage is incomplete; record all missing IDs")
    parser.add_argument("--search-internal-results", type=int, default=64)
    parser.add_argument("--max-checks", type=int, nargs="+", default=CHECKS)
    args = parser.parse_args()
    root = args.root.resolve()
    require(sys.platform == "linux" and root.is_relative_to("/data") and root != Path("/data"),
            "Use a new owned workspace under /data on Linux")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    require((root / "source").is_dir() and (root / "ann-benchmarks").is_dir(),
            "Prepare the isolated source checkouts described in README.md")
    for directory in ("tmp", "cache", "logs", "input", "prepared", "models", "runs"):
        (root / directory).mkdir(exist_ok=True)
    if args.command in ("_index", "_queries", "_reload"):
        require(args.case is not None, "A case is required")
    functions = {"build": build, "prepare": prepare, "_train-opq": train_opq}
    if args.command in functions:
        functions[args.command](root)
    elif args.command == "run":
        run(root, args.case, args.search_internal_results, args.max_checks, args.allow_missing_ids)
    elif args.command == "_index":
        index_build(root, args.case)
    else:
        queries(root, args.case, args.command == "_reload", args.search_internal_results, args.max_checks)


if __name__ == "__main__":
    main()
