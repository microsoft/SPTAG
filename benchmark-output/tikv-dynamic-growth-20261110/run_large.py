import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent.parent
RECIPE = REPOSITORY / "evaluation/2026-11-05-sift10m-tikv-pool16-cq8"
TRUTH = ROOT.parent / "sift10m-fileio-dynamic-5m-1m-20261109" / (
    "perftest_batchtruth.L2.UInt8_5000000_128_1000_5_1000000_0_5")
sys.path.insert(0, str(RECIPE))
from deployment import Cluster, ENDPOINTS
from presplit import prepare_cluster, replication_issues


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def build_checker(build, output):
    commands_file = build / "compile_commands.json"
    if not commands_file.exists():
        raise RuntimeError("Configure the build with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
    entry = next(item for item in json.loads(commands_file.read_text())
                 if item["file"].endswith("/TiKVIOTest.cpp"))
    command = shlex.split(entry["command"])
    obj = output.with_suffix(".o")
    command[command.index("-o") + 1] = str(obj)
    command[command.index("-c") + 1] = str(ROOT / "check_count.cpp")
    subprocess.run(command, cwd=entry["directory"], check=True, timeout=300)
    commands = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", "SPTAGTest"], text=True)
    link = shlex.split(commands.splitlines()[-1])
    start = link.index(command[0])
    end = link.index("&&", start) if "&&" in link[start:] else len(link)
    link = [item for item in link[start:end] if not item.endswith(".o")]
    link.insert(1, str(obj))
    link[link.index("-o") + 1] = str(output)
    subprocess.run(link, cwd=build, check=True, timeout=300)
    obj.unlink()


def write_truth(path, first, last):
    data = TRUTH.read_bytes()
    if len(data) != 240008 or struct.unpack_from("<ii", data) != (12000, 5):
        raise ValueError("Unexpected six-prefix exact-truth shape")
    block = 1000 * 5 * 4
    with path.open("wb") as stream:
        stream.write(struct.pack("<ii", 4000, 5))
        for offset in (8, 8 + 6 * block):
            for prefix in (first, last):
                stream.write(data[offset + prefix * block:offset + (prefix + 1) * block])


def run_process(command, directory, environment, logfile, deadline):
    with logfile.open("w") as log, logfile.with_suffix(".memory.csv").open("w") as memory, logfile.open() as monitor:
        def ensure_real_input():
            if any("Using random generation" in line for line in monitor):
                raise RuntimeError(f"Refusing random-data fallback: {logfile}")

        memory.write("elapsed_seconds,pid,rss_kib,hwm_kib\n")
        started = time.monotonic()
        process = subprocess.Popen(command, cwd=directory, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        print(f"Started pid={process.pid}: {logfile}", flush=True)
        try:
            while process.poll() is None:
                ensure_real_input()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"One-hour phase deadline exceeded: {logfile}")
                if shutil.disk_usage(ROOT).free < 250 * 1024**3:
                    raise RuntimeError("Disk safety stop below 250 GiB")
                try:
                    lines = Path(f"/proc/{process.pid}/status").read_text().splitlines()
                except FileNotFoundError:
                    if process.poll() is None:
                        raise
                    break
                values = {line.split()[0]: int(line.split()[1])
                          for line in lines if line.startswith(("VmRSS:", "VmHWM:"))}
                if "VmRSS:" in values and "VmHWM:" in values:
                    memory.write(f"{time.monotonic()-started:.3f},{process.pid},"
                                 f"{values['VmRSS:']},{values['VmHWM:']}\n")
                    memory.flush()
                time.sleep(min(10, remaining))
            ensure_real_input()
            if process.returncode:
                raise RuntimeError(f"Exit {process.returncode}: {logfile}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    if shutil.disk_usage(ROOT).free < 500 * 1024**3:
        raise RuntimeError("Need at least 500 GiB free before large-run startup")
    attempt = ROOT / "large-attempts" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    attempt.mkdir(parents=True)
    checker = attempt / "check_count"
    build_checker(args.build_dir.resolve(), checker)
    shutil.copyfile(RECIPE / "tikv.toml", attempt / "tikv.toml")
    binary = REPOSITORY / "Release/SPTAGTest"
    prefix = "dynamic5m-" + attempt.name.lower()
    status = {
        "status": "starting",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "prefix": prefix,
        "phase_timeout_seconds": 3600,
        "automatic_resume_after_interrupted_writes": False,
        "completed_phases": [],
    }
    save(attempt / "status.json", status)
    print(f"Large TiKV run: {attempt}, prefix={prefix}", flush=True)
    cluster = Cluster(attempt, prefix)

    def interrupt(signum, frame):
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    started = time.monotonic()
    try:
        signal.alarm(3600)
        cluster.start()
        prepare_cluster(attempt, prefix, 1, 10000000, 128)
        signal.alarm(0)
        current_index = attempt / "index"
        for phase in range(7):
            directory = attempt / f"phase-{phase}"
            directory.mkdir()
            first = max(0, min(5, phase - 1))
            last = min(5, phase)
            base = 5000000 + first * 1000000
            inserted = 1000000 if 1 <= phase <= 5 else 0
            truth = directory / f"perftest_batchtruth.L2.UInt8_{base}_128_1000_5_{inserted}_0_1"
            write_truth(truth, first, last)
            config = configparser.ConfigParser()
            config.optionxform = str
            config.read(ROOT / "large.ini")
            config["Benchmark"].update(
                IndexPath=str(current_index), BaseVectorCount=str(base),
                InsertVectorCount=str(inserted), TruthPath=str(truth))
            config["BuildSSDIndex"]["TiKVKeyPrefix"] = prefix
            with (directory / "benchmark.ini").open("w") as stream:
                config.write(stream, space_around_delimiters=False)
            environment = os.environ.copy()
            environment.update(BENCHMARK_CONFIG=str(directory / "benchmark.ini"),
                               BENCHMARK_OUTPUT=str(directory / "output.json"),
                               OMP_NUM_THREADS="32")
            status.update(status="running", current_phase=phase)
            save(attempt / "status.json", status)
            deadline = time.monotonic() + 3600
            run_process([str(binary), "--run_test=SPFreshTest/BenchmarkFromConfig",
                         "--log_level=test_suite"], directory, environment,
                        directory / "run.log", deadline)
            output = json.loads((directory / "output.json").read_text())
            results = output["results"]
            query = results["benchmark0_query_before_insert"]
            if inserted:
                batch = results["benchmark1_insert"]["batch_1"]
                if batch["inserted"] != inserted or batch["concurrentQueryErrors"] != 0:
                    raise RuntimeError("Inserted count or concurrent-query errors mismatch")
                if batch["concurrentQueryCount"] <= 0:
                    raise RuntimeError("No insertion-time queries completed")
                query = batch["search"]
                current_index = Path(str(current_index) + "_0")
            layers = query["postingLayerCount"]
            if query["recall"]["recallAtK"] < 0.95:
                raise RuntimeError("Recall5@5 below 0.95")
            if phase == 0 and layers != 1 or phase >= 5 and layers != 2:
                raise RuntimeError("Expected initial one layer and final two layers")
            run_process([str(checker), str(current_index), "5000000", str(base + inserted),
                         "1000000", str(layers)], directory, environment,
                        directory / "reload-count.log", deadline)
            status["completed_phases"].append({
                "phase": phase, "vectors": base + inserted, "layers": layers,
                "recall5": query["recall"]["recallAtK"], "index": str(current_index),
            })
            save(attempt / "status.json", status)
        store_ids = {row["store"]["id"] for row in
                     json.loads((attempt / "stores-before.json").read_text())["stores"]}
        cluster.wait("regions", "regions-after.json", lambda data:
                     bool(data["regions"]) and not replication_issues(data["regions"], store_ids))
        status["status"] = "passed"
    except Exception as error:
        status.update(status="failed", error=str(error),
                      warning="Do not resume an interrupted write phase from local files alone: TiKV RawKV is shared.")
        raise
    finally:
        signal.alarm(0)
        try:
            cluster.close()
            reservations = []
            for node in (1, 2, 3):
                path = attempt / "cluster" / f"tikv{node}" / "tikv" / "space_placeholder_file"
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise RuntimeError(f"Unexpected owned reservation file type: {path}")
                reservations.append({"path": str(path), "bytes": info.st_size})
            if reservations:
                subprocess.run(["sudo", "-n", "rm", "--",
                                *(item["path"] for item in reservations)], check=True)
            status["released_owned_reservations"] = reservations
        except Exception as error:
            status.update(status="failed", cleanup_error=str(error))
            raise
        finally:
            status["elapsed_seconds_including_cleanup"] = time.monotonic() - started
            save(attempt / "status.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
