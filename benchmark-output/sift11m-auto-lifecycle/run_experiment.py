import argparse
import csv
import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import time


ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent.parent
PROFILE = REPO / "evaluation/2026-11-09-sift10m-fileio-layer1-matched/saved-indexloader.ini"
GIB = 1024**3


def atomic_json(path, value):
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    partial.replace(path)


def interrupt(signum, frame):
    raise InterruptedError(f"Experiment interrupted by signal {signum}")


def run_phase(directory, mode, thresholds, test_case="AutomaticLifecycle", calibration_index=None,
              pivot=10_000_000, merge_threshold=10):
    output = directory / f"{mode}-index"
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "80",
        "SPTAG_SIFT_PROFILE": str(directory / "matched-profile.ini"),
        "SPTAG_SIFT_MODE": mode,
        "SPTAG_SIFT_OUTPUT": str(output),
        "SPTAG_SIFT_VECTORS": str(ROOT / "input/sift-first-11m.u8bin"),
        "SPTAG_SIFT_QUERIES": str(ROOT / "input/query.10k.u8bin"),
        "SPTAG_SIFT_HIGH": str(thresholds["high"]),
        "SPTAG_SIFT_TARGET": str(thresholds["target"]),
        "SPTAG_SIFT_LOW": str(thresholds["low"]),
        "SPTAG_SIFT_PIVOT": str(pivot),
        "SPTAG_SIFT_MERGE_THRESHOLD": str(merge_threshold),
    })
    if calibration_index is not None:
        environment["SPTAG_SIFT_CALIBRATION_INDEX"] = str(calibration_index)
    libraries = pathlib.Path.home() / ".local/sptag-deps/usr/lib/x86_64-linux-gnu"
    environment["LD_LIBRARY_PATH"] = str(libraries) + ":" + environment.get("LD_LIBRARY_PATH", "")
    command = [str(REPO / "Release/SPTAGTest"), f"--run_test=SiftLifecycleTest/{test_case}",
               "--log_level=test_suite", "--color_output=no"]
    disk_floor = (20 if pivot == 16384 else 30) * GIB
    started = time.monotonic()
    with (directory / f"{mode}.log").open("w") as log, (directory / f"{mode}-resources.csv").open("w") as memory:
        writer = csv.writer(memory)
        writer.writerow(["elapsed_seconds", "rss_kib", "peak_rss_kib", "disk_free_bytes"])
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
        atomic_json(directory / "status.json", {"phase": mode, "pid": process.pid, "state": "running",
                                              "thresholds": thresholds, "started_unix": time.time()})
        print(f"Started {mode}: pid={process.pid}, log={log.name}", flush=True)
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                free = shutil.disk_usage(directory).free
                if free < disk_floor:
                    raise RuntimeError(
                        f"Stopping experiment: disk free {free / GIB:.2f} GiB below "
                        f"{disk_floor / GIB:.0f} GiB guard"
                    )
                deadline = 600 if pivot == 16384 else (2 if mode == "pilot" else 8) * 3600
                if elapsed > deadline:
                    raise TimeoutError(f"{mode} exceeded its wall-clock deadline")
                status = pathlib.Path(f"/proc/{process.pid}/status")
                try:
                    text = status.read_text()
                except FileNotFoundError:
                    if process.poll() is None:
                        raise
                    break
                fields = {line.split(":")[0]: line.split()[1] for line in text.splitlines()
                          if line.startswith(("VmRSS:", "VmHWM:"))}
                if int(fields.get("VmRSS", 0)) * 1024 > 64 * GIB:
                    raise RuntimeError("Stopping experiment: process RSS exceeded 64 GiB")
                writer.writerow([round(elapsed, 2), fields.get("VmRSS", ""), fields.get("VmHWM", ""), free])
                memory.flush()
                time.sleep(1)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if process.returncode:
            raise RuntimeError(f"{mode} failed with exit={process.returncode}; see {log.name}")
    return output


def thresholds_from_growth(snapshots, pivot):
    def measurement(phase, live):
        matches = [row for row in snapshots if row.get("phase") == phase]
        if len(matches) != 1 or matches[0].get("live") != live or matches[0].get("layers") != 1:
            raise ValueError(f"Growth calibration requires a single-layer {phase} measurement at {live}")
        return matches[0]["head_bytes"]

    initial = measurement("initial", 4096 if pivot == 16384 else pivot * 9 // 10)
    high = measurement("insert-12" if pivot == 16384 else "insert-10", pivot)
    final = measurement("complete", 32768 if pivot == 16384 else pivot * 11 // 10)
    if not 0 < initial < high < final:
        raise ValueError(f"Growth calibration is not increasing: initial={initial}, pivot={high}, final={final}")
    thresholds = {"high": high, "target": high * 98 // 100, "low": high * 94 // 100}
    if not 0 < thresholds["low"] < thresholds["target"] < high:
        raise ValueError("Growth calibration cannot produce separated memory thresholds")
    return thresholds


def main():
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    calibration_options = parser.add_mutually_exclusive_group()
    calibration_options.add_argument("--calibration-run")
    calibration_options.add_argument("--fullscale-calibration-run")
    calibration_options.add_argument("--growth-calibration", action="store_true",
                                     help="Measure the actual chosen scale before a fresh lifecycle")
    parser.add_argument("--pivot-count", type=int, choices=(16384, 5_000_000, 10_000_000), default=10_000_000)
    parser.add_argument("--merge-threshold", type=int, choices=(10, 24, 32, 48), default=10)
    parser.add_argument("--merge-pilot-only", action="store_true",
                        help="Calibrate and test the ordinary-merge lifecycle at 1/10 chosen scale")
    parser.add_argument("--discard-growth-index", action="store_true",
                        help="Remove only this run's successful growth-calibration index before formal build")
    args = parser.parse_args()
    if args.merge_pilot_only and not args.growth_calibration:
        raise ValueError("merge-pilot-only requires growth-calibration")
    if args.merge_pilot_only and args.pivot_count == 16384:
        raise ValueError("The small workload already uses full-size calibration; do not scale it again")
    if args.discard_growth_index and not args.growth_calibration:
        raise ValueError("discard-growth-index requires growth-calibration")
    if args.pivot_count != 10_000_000 or args.merge_threshold != 10:
        if not args.growth_calibration:
            raise ValueError("New scales or merge thresholds require fresh growth calibration")
    if pathlib.Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be a single path component")
    directory = ROOT / "runs" / args.run_id
    directory.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(REPO / "Test/src/SiftLifecycleTest.cpp", directory / "driver.cpp")
        shutil.copy2(pathlib.Path(__file__), directory / "runner.py")
        shutil.copy2(ROOT / "test_profile.py", directory / "test_profile.py")
        shutil.copy2(PROFILE, directory / "matched-profile.ini")
        (directory / "implementation.patch").write_bytes(subprocess.check_output(
            ["git", "--no-pager", "diff", "HEAD", "--", "AnnService/inc/Core/SPANN",
             "AnnService/src/Core/SPANN/SPANNIndex.cpp", "Test/src/FileIOLayerTest.cpp"], cwd=REPO))
        profile_hash = hashlib.sha256(PROFILE.read_bytes()).hexdigest()
        required_free = (80 if args.pivot_count == 5_000_000 else 100) if args.growth_calibration else 35
        if args.pivot_count == 16384:
            required_free = 24
        if args.merge_pilot_only:
            required_free = 50
        if shutil.disk_usage(directory).free < required_free * GIB:
            raise RuntimeError(f"At least {required_free} GiB free required at startup")
        if args.growth_calibration and not args.discard_growth_index:
            doubled_free = 2 * required_free - 20
            if shutil.disk_usage(directory).free < doubled_free * GIB:
                raise RuntimeError(f"Keeping both new indexes requires at least {doubled_free} GiB free")
        for path, count in [(ROOT / "input/sift-first-11m.u8bin", 11_000_000),
                            (ROOT / "input/query.10k.u8bin", 10_000)]:
            if path.stat().st_size != 8 + count * 128:
                raise ValueError(f"Dataset not prepared: {path}")
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        workload_pivot = args.pivot_count // 10 if args.merge_pilot_only else args.pivot_count
        small = args.pivot_count == 16384
        initial = 4096 if small else workload_pivot * 9 // 10
        total = 32768 if small else workload_pivot * 11 // 10
        final = 1024 if small else initial
        batch = 1024 if small else workload_pivot // 100
        atomic_json(directory / "plan.json", {
            "source_commit": source, "driver": "SiftLifecycleTest/AutomaticLifecycle",
            "binary_sha256": hashlib.sha256((REPO / "Release/SPTAGTest").read_bytes()).hexdigest(),
            "initial": initial, "total": total, "final": final,
            "pivot": workload_pivot, "batch": batch,
            "insert_batches": (total - initial) // batch,
            "delete_batches": (total - final) // batch, "delete_seed": 1729,
            "deletion": "fixed-seed shuffle of all peak IDs, remove total-final IDs",
            "storage": "FILEIO", "data": json.loads((ROOT / "data.json").read_text()),
            "minimum_recall_at_5": 0.95, "exact_queries": 10000, "top_k": 5,
            "search_threads": 32, "insert_threads": 16, "build_threads": 80,
            "search_during_insert_threads": 0,
            "internal_result_num": 64, "search_internal_result_num": 64,
            "baseline_profile_sha256": profile_hash,
            "baseline_profile": str(PROFILE.relative_to(REPO)),
            "index_configuration_override": {"BuildSSDIndex.AsyncMergeInSearch": "true",
                                             "BuildSSDIndex.MergeThreshold": str(args.merge_threshold)},
            "merge_policy": "existing MergeAsync/MergePostings with reassignment; no dedicated compaction",
            "small_workload_overrides": ({"GrowthFileSizeGB": 1, "LayerResizeCooldownSeconds": 1,
                                          "LayerResizeTimeoutSeconds": 60,
                                          "maintenance_timeout_seconds": 60,
                                          "convergence_timeout_seconds": 180} if small else {}),
            "merge_pilot_only": args.merge_pilot_only,
            "calibration_run": args.calibration_run,
            "fullscale_calibration_run": args.fullscale_calibration_run,
            "growth_calibration": args.growth_calibration,
            "discard_growth_index": args.discard_growth_index,
            "trigger": "head-memory only; calibrated thresholds frozen before formal build",
            "calibration": ("same acceptance scale growth; high=measured pivot head bytes; target=0.98*high; "
                            "low=0.94*high (deletion-side hypothesis)" if args.growth_calibration else
                            "1/10 scale; high=10*pilot head bytes at 1M; target=0.98*high; low=0.94*high"),
            "pilot_timeout_seconds": 600 if small else 7200,
            "formal_timeout_seconds": 600 if small else 28800,
            "disk_stop_bytes": (20 if small else 30) * GIB,
            "disk_monitor_interval_seconds": 1,
        })
        if args.growth_calibration:
            growth_mode = "pilot" if args.merge_pilot_only else "growth-calibration"
            growth_index = run_phase(directory, growth_mode,
                                     {"high": 10**12, "target": 9 * 10**11, "low": 8 * 10**11},
                                     pivot=args.pivot_count, merge_threshold=args.merge_threshold)
            metrics_name = "pilot-metrics.jsonl" if args.merge_pilot_only else "growth-metrics.jsonl"
            snapshots = [json.loads(line) for line in (directory / metrics_name).read_text().splitlines()]
            thresholds = thresholds_from_growth(snapshots, args.pivot_count // 10 if args.merge_pilot_only else args.pivot_count)
            atomic_json(directory / "frozen-thresholds.json", thresholds)
            shutil.copy2(growth_index / "indexloader.ini", directory / "growth-indexloader.ini")
            shutil.copy2(directory / "configured.ini", directory / "growth-configured.ini")
            if args.discard_growth_index:
                expected = directory / f"{growth_mode}-index"
                if growth_index != expected or growth_index.is_symlink() or growth_index.resolve() != expected.absolute():
                    raise ValueError("Refusing to clean an unexpected growth calibration index")
                shutil.rmtree(growth_index)
                print("Removed this run's temporary growth index; retained calibration measurements", flush=True)
            if shutil.disk_usage(directory).free < required_free * GIB:
                raise RuntimeError(f"At least {required_free} GiB free required before formal build")
            print(f"Frozen measured-growth thresholds: {thresholds}", flush=True)
            run_phase(directory, "merge-pilot" if args.merge_pilot_only else "formal", thresholds,
                      pivot=args.pivot_count, merge_threshold=args.merge_threshold)
            atomic_json(directory / "status.json", {"state": "passed", "phase": "complete", "thresholds": thresholds})
            return
        if args.fullscale_calibration_run:
            if pathlib.Path(args.fullscale_calibration_run).name != args.fullscale_calibration_run:
                raise ValueError("fullscale-calibration-run must be a single path component")
            calibration = ROOT / "runs" / args.fullscale_calibration_run
            recorded = json.loads((calibration / "plan.json").read_text())
            completed = json.loads((calibration / "status.json").read_text())
            if recorded.get("baseline_profile_sha256") != profile_hash or completed.get("state") != "passed":
                raise ValueError("Full-scale calibration must pass with the matched profile")
            thresholds = json.loads((calibration / "frozen-thresholds.json").read_text())
            if not 0 < thresholds["low"] < thresholds["target"] < thresholds["high"]:
                raise ValueError("Invalid full-scale calibration thresholds")
            atomic_json(directory / "frozen-thresholds.json", thresholds)
            print(f"Frozen full-scale thresholds: {thresholds}", flush=True)
            run_phase(directory, "formal", thresholds)
            atomic_json(directory / "status.json", {"state": "passed", "phase": "complete", "thresholds": thresholds})
            return
        calibration = directory
        if args.calibration_run:
            if pathlib.Path(args.calibration_run).name != args.calibration_run:
                raise ValueError("calibration-run must be a single path component")
            calibration = ROOT / "runs" / args.calibration_run
            recorded = json.loads((calibration / "plan.json").read_text())
            if recorded.get("baseline_profile_sha256") != profile_hash:
                raise ValueError("Calibration profile differs from matched configuration; recalibrate from scratch")
        else:
            run_phase(directory, "pilot", {"high": 10**12, "target": 9 * 10**11, "low": 8 * 10**11})
        snapshots = [json.loads(line) for line in (calibration / "pilot-metrics.jsonl").read_text().splitlines()]
        if not any(row.get("phase") == "complete" and row.get("live") == 1100000 and row.get("layers") == 1
                   for row in snapshots):
            raise RuntimeError("The source pilot did not complete its single-layer 1.1M calibration")
        pivot = next(row for row in snapshots if row.get("phase") == "insert-10")
        initial = next(row for row in snapshots if row.get("phase") == "initial")
        high = pivot["head_bytes"] * 10
        if initial["head_bytes"] * 10 >= high:
            raise RuntimeError("Pilot head growth is non-increasing; threshold calibration is not usable")
        thresholds = {"high": high, "target": high * 98 // 100, "low": high * 94 // 100}
        if args.calibration_run:
            shutil.copy2(calibration / "pilot-metrics.jsonl", directory / "pilot-metrics.jsonl")
        atomic_json(directory / "frozen-thresholds.json", thresholds)
        print(f"Frozen formal thresholds: {thresholds}", flush=True)
        run_phase(directory, "formal", thresholds)
        atomic_json(directory / "status.json", {"state": "passed", "phase": "complete", "thresholds": thresholds})
    except BaseException as error:
        status_path = directory / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        status.update({"state": "stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                       "error": str(error), "ended_unix": time.time()})
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
