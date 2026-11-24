import argparse
import hashlib
import json
import pathlib
import shutil
import signal

from run_experiment import PROFILE, REPO, ROOT, GIB, atomic_json, interrupt, run_phase


def thresholds_from_trajectories(growth, deletion):
    pivot = next(row for row in growth if row.get("phase") == "insert-10")
    at_ten = next(row for row in deletion if row.get("phase") == "evaluated-10")
    below_ten = next(row for row in deletion if row.get("phase") == "delete-11")
    final = next(row for row in deletion if row.get("phase") == "complete")
    high = pivot["head_bytes"]
    before = at_ten["compact_estimate_bytes"]
    after = below_ten["compact_estimate_bytes"]
    low = (before + after) // 2
    target = (low + high) // 2
    if not final["compact_estimate_bytes"] <= after < before or not 0 < low < target < high:
        raise ValueError(
            "Matched configuration cannot place a shrink low-watermark below the growth high-watermark "
            f"at the requested counts: high={high}, delete10M={before}, "
            f"delete9.9M={after}, delete9M={final['compact_estimate_bytes']}. "
            "Do not force resizing or change unrelated parameters."
        )
    return {"high": high, "target": target, "low": low}


def main():
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    for value in (args.source_run, args.run_id):
        if pathlib.Path(value).name != value:
            raise ValueError("Run IDs must be single path components")
    source = ROOT / "runs" / args.source_run
    directory = ROOT / "runs" / args.run_id
    directory.mkdir(parents=True, exist_ok=False)
    try:
        if shutil.disk_usage(directory).free < 20 * GIB:
            raise RuntimeError("At least 20 GiB free required for in-place deletion calibration")
        digest = hashlib.sha256(PROFILE.read_bytes()).hexdigest()
        recorded = json.loads((source / "plan.json").read_text())
        if recorded.get("baseline_profile_sha256") != digest:
            raise ValueError("Source index must use the historical matched profile")
        growth = [json.loads(line) for line in (source / "metrics.jsonl").read_text().splitlines()]
        if not any(row.get("phase") == "insert-20" and row.get("live") == 11000000 for row in growth):
            raise ValueError("Source did not reach 11M")
        shutil.copy2(PROFILE, directory / "matched-profile.ini")
        shutil.copy2(REPO / "Test/src/SiftLifecycleTest.cpp", directory / "driver.cpp")
        shutil.copy2(pathlib.Path(__file__), directory / "calibrate_fullscale.py")
        shutil.copy2(ROOT / "run_experiment.py", directory / "runner.py")
        shutil.copy2(source / "metrics.jsonl", directory / "growth-metrics.jsonl")
        shutil.copy2(source / "formal-index/indexloader.ini", directory / "source-indexloader.ini")
        atomic_json(directory / "plan.json", {
            "baseline_profile_sha256": digest,
            "source_run": args.source_run,
            "purpose": "Fixed-topology calibration only; not automatic-resize acceptance",
            "mutates_source_index": True,
            "deletion_seed": 1729,
            "deletion": "11M to 9M in twenty 100K batches; ordinary queries only",
        })
        run_phase(directory, "deletion-calibration",
                  {"high": 10**12, "target": 9 * 10**11, "low": 8 * 10**11},
                  test_case="DeletionCalibration", calibration_index=source / "formal-index")
        deletion = [json.loads(line) for line in (directory / "deletion-metrics.jsonl").read_text().splitlines()]
        thresholds = thresholds_from_trajectories(growth, deletion)
        atomic_json(directory / "frozen-thresholds.json", thresholds)
        atomic_json(directory / "status.json", {"state": "passed", "phase": "calibrated", "thresholds": thresholds})
        print(f"Full-scale calibrated thresholds: {thresholds}", flush=True)
    except BaseException as error:
        status_path = directory / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        status.update({"state": "stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                       "error": str(error)})
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
