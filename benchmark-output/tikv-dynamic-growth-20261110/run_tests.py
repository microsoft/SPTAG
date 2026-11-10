import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent.parent
RECIPE = REPOSITORY / "evaluation/2026-11-05-sift10m-tikv-pool16-cq8"
sys.path.insert(0, str(RECIPE))
from deployment import Cluster, ENDPOINTS
from presplit import replication_issues


def main():
    if shutil.disk_usage(ROOT).free < 250 * 1024**3:
        raise RuntimeError("Functional test safety stop: need at least 250 GiB free")
    attempt = ROOT / "attempts" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    attempt.mkdir(parents=True)
    shutil.copyfile(RECIPE / "tikv.toml", attempt / "tikv.toml")
    prefix = "dynamic-" + attempt.name.lower()
    binary = REPOSITORY / "Release/SPTAGTest"
    command = [str(binary), "--run_test=TiKVDynamicLayerIntegrationTest/RealTiKVGrowthAndReload",
               "--log_level=test_suite"]
    environment = os.environ.copy()
    environment.update(
        SPTAG_TIKV_TEST_PD=ENDPOINTS,
        SPTAG_TIKV_TEST_PREFIX=prefix,
        SPTAG_TIKV_TEST_DIRECTORY=str(attempt / "indexes"),
    )
    manifest = {
        "command": command,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "prefix": prefix,
        "pd_addresses": ENDPOINTS,
        "timeout_seconds": 3600,
        "scope": "real three-replica functional test; not the 5M-to-10M benchmark",
    }
    (attempt / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Real TiKV test: {attempt}, prefix={prefix}, timeout=3600s", flush=True)
    cluster = Cluster(attempt, prefix)
    started = time.monotonic()
    result = {"status": "failed"}
    try:
        cluster.start()
        stores = json.loads((attempt / "stores-before.json").read_text())
        store_ids = {row["store"]["id"] for row in stores["stores"]}

        def replicated(data):
            return (data["count"] == len(data["regions"]) and bool(data["regions"])
                    and not replication_issues(data["regions"], store_ids))

        cluster.wait("regions", "regions-before.json", replicated)
        remaining = 3600 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Cluster startup exceeded one hour")
        with (attempt / "run.log").open("w") as log:
            process = subprocess.Popen(command, cwd=REPOSITORY, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            print(f"Cluster ready; test pid={process.pid}", flush=True)
            try:
                returncode = process.wait(timeout=remaining)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        if returncode:
            raise RuntimeError(f"Real TiKV test failed with exit code {returncode}: {attempt / 'run.log'}")
        cluster.wait("regions", "regions-after.json", replicated)
        result["status"] = "passed"
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        result["workload_elapsed_seconds"] = time.monotonic() - started
        try:
            cluster.close()
        except Exception as error:
            result["status"] = "failed"
            result["cleanup_error"] = str(error)
            raise
        finally:
            result["elapsed_seconds_including_cleanup"] = time.monotonic() - started
            (attempt / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
