import csv
import os
import pathlib
import shutil
import subprocess
import time


ROOT = pathlib.Path(__file__).resolve().parent
BINARY = "/mnt_ssd/qianxi/SPTAG/Release/SPTAGTest"
TIMEOUT_SECONDS = 3600


def main():
    attempt = ROOT / "attempts" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    attempt.mkdir(parents=True)
    shutil.copyfile(ROOT / "benchmark.ini", attempt / "benchmark.ini")
    environment = os.environ.copy()
    environment["BENCHMARK_CONFIG"] = str(attempt / "benchmark.ini")
    environment["BENCHMARK_OUTPUT"] = str(attempt / "output.json")
    environment["OMP_NUM_THREADS"] = "32"
    started = time.monotonic()
    with (attempt / "run.log").open("w") as log, (attempt / "memory.csv").open("w") as memory:
        writer = csv.writer(memory)
        writer.writerow(["elapsed_seconds", "pid", "rss_kib", "peak_rss_kib"])
        process = subprocess.Popen(
            [BINARY, "--run_test=SPFreshTest/BenchmarkFromConfig", "--log_level=test_suite"],
            cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
        )
        print(f"FILEIO experiment started: pid={process.pid}, directory={attempt}, timeout={TIMEOUT_SECONDS}s", flush=True)
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= TIMEOUT_SECONDS:
                    process.kill()
                    process.wait()
                    raise TimeoutError("The FILEIO experiment exceeded one hour")
                status = pathlib.Path(f"/proc/{process.pid}/status")
                if status.exists():
                    values = {}
                    try:
                        text = status.read_text()
                    except FileNotFoundError:
                        if process.poll() is None:
                            raise
                        break
                    for line in text.splitlines():
                        if line.startswith(("VmRSS:", "VmHWM:")):
                            key, value, _ = line.split()
                            values[key] = int(value)
                    if "VmRSS:" in values and "VmHWM:" in values:
                        writer.writerow([round(elapsed, 3), process.pid, values["VmRSS:"], values["VmHWM:"]])
                        memory.flush()
                time.sleep(min(10, max(0, TIMEOUT_SECONDS - elapsed)))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"FILEIO benchmark failed with exit code {process.returncode}; see {attempt / 'run.log'}")
    print(f"FILEIO experiment completed in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
