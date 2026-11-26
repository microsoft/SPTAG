import configparser
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import unittest

import run_large


class RunnerTests(unittest.TestCase):
    def test_exact_truth_slices(self):
        source = run_large.TRUTH.read_bytes()
        block = 1000 * 5 * 4
        with tempfile.TemporaryDirectory() as scratch:
            for phase in range(7):
                first, last = max(0, min(5, phase - 1)), min(5, phase)
                path = Path(scratch) / f"{phase}.truth"
                run_large.write_truth(path, first, last)
                expected = struct.pack("<ii", 4000, 5) + b"".join(
                    source[offset + prefix * block:offset + (prefix + 1) * block]
                    for offset in (8, 8 + 6 * block) for prefix in (first, last))
                self.assertEqual(path.read_bytes(), expected)

    def test_ini_paths_have_no_leading_whitespace(self):
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(run_large.ROOT / "large.ini")
        stream = io.StringIO()
        config.write(stream, space_around_delimiters=False)
        line = next(line for line in stream.getvalue().splitlines() if line.startswith("VectorPath="))
        self.assertEqual(line.split("=", 1)[1], config["Benchmark"]["VectorPath"])

    def test_random_fallback_is_rejected_on_successful_exit(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            with self.assertRaisesRegex(RuntimeError, "Refusing random-data fallback"):
                run_large.run_process(
                    [sys.executable, "-c", "print('Using random generation', flush=True)"],
                    directory, os.environ.copy(), directory / "run.log", time.monotonic() + 2)

    def test_benchmark_json_with_and_without_insertions(self):
        binary = run_large.REPOSITORY / "Release/SPTAGTest"
        with tempfile.TemporaryDirectory(prefix="serializer-", dir=run_large.ROOT) as scratch:
            for inserted in (0, 1000):
                with self.subTest(inserted=inserted):
                    directory = Path(scratch) / str(inserted)
                    directory.mkdir()
                    config = configparser.ConfigParser()
                    config.optionxform = str
                    config.read(run_large.ROOT / "large.ini")
                    config["Benchmark"].update(
                        BaseVectorCount="5000", InsertVectorCount=str(inserted),
                        TruthPath="none", IndexPath=str(directory / "index"),
                        NumQueries="100", NumSearchThreads="2", NumInsertThreads="2",
                        NumBuildThreads="4", NumSearchDuringInsertThreads="0")
                    config["SelectHead"]["NumberOfThreads"] = "4"
                    config["BuildHead"]["NumberOfThreads"] = "4"
                    config["BuildSSDIndex"].update(
                        Storage="FILEIO", NumberOfThreads="4", AppendThreadNum="2",
                        DistributedVersionMap="false", EnableDynamicLayers="false")
                    with (directory / "benchmark.ini").open("w") as stream:
                        config.write(stream, space_around_delimiters=False)
                    environment = os.environ.copy()
                    environment.update(
                        BENCHMARK_CONFIG=str(directory / "benchmark.ini"),
                        BENCHMARK_OUTPUT=str(directory / "output.json"),
                        OMP_NUM_THREADS="4")
                    with (directory / "run.log").open("w") as log:
                        result = subprocess.run(
                            [str(binary), "--run_test=SPFreshTest/BenchmarkFromConfig"],
                            cwd=directory, env=environment, stdout=log,
                            stderr=subprocess.STDOUT, timeout=180)
                    text = (directory / "run.log").read_text()
                    self.assertEqual(result.returncode, 0, text[-4000:])
                    self.assertNotIn("Using random generation", text)
                    results = json.loads((directory / "output.json").read_text())["results"]
                    self.assertIn("benchmark0b_query_before_insert_round2", results)
                    if inserted:
                        self.assertEqual(results["benchmark1_insert"]["batch_1"]["inserted"], inserted)
                    else:
                        self.assertNotIn("benchmark1_insert", results)


if __name__ == "__main__":
    unittest.main()
