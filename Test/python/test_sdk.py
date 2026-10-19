import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import SPTAG


class SDKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vectors = np.arange(128, dtype=np.float32)[:, None] + np.arange(8, dtype=np.float32)[None, :] / 8
        cls.indexes = {}
        for algorithm in ("BKT", "KDT"):
            index = SPTAG.AnnIndex(algorithm, "Float", 8)
            index.SetBuildParam("DistCalcMethod", "L2", "Index")
            index.SetBuildParam("NumberOfThreads", "1", "Index")
            if not index.Build(cls.vectors, len(cls.vectors), False):
                raise RuntimeError("Building {} failed".format(algorithm))
            cls.indexes[algorithm] = index

    def check_search(self, index):
        for row in (0, 31, 63):
            ids, distances, metadata, relaxed = index.Search(self.vectors[row], 3)
            self.assertEqual(len(ids), 3)
            self.assertEqual(len(distances), 3)
            self.assertEqual(ids[0], row)
            self.assertEqual(distances[0], 0)
            self.assertEqual(metadata, [None] * 3)
            self.assertEqual(relaxed, [False] * 3)

    def test_bkt_and_kdt_search_result_contract(self):
        for algorithm, index in self.indexes.items():
            with self.subTest(algorithm=algorithm):
                self.check_search(index)

    def test_save_load_and_repeat_search(self):
        for algorithm, index in self.indexes.items():
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory(prefix="sptag-sdk-test-") as directory:
                path = str(Path(directory) / "index")
                self.assertTrue(index.Save(path))
                loaded = SPTAG.AnnIndex.Load(path)
                self.assertTrue(loaded.ReadyToServe())
                self.check_search(loaded)
                self.check_search(loaded)

    def test_metadata_is_returned_as_bytes(self):
        index = SPTAG.AnnIndex("BKT", "Float", 8)
        index.SetBuildParam("DistCalcMethod", "L2", "Index")
        index.SetBuildParam("NumberOfThreads", "1", "Index")
        metadata = "".join("document-{}\n".format(row) for row in range(len(self.vectors))).encode("utf-8")
        self.assertTrue(index.BuildWithMetaData(self.vectors, metadata, len(self.vectors), False, False))
        ids, distances, values, relaxed = index.SearchWithMetaData(self.vectors[31], 3)
        self.assertEqual(ids[0], 31)
        self.assertEqual(distances[0], 0)
        self.assertEqual(values[0].rstrip(b"\n"), b"document-31")
        self.assertTrue(all(isinstance(value, bytes) for value in values))
        self.assertEqual(relaxed, [False] * 3)

    def test_repeated_result_lifetimes(self):
        for _ in range(500):
            ids, distances, metadata, relaxed = self.indexes["BKT"].Search(self.vectors[0], 3)
            self.assertEqual(metadata, [None] * 3)
            self.assertTrue(all(isinstance(value, bool) for value in relaxed))
        del ids, distances, metadata, relaxed
        gc.collect()

    def test_wrong_buffer_size_is_rejected(self):
        index = SPTAG.AnnIndex("BKT", "Float", 8)
        self.assertFalse(index.Build(self.vectors.astype(np.float64), len(self.vectors), False))


class ExampleTest(unittest.TestCase):
    script = Path(__file__).resolve().parents[2] / "Examples" / "python" / "quickstart.py"

    def run_example(self, *args):
        return subprocess.run(
            [sys.executable, str(self.script), *map(str, args)],
            capture_output=True, text=True, timeout=60, check=False,
        )

    def test_own_data_and_independent_process_reload(self):
        with tempfile.TemporaryDirectory(prefix="sptag-own-data-") as directory:
            root = Path(directory)
            vectors = np.array([[0, 0], [1, 0], [0, 2], [3, 3]], dtype=np.float32)
            np.save(root / "vectors.npy", vectors)
            np.save(root / "queries.npy", vectors[:1])
            ids = ["doc-A", "doc-B", "doc-C", "doc-D"]
            (root / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
            index = root / "output"
            build = self.run_example(
                "--index", index, "--vectors", root / "vectors.npy",
                "--ids", root / "ids.json", "--queries", root / "queries.npy", "--k", 3,
            )
            load = self.run_example("--index", index, "--load", "--queries", root / "queries.npy", "--k", 3)
            for operation, result in (("build", build), ("load", load)):
                with self.subTest(operation=operation):
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]["neighbors"][0], {"id": "doc-A", "row_id": 0, "distance": 0.0})
                    self.assertEqual(len(rows[0]["neighbors"]), 3)
            self.assertTrue((index / "index" / "indexloader.ini").is_file())
            self.assertEqual(json.loads((index / "ids.json").read_text(encoding="utf-8")), ids)
            repeated = self.run_example("--index", index)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("Output exists", repeated.stderr)

    def test_invalid_inputs_fail_before_creating_output(self):
        with tempfile.TemporaryDirectory(prefix="sptag-invalid-data-") as directory:
            root = Path(directory)
            valid = np.zeros((4, 2), dtype=np.float32)
            np.save(root / "valid.npy", valid)
            np.save(root / "double.npy", valid.astype(np.float64))
            np.save(root / "flat.npy", valid.reshape(-1))
            np.save(root / "empty.npy", np.zeros((0, 2), dtype=np.float32))
            np.save(root / "nonfinite.npy", np.full((4, 2), np.nan, dtype=np.float32))
            np.save(root / "wrong-dimension.npy", np.zeros((1, 3), dtype=np.float32))
            (root / "duplicate-ids.json").write_text(json.dumps(["same"] * 4), encoding="utf-8")
            (root / "short-ids.json").write_text(json.dumps(["one"]), encoding="utf-8")
            cases = [
                (["--vectors", root / "double.npy"], "finite float32"),
                (["--vectors", root / "flat.npy"], "two-dimensional"),
                (["--vectors", root / "empty.npy"], "two-dimensional"),
                (["--vectors", root / "nonfinite.npy"], "finite float32"),
                (["--vectors", root / "missing.npy"], "No such file"),
                (["--vectors", root / "valid.npy", "--queries", root / "wrong-dimension.npy"], "dimensions must match"),
                (["--vectors", root / "valid.npy", "--ids", root / "duplicate-ids.json"], "unique"),
                (["--vectors", root / "valid.npy", "--ids", root / "short-ids.json"], "one entry per vector"),
                (["--k", 0], "must be positive"),
                (["--vectors", root / "valid.npy", "--k", 5], "must not exceed"),
                (["--load"], "requires --queries"),
                (["--load", "--vectors", root / "valid.npy"], "cannot be combined"),
                (["--ids", root / "short-ids.json"], "requires --vectors"),
            ]
            for number, (args, message) in enumerate(cases):
                with self.subTest(args=args):
                    output = root / "output-{}".format(number)
                    result = self.run_example("--index", output, *args)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
