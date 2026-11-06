import configparser
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import deployment
import reproduce


ROOT = Path(__file__).resolve().parent


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "tikv-output.json").read_text())
        self.log = (ROOT / "tikv-run.log").read_text()
        self.prefix = "qianxi_sift10m_tikv_layer1_build80_pool16_cq8_20261105"
        self.saved = configparser.ConfigParser()
        self.saved["Base"] = {"VectorSize": "10000000"}
        self.saved.read(ROOT / "tikv-recorded.ini")
        self.saved["BuildSSDIndex"]["Layers"] = "1"

    def test_recorded_results_match_all_published_rounds(self):
        rows, batches = reproduce.validate_measurements(self.data, self.log, self.saved, self.prefix)
        expected = json.loads((ROOT / "results.json").read_text())
        self.assertEqual(len(batches), 10)
        self.assertEqual(len(rows), 22)
        self.assertEqual(rows, expected["query_rounds"])
        self.assertEqual(batches, expected["insertions"])

    def test_prepare_rejects_unmeasured_source_before_creating_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "new-run"
            args = SimpleNamespace(prefix="pin-check", run_root=root, source_root=ROOT.parent.parent)
            with patch.object(reproduce.subprocess, "check_output", return_value="unmeasured\n"):
                with self.assertRaisesRegex(RuntimeError, "fixed measured source commit"):
                    reproduce.prepare(args)
            self.assertFalse(root.exists())

    def test_rejects_missing_round(self):
        del self.data["results"]["benchmark1_insert"]["batch_10"]["search_round2"]
        with self.assertRaises(KeyError):
            reproduce.validate_measurements(self.data, self.log, self.saved, self.prefix)

    def test_rejects_nonpositive_insertion_time(self):
        for value in (0, -1):
            with self.subTest(value=value):
                self.data["results"]["benchmark1_insert"]["batch_1"]["insert timeSeconds"] = value
                with self.assertRaisesRegex(RuntimeError, "Insertion time"):
                    reproduce.validate_measurements(self.data, self.log, self.saved, self.prefix)

    def test_rejects_wrong_count(self):
        log = self.log.replace("batch 10/10 layer=0 samples=10000000", "batch 10/10 layer=0 samples=9900000")
        with self.assertRaisesRegex(RuntimeError, "membership mismatch"):
            reproduce.validate_measurements(self.data, log, self.saved, self.prefix)

    def test_rejects_disabled_version_checks(self):
        self.saved["BuildSSDIndex"]["DistributedVersionMap"] = "false"
        with self.assertRaisesRegex(RuntimeError, "Version checking"):
            reproduce.validate_measurements(self.data, self.log, self.saved, self.prefix)

    def test_rejects_read_failure(self):
        with self.assertRaisesRegex(RuntimeError, "failure marker"):
            reproduce.validate_measurements(
                self.data, self.log + "\nBatchGetVersions: MultiGet failed", self.saved, self.prefix)

    def test_extracts_exact_prefix_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(struct.pack("<II", 4, 2) + bytes(range(8)))
            reproduce.extract_vectors(source, root / "first", count=3, dimension=2)
            self.assertEqual((root / "first").read_bytes(), struct.pack("<II", 3, 2) + bytes(range(6)))
            source.write_bytes(source.read_bytes()[:-1])
            with self.assertRaisesRegex(RuntimeError, "shape/length"):
                reproduce.extract_vectors(source, root / "bad", count=3, dimension=2)

    def test_deployment_matches_recorded_container_commands(self):
        recorded = json.loads((ROOT / "cluster-shutdown.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tikv.toml").write_text((ROOT / "tikv.toml").read_text())
            cluster = deployment.Cluster(root, "test-repro")
            calls = []
            with patch.object(cluster, "docker"), patch.object(cluster, "wait"), \
                    patch.object(cluster, "start_container", side_effect=lambda *args: calls.append(args)), \
                    patch.object(deployment.socket, "socket"):
                cluster.start()
            self.assertEqual(len(calls), 6)
            for name, image, args, mounts in calls:
                role = name.rsplit("-", 1)[-1]
                original = next(row for row in recorded if row["Name"].endswith("-" + role))
                self.assertEqual(image, original["Config"]["Image"])
                self.assertEqual(args, original["Config"]["Cmd"])
                self.assertEqual(len(mounts), 2 if role.startswith("tikv") else 1)

    def test_shutdown_refuses_foreign_container_labels(self):
        cluster = deployment.Cluster(ROOT, "our-run")
        cluster.owned = ["our-run-tikv1"]
        result = type("Result", (), {"stdout": '[{"Config":{"Labels":{"sptag.run":"other-run"}}}]'})()
        with patch.object(cluster, "docker", return_value=result) as docker:
            with self.assertRaisesRegex(RuntimeError, "different ownership"):
                cluster.close()
            self.assertEqual(docker.call_count, 1)

    @unittest.skipUnless(os.environ.get("EXACT_TRUTH_BIN"), "Set EXACT_TRUTH_BIN after building helpers")
    def test_exact_truth_matches_integer_bruteforce_including_ties(self):
        rng = random.Random(20261102)
        vectors = [bytes(rng.randrange(256) for _ in range(128)) for _ in range(100)]
        vectors[91] = vectors[90]
        queries = [vectors[90], vectors[0]] + [
            bytes(rng.randrange(256) for _ in range(128)) for _ in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, rows in (("base", vectors), ("query", queries)):
                (root / name).write_bytes(struct.pack("<II", len(rows), 128) + b"".join(rows))
            subprocess.run([os.environ["EXACT_TRUTH_BIN"], str(root / "base"), str(root / "query"),
                            "90", "1", "10", "2", str(root / "truth")],
                           check=True, stdout=subprocess.DEVNULL)
            data = (root / "truth").read_bytes()
            size = 11 * len(queries) * 5
            self.assertEqual(struct.unpack_from("<II", data), (22 * len(queries), 5))
            ids = struct.unpack_from("<" + str(size) + "i", data, 8)
            distances = struct.unpack_from("<" + str(size) + "f", data, 8 + size * 4)
            self.assertEqual(len(data), 8 + size * 8)
            for phase in range(11):
                for q, query in enumerate(queries):
                    expected = sorted((sum((a - b) ** 2 for a, b in zip(vector, query)), vid)
                                      for vid, vector in enumerate(vectors[:90 + phase]))[:5]
                    offset = (phase * len(queries) + q) * 5
                    self.assertEqual(ids[offset:offset + 5], tuple(row[1] for row in expected))
                    self.assertEqual(distances[offset:offset + 5], tuple(row[0] for row in expected))


if __name__ == "__main__":
    unittest.main()
