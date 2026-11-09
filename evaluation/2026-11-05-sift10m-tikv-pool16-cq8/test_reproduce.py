import configparser
import copy
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
import presplit
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
            args = SimpleNamespace(prefix="pin-check", run_root=root, source_root=ROOT.parent.parent,
                                   presplit_regions=0, storage="TIKVIO")
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

    def two_layer_fixture(self):
        data = copy.deepcopy(self.data)
        data["config"]["layers"] = 2
        results = data["results"]
        queries = [results["benchmark0_query_before_insert"],
                   results["benchmark0b_query_before_insert_round2"]]
        for batch in results["benchmark1_insert"].values():
            queries.extend([batch["search"], batch["search_round2"]])
        for query in queries:
            upper = copy.deepcopy(query["latencyBreakdown"]["layers"]["0"])
            upper["versionCheckCountMean"] = 0
            upper["versionMapMs"]["mean"] = 0
            query["latencyBreakdown"]["layers"]["1"] = upper
        self.saved["BuildSSDIndex"]["Layers"] = "2"
        log = "\n".join(line for line in self.log.splitlines()
                        if not ("Checkpoint layer stats:" in line and " layer=1 " in line))
        log += "\nUsing distributed TiKV VersionMap (layer=1)\n"
        log += "".join(f"Checkpoint layer stats: batch {phase}/10 layer=1 samples=10000000 "
                       "deleted=9900000 headMapping=100000\n" for phase in range(1, 11))
        return data, log

    def test_two_layers_retain_version_checks_only_on_bottom(self):
        data, log = self.two_layer_fixture()
        rows, batches = reproduce.validate_measurements(
            data, log, self.saved, self.prefix, layers=2)
        self.assertEqual(len(rows), 22)
        self.assertEqual(len(batches), 10)

    def test_two_layers_reject_upper_search_version_checks(self):
        data, log = self.two_layer_fixture()
        data["results"]["benchmark0_query_before_insert"]["latencyBreakdown"]["layers"]["1"][
            "versionCheckCountMean"] = 64
        with self.assertRaisesRegex(RuntimeError, "Upper-layer search"):
            reproduce.validate_measurements(data, log, self.saved, self.prefix, layers=2)

    def test_two_layers_reject_missing_upper_checkpoint(self):
        data, log = self.two_layer_fixture()
        log = log.replace("Checkpoint layer stats: batch 10/10 layer=1", "Missing layer 1")
        with self.assertRaisesRegex(RuntimeError, "upper-layer checkpoint"):
            reproduce.validate_measurements(data, log, self.saved, self.prefix, layers=2)

    def fileio_fixture(self):
        data, log = self.two_layer_fixture()
        self.saved["BuildSSDIndex"]["Storage"] = "FILEIO"
        self.saved["BuildSSDIndex"]["DistributedVersionMap"] = "false"
        log = log.replace("SPANNIndex:UseTiKV", "SPANNIndex:UseFileIO")
        log = log.replace("Using distributed TiKV VersionMap", "Using local in-memory VersionMap")
        results = data["results"]
        queries = [results["benchmark0_query_before_insert"],
                   results["benchmark0b_query_before_insert_round2"]]
        for batch in results["benchmark1_insert"].values():
            queries.extend([batch["search"], batch["search_round2"]])
        for query in queries:
            for layer in query["latencyBreakdown"]["layers"].values():
                layer["versionCheckCountMean"] = 0
                layer["versionMapMs"]["mean"] = 0
        return data, log

    def test_fileio_two_layers_keep_local_version_checks(self):
        data, log = self.fileio_fixture()
        rows, batches = reproduce.validate_measurements(
            data, log, self.saved, self.prefix, layers=2, storage="FILEIO")
        self.assertEqual(len(rows), 22)
        self.assertEqual(len(batches), 10)

    def test_fileio_one_layer_uses_local_scan_telemetry(self):
        data, log = self.fileio_fixture()
        data["config"]["layers"] = 1
        self.saved["BuildSSDIndex"]["Layers"] = "1"
        results = data["results"]
        queries = [results["benchmark0_query_before_insert"],
                   results["benchmark0b_query_before_insert_round2"]]
        for batch in results["benchmark1_insert"].values():
            queries.extend([batch["search"], batch["search_round2"]])
        for query in queries:
            del query["latencyBreakdown"]["layers"]["1"]
        rows, batches = reproduce.validate_measurements(
            data, log, self.saved, self.prefix, storage="FILEIO")
        self.assertEqual(len(rows), 22)
        self.assertEqual(len(batches), 10)

    def test_fileio_rejects_distributed_version_check_telemetry(self):
        data, log = self.fileio_fixture()
        data["results"]["benchmark0_query_before_insert"]["latencyBreakdown"]["layers"]["1"][
            "versionCheckCountMean"] = 64
        with self.assertRaisesRegex(RuntimeError, "local FILEIO layer telemetry"):
            reproduce.validate_measurements(data, log, self.saved, self.prefix, layers=2, storage="FILEIO")

    def test_fileio_rejects_missing_upper_local_versionmap(self):
        data, log = self.fileio_fixture()
        log = log.replace("Using local in-memory VersionMap (layer=1)", "Missing upper local map")
        with self.assertRaisesRegex(RuntimeError, "Missing upper-layer VersionMap"):
            reproduce.validate_measurements(data, log, self.saved, self.prefix, layers=2, storage="FILEIO")

    def test_fileio_rejects_unsearched_upper_layer(self):
        data, log = self.fileio_fixture()
        data["results"]["benchmark0_query_before_insert"]["latencyBreakdown"]["layers"]["1"][
            "postingCountMean"] = 0
        with self.assertRaisesRegex(RuntimeError, "local FILEIO layer telemetry"):
            reproduce.validate_measurements(data, log, self.saved, self.prefix, layers=2, storage="FILEIO")

    def test_fileio_rejects_distributed_runtime(self):
        data, log = self.fileio_fixture()
        with self.assertRaisesRegex(RuntimeError, "failure marker"):
            reproduce.validate_measurements(
                data, log + "\nUsing distributed TiKV VersionMap", self.saved,
                self.prefix, layers=2, storage="FILEIO")

    def test_fileio_rejects_presplit_before_creating_files(self):
        args = SimpleNamespace(storage="FILEIO", presplit_regions=128)
        with self.assertRaisesRegex(RuntimeError, "TiKV-only"):
            reproduce.prepare(args)

    def test_fileio_runner_never_starts_a_tikv_cluster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fileio").mkdir()
            (root / "checksums.json").write_text("{}")
            (root / "run-manifest.json").write_text(json.dumps({
                "storage": "FILEIO", "prefix": "fio-run", "recipe_sha256": {},
                "presplit_regions": 0}))
            process = SimpleNamespace(returncode=0, poll=lambda: 0)
            with patch.object(reproduce, "Cluster", side_effect=AssertionError("TiKV must not start")), \
                    patch.object(reproduce.shutil, "disk_usage", return_value=SimpleNamespace(free=600 * reproduce.GIB)), \
                    patch.object(reproduce.subprocess, "Popen", return_value=process), \
                    patch.object(reproduce.signal, "signal"), \
                    patch.object(reproduce, "validate") as validate:
                reproduce.run(root)
            validate.assert_called_once_with(root, live=True)
            self.assertEqual((root / "fileio/exit-code.txt").read_text(), "0\n")
            self.assertEqual(json.loads((root / "status.json").read_text())["phase"], "complete")

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


class PresplitTests(unittest.TestCase):
    def setUp(self):
        self.lower, self.upper, self.keys, self.samples = presplit.plan(
            "test-presplit", 2, 10000000, 128)
        starts = [b"", *self.keys]
        self.regions = [{
            "id": i + 1, "start_key": start.hex(),
            "end_key": starts[i + 1].hex() if i + 1 < len(starts) else "",
            "peers": [{"store_id": store, "role_name": "Voter"} for store in (1, 2, 3)],
            "leader": {"store_id": i % 3 + 1},
        } for i, start in enumerate(starts)]

    def inspect(self):
        return presplit.inspect_plan(self.regions, {1, 2, 3}, self.lower, self.upper,
                                     self.keys, self.samples)

    def test_exact_posting_and_both_version_key_layouts(self):
        self.assertEqual(len(self.keys), 391)
        self.assertEqual(self.keys, sorted(set(self.keys)))
        self.assertIn(self.lower + bytes([2, 0, 0, 0]), self.keys)
        for layer in (0, 1):
            self.assertIn(self.lower + f"v:{layer}:0000000000".encode(), self.keys)
            self.assertIn(self.lower + f"v:{layer}:0010000000".encode(), self.keys)
            self.assertEqual(len(self.samples[f"version{layer}"]), 128)

    def test_one_layer_preserves_posting_and_version0_layout(self):
        lower, upper, keys, samples = presplit.plan("test-presplit", 1, 10000000, 128)
        self.assertEqual((lower, upper), (self.lower, self.upper))
        self.assertEqual(len(keys), 260)
        self.assertTrue(set(keys).issubset(self.keys))
        self.assertEqual(samples, {key: self.samples[key] for key in ("posting", "version0")})

    def test_complete_replicated_balanced_plan(self):
        state = self.inspect()
        self.assertEqual(state["missing_boundaries"], [])
        self.assertEqual(state["unready_region_ids"], [])
        self.assertTrue(state["balanced"])
        self.assertTrue(all(len(ids) == 128 for ids in state["groups"].values()))

    def test_rejects_leader_concentration(self):
        for region in self.regions:
            region["leader"] = {"store_id": 1}
        self.assertFalse(self.inspect()["balanced"])

    def test_pending_down_and_learner_peers_are_not_ready(self):
        for change in ({"pending_peers": [{"store_id": 3}]},
                       {"down_peers": [{"peer": {"store_id": 3}}]},
                       {"leader": None},
                       {"peers": [{"store_id": store, "role_name": "Learner"}
                                  for store in (1, 2, 3)]}):
            with self.subTest(change=change):
                region = {**self.regions[1], **change}
                self.assertEqual(presplit.replication_issues([region], {1, 2, 3}), [region["id"]])

    def test_invalid_key_plans_fail_explicitly(self):
        for prefix, layers, vectors, buckets in (("../other", 2, 10000000, 128),
                                                 ("test", 2, 10000000, 0),
                                                 ("test", 2, 100, 128)):
            with self.subTest(prefix=prefix, buckets=buckets):
                with self.assertRaises(ValueError):
                    presplit.plan(prefix, layers, vectors, buckets)


if __name__ == "__main__":
    unittest.main()
