import json
from pathlib import Path
import tempfile
import unittest

import deployment
import presplit
import publish_report
import reproduce
import test_reproduce


class PublicationTests(unittest.TestCase):
    def setUp(self):
        fixture = test_reproduce.ReproductionTests()
        fixture.setUp()
        self.data = fixture.data
        self.rows, self.updates = reproduce.validate_measurements(
            fixture.data, fixture.log, fixture.saved, fixture.prefix)

    def test_all_metrics_and_throughput_denominator_match_raw_records(self):
        metrics = publish_report.measurements(self.data, self.rows, self.updates)
        self.assertEqual(len(metrics["query_rounds"]), 22)
        self.assertEqual(len(metrics["insertions"]), 10)
        total = sum(row["insert_seconds"] for row in self.updates)
        self.assertEqual(metrics["aggregate_insert_vectors_per_second"], 100000 / total)
        mean_rates = sum(row["insert_vectors_per_second"] for row in self.updates) / 10
        self.assertNotAlmostEqual(metrics["aggregate_insert_vectors_per_second"], mean_rates)
        query = self.data["results"]["benchmark1_insert"]["batch_10"]["search_round2"]
        final = metrics["query_rounds"][-1]
        self.assertEqual((final["phase"], final["round"], final["active_vectors"]), (10, 2, 10000000))
        self.assertEqual(final["p50_ms"], query["p50"])
        self.assertEqual(final["latency_breakdown"], query["latencyBreakdown"])
        self.assertEqual(metrics["barrier_wait_seconds"], sum(
            batch["batch barrier waitSeconds"]
            for batch in self.data["results"]["benchmark1_insert"].values()))

    def test_rejects_inconsistent_insertion_rate(self):
        self.updates[0]["insert_vectors_per_second"] *= 2
        with self.assertRaisesRegex(RuntimeError, "count / duration"):
            publish_report.measurements(self.data, self.rows, self.updates)

    def test_rejects_barrier_outside_insertion(self):
        self.data["results"]["benchmark1_insert"]["batch_1"]["batch barrier waitSeconds"] = (
            self.updates[0]["insert_seconds"] + 1)
        with self.assertRaisesRegex(RuntimeError, "outside insertion"):
            publish_report.measurements(self.data, self.rows, self.updates)

    def test_rejects_nonmonotonic_latency_percentiles(self):
        self.data["results"]["benchmark0_query_before_insert"]["p50"] = 1000000
        with self.assertRaisesRegex(RuntimeError, "latency percentiles"):
            publish_report.measurements(self.data, self.rows, self.updates)

    def test_rejects_nonfinite_qps(self):
        self.rows[0]["qps"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "query performance"):
            publish_report.measurements(self.data, self.rows, self.updates)

    def test_rejects_nonfinite_checkpoint_duration(self):
        self.updates[0]["clone_seconds"] = float("inf")
        with self.assertRaisesRegex(RuntimeError, "checkpoint performance"):
            publish_report.measurements(self.data, self.rows, self.updates)

    def test_publish_never_overwrites_previous_report(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                publish_report.publish(Path("/unused"), Path(directory))

    def presplit_fixture(self, root):
        prefix = "publication-test"
        lower, upper, keys, samples = presplit.plan(prefix, 2, 10000000, 128)
        starts = [b"", *keys]
        regions = [{
            "id": i + 1, "start_key": start.hex(),
            "end_key": starts[i + 1].hex() if i + 1 < len(starts) else "",
            "peers": [{"store_id": store, "role_name": "Voter"} for store in (1, 2, 3)],
            "leader": {"store_id": i % 3 + 1},
        } for i, start in enumerate(starts)]
        evidence = presplit.inspect_plan(regions, {1, 2, 3}, lower, upper, keys, samples)
        evidence.update(status="verified", split_keys=[key.hex() for key in keys],
                        stores_after={"stores": [{"store": {"id": i}} for i in (1, 2, 3)]},
                        split_responses=[{"processed-percentage": 100}],
                        scatter_responses={"posting": {"processed-percentage": 100}})
        (root / "presplit-validation.json").write_text(json.dumps(evidence))
        (root / "regions-before-workload.json").write_text(json.dumps({
            "count": len(regions), "regions": regions}))
        stopped = [{"Name": f"/{prefix}-{role}{i}",
                    "State": {"Running": False, "OOMKilled": False, "ExitCode": 0},
                    "Config": {"Image": deployment.PD_IMAGE if role == "pd" else deployment.TIKV_IMAGE}}
                   for role in ("pd", "tikv") for i in (1, 2, 3)]
        (root / "cluster-stopped.json").write_text(json.dumps(stopped))
        return {"prefix": prefix, "layers": 2}

    def test_recomputes_presplit_replicas_boundaries_and_leaders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.presplit_fixture(root)
            publish_report.verify_presplit(root, manifest)
            snapshot = json.loads((root / "regions-before-workload.json").read_text())
            snapshot["regions"][1]["pending_peers"] = [{"store_id": 3}]
            (root / "regions-before-workload.json").write_text(json.dumps(snapshot))
            with self.assertRaisesRegex(RuntimeError, "acceptance"):
                publish_report.verify_presplit(root, manifest)

    def test_rejects_wrong_server_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.presplit_fixture(root)
            stopped = json.loads((root / "cluster-stopped.json").read_text())
            stopped[-1]["Config"]["Image"] = "unrecorded-image"
            (root / "cluster-stopped.json").write_text(json.dumps(stopped))
            with self.assertRaisesRegex(RuntimeError, "wrong-image"):
                publish_report.verify_presplit(root, manifest)


if __name__ == "__main__":
    unittest.main()
