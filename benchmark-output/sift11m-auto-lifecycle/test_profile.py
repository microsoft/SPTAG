import configparser
from decimal import Decimal, InvalidOperation
import os
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from calibrate_fullscale import thresholds_from_trajectories
from run_experiment import GIB, run_phase, thresholds_from_growth


ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent.parent
BASELINE = REPO / "evaluation/2026-11-09-sift10m-fileio-layer1-matched"


def canonical(value):
    value = value.strip().lower()
    if value in ("true", "false"):
        return value == "true"
    try:
        return Decimal(value)
    except InvalidOperation:
        return value


class MatchedProfileTest(unittest.TestCase):
    def test_applied_effective_parameters_match_historical_profile(self):
        self.check_profile(10_000_000, 10)

    def test_ordinary_merge_candidates_change_only_the_declared_threshold(self):
        for threshold in (24, 32, 48):
            with self.subTest(threshold=threshold):
                self.check_profile(5_000_000, threshold)

    def check_profile(self, pivot, merge_threshold):
        profile = configparser.ConfigParser(interpolation=None)
        profile.read(BASELINE / "saved-indexloader.ini")
        benchmark = configparser.ConfigParser(interpolation=None)
        benchmark.read(BASELINE / "benchmark.ini")
        with tempfile.TemporaryDirectory(prefix="sptag-matched-profile-") as directory:
            environment = os.environ.copy()
            environment.update({
                "SPTAG_SIFT_MODE": "config",
                "SPTAG_SIFT_OUTPUT": str(pathlib.Path(directory) / "index"),
                "SPTAG_SIFT_PROFILE": str(BASELINE / "saved-indexloader.ini"),
                "SPTAG_SIFT_HIGH": "1000000000000",
                "SPTAG_SIFT_TARGET": "900000000000",
                "SPTAG_SIFT_LOW": "800000000000",
                "SPTAG_SIFT_PIVOT": str(pivot),
                "SPTAG_SIFT_MERGE_THRESHOLD": str(merge_threshold),
            })
            result = subprocess.run(
                [os.environ.get("SPTAG_TEST_BINARY", str(REPO / "Release/SPTAGTest")),
                 "--run_test=SiftLifecycleTest/AutomaticLifecycle",
                 "--log_level=error", "--color_output=no"],
                env=environment, cwd=REPO, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            applied = configparser.ConfigParser(interpolation=None)
            applied.read(pathlib.Path(directory) / "configured.ini")
            for section in ("Base", "SelectHead", "BuildHead", "BuildSSDIndex"):
                for key, value in profile[section].items():
                    if section == "Base" and key in ("vectorpath", "vectorsize", "indexdirectory"):
                        continue
                    with self.subTest(section=section, key=key):
                        expected = "true" if section == "BuildSSDIndex" and key == "asyncmergeinsearch" else value
                        if section == "BuildSSDIndex" and key == "mergethreshold":
                            expected = str(merge_threshold)
                        self.assertEqual(canonical(applied[section][key]), canonical(expected))
            for key in ("NumSearchThreads", "NumInsertThreads", "NumSearchDuringInsertThreads", "NumQueries", "TopK"):
                with self.subTest(benchmark=key):
                    self.assertEqual(applied["Benchmark"][key], benchmark["Benchmark"][key])
            self.assertEqual(dict(applied["Lifecycle"]),
                             {"pivot": str(pivot), "initial": str(pivot * 9 // 10),
                              "total": str(pivot * 11 // 10), "batch": str(pivot // 100)})


class GrowthThresholdTest(unittest.TestCase):
    def snapshots(self):
        return [
            {"phase": "initial", "live": 4_500_000, "layers": 1, "head_bytes": 900},
            {"phase": "insert-10", "live": 5_000_000, "layers": 1, "head_bytes": 1000},
            {"phase": "complete", "live": 5_500_000, "layers": 1, "head_bytes": 1100},
        ]

    def test_freezes_actual_scale_growth_without_multiplication(self):
        self.assertEqual(thresholds_from_growth(self.snapshots(), 5_000_000),
                         {"high": 1000, "target": 980, "low": 940})

    def test_rejects_wrong_scale_incomplete_or_non_increasing_calibration(self):
        for key, value in (("live", 10_000_000), ("layers", 2), ("head_bytes", 900)):
            rows = self.snapshots()
            rows[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                thresholds_from_growth(rows, 5_000_000)
        with self.assertRaises(ValueError):
            thresholds_from_growth(self.snapshots()[:-1], 5_000_000)


class DiskGuardTest(unittest.TestCase):
    def test_disk_guard_terminates_child_before_expansion_consumes_reserve(self):
        child = mock.Mock()
        child.pid = 1234
        child.poll.return_value = None
        with tempfile.TemporaryDirectory(prefix="sptag-disk-guard-") as directory:
            with mock.patch("run_experiment.subprocess.Popen", return_value=child), \
                    mock.patch("run_experiment.shutil.disk_usage", return_value=mock.Mock(free=29 * GIB)):
                with self.assertRaisesRegex(RuntimeError, "disk free 29.00 GiB below 30 GiB guard"):
                    run_phase(pathlib.Path(directory), "growth-calibration",
                              {"high": 1000, "target": 980, "low": 940},
                              pivot=5_000_000, merge_threshold=32)
        child.terminate.assert_called_once()
        child.wait.assert_called_once_with(timeout=30)


class FullscaleThresholdTest(unittest.TestCase):
    def trajectories(self, before, after, final):
        return (
            [{"phase": "insert-10", "head_bytes": 1000}],
            [{"phase": "evaluated-10", "compact_estimate_bytes": before},
             {"phase": "delete-11", "compact_estimate_bytes": after},
             {"phase": "complete", "compact_estimate_bytes": final}],
        )

    def test_freezes_separated_thresholds_from_real_growth_and_deletion(self):
        self.assertEqual(thresholds_from_trajectories(*self.trajectories(900, 800, 700)),
                         {"high": 1000, "target": 925, "low": 850})
        self.assertEqual(thresholds_from_trajectories(*self.trajectories(1050, 900, 850)),
                         {"high": 1000, "target": 987, "low": 975})

    def test_rejects_unreachable_or_non_decreasing_shrink_memory(self):
        for values in ((1200, 1150, 1100), (900, 900, 800), (900, 800, 850)):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "cannot place"):
                thresholds_from_trajectories(*self.trajectories(*values))


if __name__ == "__main__":
    unittest.main()
