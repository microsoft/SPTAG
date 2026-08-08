#!/usr/bin/env python3
"""Render SIFT1B raw-STATIC STM1 Recall@10/QPS curves."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LEVELS = ("unfilter", "org", "dept", "team", "project")
COLORS = {
    "unfilter": "#1f77b4",
    "org": "#2ca02c",
    "dept": "#ff7f0e",
    "team": "#d62728",
    "project": "#9467bd",
}


def load_series(path: Path) -> dict[str, list[dict]]:
    series: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("RESULT "):
            row = json.loads(line[7:])
            series[row["level"]].append(row)

    missing = set(LEVELS) - set(series)
    if missing:
        raise RuntimeError(f"Missing result series: {sorted(missing)}")
    for level in LEVELS:
        series[level].sort(key=lambda row: row["internal_result_num"])
    return series


def plot_level(axis: plt.Axes, level: str, rows: list[dict]) -> None:
    recalls = [100.0 * row["recall"] for row in rows]
    qps = [row["qps"] for row in rows]
    axis.plot(recalls, qps, "o-", color=COLORS[level], linewidth=1.8)
    for recall, throughput, row in zip(recalls, qps, rows):
        axis.annotate(
            str(row["internal_result_num"]),
            (recall, throughput),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_title(level)
    axis.set_xlabel("Recall@10 (%)")
    axis.set_ylabel("QPS (1 thread)")
    axis.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    series = load_series(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    flat_axes = axes.flat
    for axis, level in zip(flat_axes, LEVELS):
        plot_level(axis, level, series[level])
    flat_axes[-1].axis("off")
    figure.suptitle("SIFT1B raw STATIC STM1 with in-posting ACL mask")
    figure.tight_layout()
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
