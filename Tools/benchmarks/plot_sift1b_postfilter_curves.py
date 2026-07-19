#!/usr/bin/env python3
"""Plot SIFT1B recall/QPS curves from the distance-first post-filter sweep."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LEVELS = ["unfilter", "org", "dept", "team", "project"]
COLORS = {
    "unfilter": "#9467bd",
    "org": "#2ca02c",
    "dept": "#ff7f0e",
    "team": "#d62728",
    "project": "#8c564b",
}


def load_results(path: Path) -> dict[str, list[dict]]:
    series = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.startswith("RESULT "):
                continue
            row = json.loads(line[7:])
            series[row["level"]].append(row)
    for level in series:
        series[level].sort(key=lambda row: row["nprobe"])
    missing = set(LEVELS) - set(series)
    if missing:
        raise RuntimeError(f"Missing result series: {sorted(missing)}")
    return series


def plot_series(level: str, rows: list[dict], out_path: Path) -> None:
    recalls = [100.0 * row["recall"] for row in rows]
    qps = [row["qps"] for row in rows]
    plt.figure(figsize=(6.6, 4.5))
    plt.plot(recalls, qps, "o-", color=COLORS[level], label="SPANN post-filter")
    for recall, throughput, row in zip(recalls, qps, rows):
        plt.annotate(f'{row["nprobe"]}', (recall, throughput), xytext=(4, 4),
                     textcoords="offset points", fontsize=8)
    plt.xlabel("Recall@100 (%)")
    plt.ylabel("QPS (1 thread)")
    plt.title(f"SIFT1B filter={level}, PipePQ32, rerankL=500")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_all(series: dict[str, list[dict]], out_path: Path) -> None:
    plt.figure(figsize=(8.2, 5.8))
    for level in LEVELS:
        rows = series[level]
        plt.plot(
            [100.0 * row["recall"] for row in rows],
            [row["qps"] for row in rows],
            "o-",
            color=COLORS[level],
            label=level,
        )
    plt.xlabel("Recall@100 (%)")
    plt.ylabel("QPS (1 thread)")
    plt.title("SIFT1B distance-first posting post-filter, PipePQ32, rerankL=500")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    series = load_results(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for level in LEVELS:
        plot_series(level, series[level], args.out_dir / f"recall_qps_{level}.png")
    plot_all(series, args.out_dir / "recall_qps_all.png")

    for level in LEVELS:
        print(level, [(row["nprobe"], row["recall"], row["qps"]) for row in series[level]])


if __name__ == "__main__":
    main()
