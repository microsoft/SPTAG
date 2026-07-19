#!/usr/bin/env python3
"""Render the matched-recall SIFT1B unfilter concurrency comparison."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_result_lines(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.startswith("RESULT "):
                continue
            rows.append(json.loads(line[7:]))
    if not rows:
        raise RuntimeError(f"No RESULT rows found in {path}")
    return rows


def select_rows(
    rows: list[dict],
    engine: str,
    budget_key: str,
    budget: int,
) -> dict[int, dict]:
    selected = {}
    for row in rows:
        if row.get("engine") != engine or row.get(budget_key) != budget:
            continue
        threads = row.get("threads")
        if not isinstance(threads, int) or threads <= 0:
            raise RuntimeError(f"Invalid thread count in {row}")
        if threads in selected:
            raise RuntimeError(f"Duplicate {engine} result for {threads} threads")
        selected[threads] = row
    if not selected:
        raise RuntimeError(f"No {engine} rows found at {budget_key}={budget}")
    return selected


def require_shared_protocol(spann: dict[int, dict], pipeann: dict[int, dict]) -> list[int]:
    threads = sorted(spann)
    if threads != sorted(pipeann):
        raise RuntimeError(
            f"Thread grids differ: SPANN={threads}, PipeANN={sorted(pipeann)}"
        )

    for name in ("warmup_queries", "measured_queries", "topk"):
        values = {row[name] for row in spann.values()} | {row[name] for row in pipeann.values()}
        if len(values) != 1:
            raise RuntimeError(f"Protocol mismatch for {name}: {sorted(values)}")
    for row in pipeann.values():
        if row.get("path") != "pipe_search" or row.get("search_mode") != 2:
            raise RuntimeError("PipeANN comparison must use pipe_search (search_mode=2)")
    return threads


def build_rows(
    threads: list[int], spann: dict[int, dict], pipeann: dict[int, dict]
) -> list[dict]:
    rows = []
    for thread_count in threads:
        spann_row = spann[thread_count]
        pipeann_row = pipeann[thread_count]
        spann_qps = float(spann_row["qps"])
        pipeann_qps = float(pipeann_row["qps"])
        rows.append(
            {
                "threads": thread_count,
                "spann_nprobe": spann_row["nprobe"],
                "spann_recall_percent": float(spann_row["recall_percent"]),
                "spann_qps": spann_qps,
                "spann_p99_ms": float(spann_row["p99_latency_us"]) / 1000.0,
                "pipeann_L": pipeann_row["L"],
                "pipeann_recall_percent": float(pipeann_row["recall_percent"]),
                "pipeann_qps": pipeann_qps,
                "pipeann_p99_ms": float(pipeann_row["p99_latency_us"]) / 1000.0,
                "recall_gap_pipeann_minus_spann_pp": (
                    float(pipeann_row["recall_percent"])
                    - float(spann_row["recall_percent"])
                ),
                "pipeann_over_spann_qps": pipeann_qps / spann_qps,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(path: Path, rows: list[dict]) -> None:
    threads = [row["threads"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    figure.suptitle(
        "SIFT1B unfilter concurrency at matched Recall@100",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.02,
        "Shared workload: 4,200 queries (200 warmup, 4,000 measured), top-k=100. "
        "SPANN: nprobe=500, 96 AIO contexts x 512 events. "
        "PipeANN: pipe_search, L=350.",
        ha="center",
        fontsize=9,
    )

    series = (
        ("SPANN nprobe=500", "spann", "#d55e00"),
        ("PipeANN pipe_search L=350", "pipeann", "#0072b2"),
    )
    for label, prefix, color in series:
        axes[0].plot(
            threads,
            [row[f"{prefix}_qps"] for row in rows],
            "o-",
            color=color,
            label=label,
        )
        axes[1].plot(
            threads,
            [row[f"{prefix}_recall_percent"] for row in rows],
            "o-",
            color=color,
            label=label,
        )
        axes[2].plot(
            threads,
            [row[f"{prefix}_p99_ms"] for row in rows],
            "o-",
            color=color,
            label=label,
        )

    axes[0].set_title("Aggregate throughput")
    axes[0].set_ylabel("QPS")
    axes[1].set_title("Recall stability")
    axes[1].set_ylabel("Recall@100 (%)")
    axes[2].set_title("Tail latency")
    axes[2].set_ylabel("p99 latency (ms, log scale)")
    axes[2].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("Concurrent client threads")
        axis.set_xticks(threads)
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    figure.tight_layout(rect=(0, 0.07, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spann-log", required=True, type=Path)
    parser.add_argument("--pipeann-jsonl", required=True, type=Path)
    parser.add_argument("--nprobe", type=int, default=500)
    parser.add_argument("--pipeann-L", type=int, default=350)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    spann = select_rows(
        parse_result_lines(args.spann_log), "spann", "nprobe", args.nprobe
    )
    pipeann = select_rows(
        parse_result_lines(args.pipeann_jsonl), "pipeann", "L", args.pipeann_L
    )
    threads = require_shared_protocol(spann, pipeann)
    rows = build_rows(threads, spann, pipeann)

    csv_path = args.output_prefix.with_suffix(".csv")
    png_path = args.output_prefix.with_suffix(".png")
    write_csv(csv_path, rows)
    render(png_path, rows)

    for row in rows:
        print(
            f"threads={row['threads']:>2} "
            f"SPANN={row['spann_qps']:.2f} QPS @ {row['spann_recall_percent']:.3f}% "
            f"PipeANN={row['pipeann_qps']:.2f} QPS @ {row['pipeann_recall_percent']:.3f}% "
            f"ratio={row['pipeann_over_spann_qps']:.2f}x"
        )


if __name__ == "__main__":
    main()
