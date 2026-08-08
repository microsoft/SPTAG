#!/usr/bin/env python3
"""Render the SIFT1B TopK-500 PipeANN/SPANN comparison."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


AUDIT_ROOT = Path(
    "/mnt/nvme/baotonglu/mocheng/pipeann/audits/"
    "sift1b_topk500_scale_check_20260720"
)


def load_rows(path: Path) -> list[dict]:
    rows = [
        json.loads(line[7:])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("RESULT ")
    ]
    if not rows:
        raise RuntimeError(f"No RESULT rows in {path}")
    return rows


def interpolate_pipeann(rows: list[dict], recall_percent: float) -> float:
    rows = sorted(rows, key=lambda row: row["recall_percent"])
    for lower, upper in zip(rows, rows[1:], strict=False):
        if lower["recall_percent"] <= recall_percent <= upper["recall_percent"]:
            span = upper["recall_percent"] - lower["recall_percent"]
            fraction = (recall_percent - lower["recall_percent"]) / span
            return lower["qps"] + fraction * (upper["qps"] - lower["qps"])
    raise RuntimeError(f"Cannot interpolate PipeANN at recall {recall_percent}")


def find_row(rows: list[dict], key: str, value: int) -> dict:
    for row in rows:
        if int(row[key]) == value:
            return row
    raise RuntimeError(f"Missing {key}={value}")


def main() -> None:
    pipeann = load_rows(AUDIT_ROOT / "pipeann" / "curve.jsonl")
    spann_500 = load_rows(
        AUDIT_ROOT / "spann" / "topk_500_rerank_l_500" / "curve.jsonl"
    )
    spann_1000 = load_rows(
        AUDIT_ROOT / "spann" / "topk_500_rerank_l_1000" / "curve.jsonl"
    )

    pipeann_90 = find_row(pipeann, "L", 1000)
    spann_match = find_row(spann_1000, "nprobe", 750)
    pipeann_at_match = interpolate_pipeann(pipeann, spann_match["recall"] * 100)
    comparison = {
        "protocol": {
            "dataset": "sift1b",
            "topk": 500,
            "value_type": "UInt8",
            "warmup_queries": 20,
            "measured_queries": 100,
            "threads": 1,
            "ground_truth": "official idx_1000M.ivecs, validated against existing TopK-100",
        },
        "pipeann_nearest_above_90": pipeann_90,
        "spann_near_90": spann_match,
        "pipeann_interpolated_qps_at_spann_recall": round(pipeann_at_match, 3),
        "pipeann_over_spann_qps_at_spann_recall": round(
            pipeann_at_match / spann_match["qps"], 3
        ),
        "spann_rerank_l_500": spann_500,
        "spann_rerank_l_1000": spann_1000,
        "topk500_phase_ms": {
            "nprobe": 750,
            "rerank_l": 1000,
            "postings": 750,
            "survivors": 1000,
            "posting_io": 12.571,
            "adc": 8.946,
            "rerank": 7.734,
            "post_total": 29.309,
            "graph_total": 4.901,
            "total": 34.583,
        },
    }
    (AUDIT_ROOT / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fig, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    axis.plot(
        [row["recall_percent"] for row in pipeann],
        [row["qps"] for row in pipeann],
        "o-",
        color="#1f77b4",
        label="PipeANN",
    )
    axis.plot(
        [row["recall"] * 100 for row in spann_500],
        [row["qps"] for row in spann_500],
        "s--",
        color="#ff9896",
        label="SPANN (RerankL=500)",
    )
    axis.plot(
        [row["recall"] * 100 for row in spann_1000],
        [row["qps"] for row in spann_1000],
        "s-",
        color="#d62728",
        label="SPANN (RerankL=1000)",
    )
    axis.axvline(90, color="black", linewidth=0.8, linestyle=":")
    axis.annotate(
        "PipeANN L=1000\n91.23%, 94.8 QPS",
        (pipeann_90["recall_percent"], pipeann_90["qps"]),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=8,
    )
    axis.annotate(
        "SPANN n=750, L=1000\n90.77%, 27.2 QPS",
        (spann_match["recall"] * 100, spann_match["qps"]),
        xytext=(8, -28),
        textcoords="offset points",
        fontsize=8,
    )
    axis.set_title("SIFT1B unfilter TopK=500")
    axis.set_xlabel("Recall@500 (%)")
    axis.set_ylabel("QPS")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(AUDIT_ROOT / "sift1b_topk500_spann_pipeann_recall_qps.png", dpi=200)


if __name__ == "__main__":
    main()
