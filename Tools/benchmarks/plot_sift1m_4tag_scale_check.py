#!/usr/bin/env python3
"""Render the matched SIFT1M PipeANN/SPANN Recall@10 comparison."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


AUDIT_ROOT = Path(
    "/mnt/nvme/baotonglu/mocheng/pipeann/audits/sift1m_4tag_scale_check_20260720"
)
PIPEANN_CURVE = Path(
    "/mnt/nvme/baotonglu/mocheng/pipeann/sift1m_4tag_demo/search/curve.jsonl"
)
PIPEANN_REFINED = AUDIT_ROOT / "pipeann_unfilter_refine" / "curve.jsonl"
SPANN_ROOT = AUDIT_ROOT / "spann"


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("RESULT "):
            rows.append(json.loads(line[7:]))
    if not rows:
        raise RuntimeError(f"No RESULT rows in {path}")
    return rows


def pipeann_rows(rows: list[dict], level: str) -> list[dict]:
    return [row for row in rows if row["level"] == level]


def spann_rows(rows: list[dict], level: str) -> list[dict]:
    return [row for row in rows if row["level"] == level]


def pipeann_xy(rows: list[dict]) -> tuple[list[float], list[float]]:
    rows = sorted(rows, key=lambda row: row["recall_percent"])
    return [row["recall_percent"] for row in rows], [row["qps"] for row in rows]


def spann_xy(rows: list[dict]) -> tuple[list[float], list[float]]:
    rows = sorted(rows, key=lambda row: row["recall"])
    return [row["recall"] * 100 for row in rows], [row["qps"] for row in rows]


def find_row(rows: list[dict], key: str, value: int) -> dict:
    for row in rows:
        if int(row[key]) == value:
            return row
    raise RuntimeError(f"Missing {key}={value}")


def main() -> None:
    pipeann_base = load_rows(PIPEANN_CURVE)
    pipeann_refined = load_rows(PIPEANN_REFINED)
    spann_100 = load_rows(SPANN_ROOT / "rerank_l_100" / "curve.jsonl")
    spann_25 = load_rows(SPANN_ROOT / "rerank_l_25" / "curve.jsonl")

    pipeann_unfilter_by_l = {
        int(row["L"]): row for row in pipeann_rows(pipeann_base, "unfilter")
    }
    pipeann_unfilter_by_l.update(
        {int(row["L"]): row for row in pipeann_rows(pipeann_refined, "unfilter")}
    )
    pipeann_unfilter = list(pipeann_unfilter_by_l.values())
    pipeann_org = pipeann_rows(pipeann_base, "org")
    spann_100_unfilter = spann_rows(spann_100, "unfilter")
    spann_100_org = spann_rows(spann_100, "org")
    spann_25_unfilter = spann_rows(spann_25, "unfilter")

    pipeann_90 = find_row(pipeann_refined, "L", 17)
    spann_25_90 = find_row(spann_25_unfilter, "nprobe", 19)
    spann_100_90 = find_row(spann_100_unfilter, "nprobe", 17)
    pipeann_org_90 = find_row(pipeann_org, "L", 15)
    spann_org_90 = find_row(spann_100_org, "nprobe", 10)

    comparison = {
        "protocol": {
            "dataset": "SIFT1M four-tag demo",
            "value_type": "UInt8",
            "topk": 10,
            "warmup_queries": 20,
            "measured_queries": 100,
            "threads": 1,
            "ground_truth": "shared PipeANN exact UInt8 ground truth",
            "spann_runtime_control": "native INI overlays",
        },
        "unfilter": {
            "pipeann": pipeann_90,
            "spann_tuned": spann_25_90,
            "spann_canonical_rerank_l_100": spann_100_90,
            "pipeann_over_tuned_spann_qps": round(
                pipeann_90["qps"] / spann_25_90["qps"], 3
            ),
            "pipeann_over_rerank_l_100_spann_qps": round(
                pipeann_90["qps"] / spann_100_90["qps"], 3
            ),
        },
        "org_filter": {
            "pipeann": pipeann_org_90,
            "spann": spann_org_90,
            "spann_over_pipeann_qps": round(
                spann_org_90["qps"] / pipeann_org_90["qps"], 3
            ),
        },
    }
    (AUDIT_ROOT / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    unfilter, org = axes

    x, y = pipeann_xy(pipeann_unfilter)
    unfilter.plot(x, y, "o-", color="#1f77b4", label="PipeANN (pipe_search mode 2)")
    x, y = spann_xy(spann_25_unfilter)
    unfilter.plot(x, y, "s-", color="#d62728", label="SPANN (RerankL=25)")
    x, y = spann_xy(spann_100_unfilter)
    unfilter.plot(
        x, y, "s--", color="#ff9896", alpha=0.9, label="SPANN (RerankL=100)"
    )
    unfilter.axvline(90, color="black", linewidth=0.8, linestyle=":")
    unfilter.annotate(
        "PipeANN L=17\n90.5%, 1047 QPS",
        (pipeann_90["recall_percent"], pipeann_90["qps"]),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=8,
    )
    unfilter.annotate(
        "SPANN n=19, L=25\n90.4%, 913 QPS",
        (spann_25_90["recall"] * 100, spann_25_90["qps"]),
        xytext=(8, -28),
        textcoords="offset points",
        fontsize=8,
    )
    unfilter.set_title("Unfilter")

    x, y = pipeann_xy(pipeann_org)
    org.plot(x, y, "o-", color="#1f77b4", label="PipeANN")
    x, y = spann_xy(spann_100_org)
    org.plot(x, y, "s-", color="#d62728", label="SPANN (RerankL=100)")
    org.axvline(90, color="black", linewidth=0.8, linestyle=":")
    org.set_title("Org filter")

    for axis in axes:
        axis.set_xlabel("Recall@10 (%)")
        axis.set_ylabel("QPS")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    fig.savefig(AUDIT_ROOT / "sift1m_spann_pipeann_recall_qps.png", dpi=200)


if __name__ == "__main__":
    main()
