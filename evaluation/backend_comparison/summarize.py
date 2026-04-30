#!/usr/bin/env python3
"""Summarize SPTAG backend benchmark JSON results into a markdown table row.

Usage:
    summarize.py <result.json> [<result.json> ...]

Or:
    summarize.py --remote 10.11.0.7:~/zhangt/SPTAG/evaluation/backend_comparison/results/*.json
"""
import json
import sys
from pathlib import Path


def summarize_one(path):
    with open(path) as f:
        d = json.load(f)
    cfg = d.get("config", {})
    res = d.get("results", {})

    name = Path(path).stem
    base = cfg.get("baseVectorCount", "?")
    ins = cfg.get("insertVectorCount", "?")
    layers = cfg.get("layers", "?")
    build_s = res.get("build timeSeconds", "?")

    # Pre-insert query
    b0 = res.get("benchmark0_query_before_insert", {})
    pre_qps = b0.get("qps")
    pre_recall = (b0.get("recall") or {}).get("recallAtK")
    pre_lat = b0.get("meanLatency")

    # Insert benchmark — average across batches
    b1 = res.get("benchmark1_insert", {})
    batches = sorted(k for k in b1 if k.startswith("batch_"))
    ins_throughputs = []
    sd_qps = []
    sd_recalls = []
    for bk in batches:
        b = b1[bk]
        if "insert throughput" in b:
            ins_throughputs.append(b["insert throughput"])
        s = b.get("search") or {}
        if s.get("qps") is not None:
            sd_qps.append(s["qps"])
        r = (s.get("recall") or {}).get("recallAtK")
        if r is not None:
            sd_recalls.append(r)

    # Post-insert query
    b2 = res.get("benchmark2_query_after", {}) or res.get("benchmark2_query_after_insert_delete", {})
    post_qps = b2.get("qps") if b2 else None
    post_recall = (b2.get("recall") or {}).get("recallAtK") if b2 else None
    post_lat = b2.get("meanLatency") if b2 else None

    # If b2 missing, find any "Benchmark 2" data in benchmark2 keys
    if post_qps is None:
        for k in res:
            if k.startswith("benchmark2"):
                v = res[k]
                if isinstance(v, dict) and "qps" in v:
                    post_qps = v["qps"]
                    post_lat = v.get("meanLatency")
                    post_recall = (v.get("recall") or {}).get("recallAtK")
                    break

    def fmt(x, p=2):
        return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"

    avg_throughput = (sum(ins_throughputs) / len(ins_throughputs)) if ins_throughputs else None
    avg_sd_qps = (sum(sd_qps) / len(sd_qps)) if sd_qps else None
    avg_sd_recall = (sum(sd_recalls) / len(sd_recalls)) if sd_recalls else None

    print(f"## {name}")
    print(f"- Scale: base={base:,}, insert={ins:,}, layers={layers}")
    print(f"- Build: {fmt(build_s, 1)} s")
    print(f"- Pre-insert search: QPS={fmt(pre_qps,1)}, latency={fmt(pre_lat)}ms, recall5@5={fmt(pre_recall,4)}")
    print(f"- Insert phase ({len(batches)} batches):")
    print(f"    avg insert throughput = {fmt(avg_throughput,1)} vec/s")
    print(f"    avg search-during-insert QPS = {fmt(avg_sd_qps,1)}, recall5@5 = {fmt(avg_sd_recall,4)}")
    print(f"- Post-insert search: QPS={fmt(post_qps,1)}, latency={fmt(post_lat)}ms, recall5@5={fmt(post_recall,4)}")
    print()


if __name__ == "__main__":
    for path in sys.argv[1:]:
        try:
            summarize_one(path)
        except Exception as e:
            print(f"# {path}: ERROR {e}")
