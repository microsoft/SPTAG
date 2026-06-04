#!/usr/bin/env python3
"""Unfilter search benchmark for one SPANN tenant-0 index at a fixed nprobe.

Loads the tenant-0 index, runs every query as an *unfilter* search
(``SearchWithACL`` with zero tags), and reports recall@topk against the
precomputed cosine groundtruth plus throughput (QPS) and mean latency.

nprobe is controlled by the ``SPTAG_FIXED_NPROBE`` env var, which the SPANN
core reads once per process — so this script handles a single nprobe value and
the driver re-invokes it per (index, nprobe).

Emits one JSON object on stdout (prefixed ``RESULT ``) for easy parsing.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import SPTAG  # Release/SPTAG.py via PYTHONPATH


DEFAULT_QUERY_DIR = "/home/v-mochengli/datasets/sift1m/multitenant/query"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index-dir", required=True)
    p.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    p.add_argument("--tenant", type=int, default=0)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--num-queries", type=int, default=0, help="0 = all")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--label", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    qdir = Path(args.query_dir)
    queries = np.ascontiguousarray(np.load(qdir / "query_vectors.npy"), dtype=np.float32)
    gt = np.load(qdir / "groundtruth_unfilter_local_ids.npy")
    if args.num_queries and args.num_queries < queries.shape[0]:
        queries = queries[: args.num_queries]
        gt = gt[: args.num_queries]
    num_queries, dim = queries.shape
    topk = args.topk

    mgr = SPTAG.CreateTenantIndexManager(dim, "SPANN", "Float")
    if not mgr.LoadAll(str(args.index_dir)):
        raise RuntimeError(f"LoadAll failed: {args.index_dir}")

    empty_tags = b""

    # warm-up (also forces tenant load + FIXED_NPROBE caching)
    for i in range(min(args.warmup, num_queries)):
        mgr.SearchWithACL(queries[i].tobytes(), args.tenant, topk, empty_tags, 0)

    hit = 0
    denom = 0
    none_count = 0
    t0 = time.perf_counter()
    for i in range(num_queries):
        result = mgr.SearchWithACL(queries[i].tobytes(), args.tenant, topk, empty_tags, 0)
        if result is None:
            none_count += 1
            denom += topk
            continue
        ids = np.asarray(result[0], dtype=np.int64)
        dists = np.asarray(result[1], dtype=np.float32)
        valid = ids[(ids >= 0) & (dists < 1e30)][:topk]
        gt_row = gt[i]
        gt_valid = gt_row[gt_row >= 0]
        k = min(topk, gt_valid.size)
        if k > 0:
            hit += np.intersect1d(valid, gt_valid[:k]).size
            denom += k
    elapsed = time.perf_counter() - t0

    recall = hit / denom if denom else 0.0
    qps = num_queries / elapsed if elapsed > 0 else 0.0
    out = {
        "label": args.label or Path(args.index_dir).name,
        "index_dir": str(args.index_dir),
        "nprobe": int(os.environ.get("SPTAG_FIXED_NPROBE", "0")),
        "num_queries": int(num_queries),
        "topk": int(topk),
        "recall": round(recall, 4),
        "qps": round(qps, 1),
        "mean_latency_ms": round(1000.0 * elapsed / num_queries, 3),
        "none_results": int(none_count),
    }
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
