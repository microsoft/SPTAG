#!/usr/bin/env python3
"""Filtered + unfilter search benchmark for one SPANN tenant-0 index at a fixed nprobe.

Mirrors compare_uextra_unfilter.py but adds --level so the same harness can run
filtered queries (per-query tag from query_tags.npy) against the matching
precomputed groundtruth, as well as the unfilter (zero-tag) case.

  --level unfilter            -> SearchWithACL(..., b"", 0), GT groundtruth_unfilter
  --level org|dept|team|project -> SearchWithACL(..., qtag.tobytes(), 1), GT groundtruth_<level>

nprobe is controlled by SPTAG_FIXED_NPROBE (read once per process by the SPANN core),
so the driver re-invokes this per (index, level, nprobe). Emits one JSON line
prefixed 'RESULT '.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import SPTAG

DEFAULT_QUERY_DIR = "/home/v-mochengli/datasets/sift1m/multitenant/query"
LEVEL_COL = {"org": 0, "dept": 1, "team": 2, "project": 3}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index-dir", required=True)
    p.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    p.add_argument("--tenant", type=int, default=0)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--num-queries", type=int, default=0, help="0 = all")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--level", default="unfilter",
                   choices=["unfilter", "org", "dept", "team", "project"])
    p.add_argument("--label", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    qdir = Path(args.query_dir)
    queries = np.ascontiguousarray(np.load(qdir / "query_vectors.npy"), dtype=np.float32)

    if args.level == "unfilter":
        gt = np.load(qdir / "groundtruth_unfilter_local_ids.npy")
        qtags = None
    else:
        gt = np.load(qdir / f"groundtruth_{args.level}_local_ids.npy")
        col = LEVEL_COL[args.level]
        qtags = np.ascontiguousarray(np.load(qdir / "query_tags.npy")[:, col], dtype=np.uint32)

    if args.num_queries and args.num_queries < queries.shape[0]:
        queries = queries[: args.num_queries]
        gt = gt[: args.num_queries]
        if qtags is not None:
            qtags = qtags[: args.num_queries]
    num_queries, dim = queries.shape
    topk = args.topk

    mgr = SPTAG.CreateTenantIndexManager(dim, "SPANN", "Float")
    if not mgr.LoadAll(str(args.index_dir)):
        raise RuntimeError(f"LoadAll failed: {args.index_dir}")

    def do_search(i):
        if qtags is None:
            return mgr.SearchWithACL(queries[i].tobytes(), args.tenant, topk, b"", 0)
        tagb = np.asarray([qtags[i]], dtype=np.uint32).tobytes()
        return mgr.SearchWithACL(queries[i].tobytes(), args.tenant, topk, tagb, 1)

    for i in range(min(args.warmup, num_queries)):
        do_search(i)

    hit = 0
    denom = 0
    none_count = 0
    t0 = time.perf_counter()
    for i in range(num_queries):
        result = do_search(i)
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
        "level": args.level,
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
