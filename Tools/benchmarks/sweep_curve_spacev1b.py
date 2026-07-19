#!/usr/bin/env python3
"""SPACEV-1B recall/QPS sweep: one nprobe per process, all ACL levels inside.

The SPANN core caches SPTAG_FIXED_NPROBE once per process, so nprobe is fixed
per invocation (the outer launcher re-invokes per nprobe). Within one process we
load the index ONCE and benchmark every requested level (unfilter + the 4 ACL
levels) to amortise the ~5 min LoadAll.

CRITICAL: the index is Int8. Queries MUST be passed as int8 bytes and the
manager MUST be created as "Int8"; query_vectors.npy holds exact int8 values.

Emits one JSON line per level prefixed 'RESULT '.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import SPTAG

LEVEL_COL = {"org": 0, "dept": 1, "team": 2, "project": 3}
ALL_LEVELS = ["unfilter", "org", "dept", "team", "project"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index-dir", required=True)
    p.add_argument("--query-dir", required=True)
    p.add_argument("--tenant", type=int, default=0)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--num-queries", type=int, default=0, help="0 = all")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--levels", default=",".join(ALL_LEVELS))
    p.add_argument("--label", default="spacev1b")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    qdir = Path(args.query_dir)
    nprobe = int(os.environ.get("SPTAG_FIXED_NPROBE", "0"))
    levels = [s.strip() for s in args.levels.split(",") if s.strip()]

    # int8 queries (query_vectors.npy holds exact int8 values)
    qf = np.load(qdir / "query_vectors.npy")
    queries = np.ascontiguousarray(np.rint(qf).astype(np.int8))
    qtags_all = np.ascontiguousarray(np.load(qdir / "query_tags.npy"), dtype=np.uint32)
    nq_total, dim = queries.shape
    nq = nq_total if not args.num_queries else min(args.num_queries, nq_total)
    topk = args.topk

    mgr = SPTAG.CreateTenantIndexManager(dim, "SPANN", "Int8")
    t_load = time.perf_counter()
    if not mgr.LoadAll(str(args.index_dir)):
        raise RuntimeError(f"LoadAll failed: {args.index_dir}")
    load_s = time.perf_counter() - t_load

    for level in levels:
        gt = np.load(qdir / f"groundtruth_{level}_local_ids.npy")
        if level == "unfilter":
            qtags = None
        else:
            col = LEVEL_COL[level]
            qtags = np.ascontiguousarray(qtags_all[:, col], dtype=np.uint32)

        def do_search(i):
            if qtags is None:
                return mgr.SearchWithACL(queries[i].tobytes(), args.tenant, topk, b"", 0)
            tagb = np.asarray([qtags[i]], dtype=np.uint32).tobytes()
            return mgr.SearchWithACL(queries[i].tobytes(), args.tenant, topk, tagb, 1)

        for i in range(min(args.warmup, nq)):
            do_search(i)

        hit = 0
        denom = 0
        none_count = 0
        t0 = time.perf_counter()
        for i in range(nq):
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
        qps = nq / elapsed if elapsed > 0 else 0.0
        out = {
            "label": args.label,
            "level": level,
            "nprobe": nprobe,
            "num_queries": int(nq),
            "topk": int(topk),
            "recall": round(recall, 4),
            "qps": round(qps, 1),
            "mean_latency_ms": round(1000.0 * elapsed / nq, 3),
            "none_results": int(none_count),
            "load_s": round(load_s, 1),
        }
        print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
