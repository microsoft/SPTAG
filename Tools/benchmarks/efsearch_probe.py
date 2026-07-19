#!/usr/bin/env python3
"""efSearch (MaxCheck) sensitivity probe.

Fix nprobe, vary head-graph search budget MaxCheck (baked at LoadAll into the
global head graph), and measure whether the top-300 result IDs and recall@100
actually change for the UNFILTER path.

MaxCheck is set by editing indexloader.ini before this process starts (it is
consumed at LoadAll), so each MaxCheck value = one process invocation. The
driver passes the value via env TEST_MAXCHECK purely for labelling/output.

Dumps top-300 result IDs to <out-dir>/top300_mc<MaxCheck>_np<nprobe>.npy so a
separate compare step can compute set overlap across MaxCheck levels.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import SPTAG


def main() -> None:
    index_dir = os.environ["INDEX_DIR"]
    query_dir = os.environ["QUERY_DIR"]
    out_dir = Path(os.environ.get("OUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    tenant = int(os.environ.get("TENANT", "0"))
    topk = int(os.environ.get("TOPK", "300"))
    nprobe = int(os.environ.get("SPTAG_FIXED_NPROBE", "0"))
    maxcheck = int(os.environ.get("TEST_MAXCHECK", "0"))
    warmup = int(os.environ.get("WARMUP", "200"))
    num_q = int(os.environ.get("NUM_QUERIES", "2000"))

    qdir = Path(query_dir)
    qf = np.load(qdir / "query_vectors.npy")
    queries = np.ascontiguousarray(np.rint(qf).astype(np.int8))
    nq_total, dim = queries.shape
    nq = nq_total if not num_q else min(num_q, nq_total)
    gt = np.load(qdir / "groundtruth_unfilter_local_ids.npy")

    mgr = SPTAG.CreateTenantIndexManager(dim, "SPANN", "Int8")
    t_load = time.perf_counter()
    if not mgr.LoadAll(str(index_dir)):
        raise RuntimeError(f"LoadAll failed: {index_dir}")
    load_s = time.perf_counter() - t_load

    def do_search(i):
        return mgr.SearchWithACL(queries[i].tobytes(), tenant, topk, b"", 0)

    for i in range(min(warmup, nq)):
        do_search(i)

    top_ids = np.full((nq, topk), -1, dtype=np.int64)
    hit100 = 0
    denom100 = 0
    none_count = 0
    t0 = time.perf_counter()
    for i in range(nq):
        result = do_search(i)
        if result is None:
            none_count += 1
            denom100 += 100
            continue
        ids = np.asarray(result[0], dtype=np.int64)
        dists = np.asarray(result[1], dtype=np.float32)
        valid = ids.copy()
        valid[(ids < 0) | (dists >= 1e30)] = -1
        top_ids[i, : min(topk, valid.size)] = valid[:topk]
        # recall@100
        v100 = valid[valid >= 0][:100]
        gt_row = gt[i]
        gt_valid = gt_row[gt_row >= 0][:100]
        if gt_valid.size:
            hit100 += np.intersect1d(v100, gt_valid).size
            denom100 += gt_valid.size
    elapsed = time.perf_counter() - t0

    recall100 = hit100 / denom100 if denom100 else 0.0
    dump = out_dir / f"top300_mc{maxcheck}_np{nprobe}.npy"
    np.save(dump, top_ids)

    out = {
        "maxcheck": maxcheck,
        "nprobe": nprobe,
        "num_queries": int(nq),
        "topk": int(topk),
        "recall@100": round(recall100, 4),
        "qps": round(nq / elapsed, 1) if elapsed > 0 else 0.0,
        "mean_latency_ms": round(1000.0 * elapsed / nq, 3),
        "none_results": int(none_count),
        "load_s": round(load_s, 1),
        "dump": str(dump),
    }
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
