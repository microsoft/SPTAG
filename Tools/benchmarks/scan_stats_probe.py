#!/usr/bin/env python3
"""Per-query scan-cost diagnostic: org vs unfilter at a FIXED nprobe.

Answers: at equal nprobe, does org scan FEWER vectors (coverage/replica deficit)
or scan the same but MATCH fewer (head-nav / posting-selection quality)?

Accumulates, per level, over NUM_QUERIES:
  readPostings   = postings actually read from SSD
  scannedVectors = vectors decoded/compared inside those postings
  matchedVectors = of those, how many pass the tag filter
plus recall@TOPK. Emits one RESULT json per level.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import SPTAG

LEVEL_COL = {"org": 0, "dept": 1, "team": 2, "project": 3}


def main() -> None:
    index_dir = os.environ["INDEX_DIR"]
    query_dir = os.environ["QUERY_DIR"]
    tenant = int(os.environ.get("TENANT", "0"))
    topk = int(os.environ.get("TOPK", "100"))
    nprobe = int(os.environ.get("SPTAG_FIXED_NPROBE", "0"))
    warmup = int(os.environ.get("WARMUP", "100"))
    num_q = int(os.environ.get("NUM_QUERIES", "1000"))
    levels = [s.strip() for s in os.environ.get("LEVELS", "unfilter,org").split(",") if s.strip()]

    qdir = Path(query_dir)
    qf = np.load(qdir / "query_vectors.npy")
    queries = np.ascontiguousarray(np.rint(qf).astype(np.int8))
    qtags_all = np.ascontiguousarray(np.load(qdir / "query_tags.npy"), dtype=np.uint32)
    nq_total, dim = queries.shape
    nq = nq_total if not num_q else min(num_q, nq_total)

    mgr = SPTAG.CreateTenantIndexManager(dim, "SPANN", "Int8")
    if not mgr.LoadAll(str(index_dir)):
        raise RuntimeError(f"LoadAll failed: {index_dir}")

    for level in levels:
        gt = np.load(qdir / f"groundtruth_{level}_local_ids.npy")
        qtags = None if level == "unfilter" else np.ascontiguousarray(qtags_all[:, LEVEL_COL[level]], dtype=np.uint32)

        def do_search(i):
            if qtags is None:
                return mgr.SearchWithACL(queries[i].tobytes(), tenant, topk, b"", 0)
            tagb = np.asarray([qtags[i]], dtype=np.uint32).tobytes()
            return mgr.SearchWithACL(queries[i].tobytes(), tenant, topk, tagb, 1)

        for i in range(min(warmup, nq)):
            do_search(i)

        hit = denom = 0
        read_p = scan_v = match_v = 0
        t0 = time.perf_counter()
        for i in range(nq):
            result = do_search(i)
            read_p += int(mgr.GetLastPostingReadCount())
            scan_v += int(mgr.GetLastScannedVectors())
            match_v += int(mgr.GetLastMatchedVectors())
            if result is None:
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

        out = {
            "level": level,
            "nprobe": nprobe,
            "num_queries": int(nq),
            "topk": int(topk),
            "recall": round(hit / denom if denom else 0.0, 4),
            "read_postings_per_q": round(read_p / nq, 1),
            "scanned_vecs_per_q": round(scan_v / nq, 1),
            "matched_vecs_per_q": round(match_v / nq, 1),
            "match_rate": round(match_v / scan_v, 4) if scan_v else 0.0,
            "vecs_per_posting": round(scan_v / read_p, 2) if read_p else 0.0,
            "qps": round(nq / elapsed, 1) if elapsed > 0 else 0.0,
        }
        print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
