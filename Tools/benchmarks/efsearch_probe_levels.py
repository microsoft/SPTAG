#!/usr/bin/env python3
"""efSearch (MaxCheck) sensitivity for MULTIPLE levels at a fixed nprobe.

Tests whether the org filtered path is head-nav-budget limited relative to the
unfilter global-head-graph path. MaxCheck is consumed at LoadAll (edited into
the ini by the driver), so one process per MaxCheck value; within the process
every requested level is benchmarked at the same fixed nprobe.

``MEASURE_OFFSET`` optionally selects a disjoint measured query range after
the warmup prefix. This makes it possible to use queries [0, warmup) solely for
warmup and [MEASURE_OFFSET, MEASURE_OFFSET + NUM_QUERIES) for scoring.

Emits one RESULT json per (level) with recall@TOPK.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import SPTAG

LEVEL_COL = {"org": 0, "dept": 1, "team": 2, "project": 3}


def read_ini_value(index_dir: str, tenant: int, key: str) -> str:
    config_path = Path(index_dir) / f"tenant_{tenant}" / "indexloader.ini"
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    raise ValueError(f"Missing {key} in {config_path}")


def read_ini_value_or_default(
    index_dir: str, tenant: int, key: str, default: str
) -> str:
    try:
        return read_ini_value(index_dir, tenant, key)
    except ValueError:
        return default


def parse_bool(value: str, key: str) -> bool:
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be one of 0/1, false/true, no/yes, or off/on")


def main() -> None:
    index_dir = os.environ["INDEX_DIR"]
    query_dir = os.environ["QUERY_DIR"]
    tenant = int(os.environ.get("TENANT", "0"))
    topk = int(os.environ.get("TOPK", "100"))
    maxcheck = int(os.environ.get("TEST_MAXCHECK", "0"))
    warmup = int(os.environ.get("WARMUP", "200"))
    num_q = int(os.environ.get("NUM_QUERIES", "2000"))
    measure_offset = int(os.environ.get("MEASURE_OFFSET", "0"))
    levels = [s.strip() for s in os.environ.get("LEVELS", "unfilter,org").split(",") if s.strip()]
    value_type = os.environ.get("SPTAG_VALUE_TYPE", "Int8")
    rerank_l = int(read_ini_value(index_dir, tenant, "RerankL"))
    configured_search_internal_result_num = int(
        read_ini_value(index_dir, tenant, "SearchInternalResultNum")
    )
    fixed_nprobe = int(
        read_ini_value_or_default(index_dir, tenant, "FixedNprobe", "0")
    )
    seed_max_check = int(
        read_ini_value_or_default(index_dir, tenant, "SeedMaxCheck", "0")
    )
    configured_force_dense_tag_search = parse_bool(
        read_ini_value(index_dir, tenant, "ForceDenseTagSearch"),
        "ForceDenseTagSearch",
    )
    force_dense_tag_search = configured_force_dense_tag_search

    qdir = Path(query_dir)
    qf = np.load(qdir / "query_vectors.npy")
    if value_type == "UInt8":
        queries = np.ascontiguousarray(np.rint(qf).astype(np.uint8))
    elif value_type == "Int8":
        queries = np.ascontiguousarray(np.rint(qf).astype(np.int8))
    elif value_type == "Float":
        queries = np.ascontiguousarray(qf.astype(np.float32))
    else:
        raise ValueError(f"Unsupported SPTAG_VALUE_TYPE: {value_type}")
    qtags_all = np.ascontiguousarray(np.load(qdir / "query_tags.npy"), dtype=np.uint32)
    nq_total, dim = queries.shape
    nq = nq_total if not num_q else min(num_q, nq_total)
    if measure_offset < 0 or measure_offset + nq > nq_total:
        raise ValueError(
            f"MEASURE_OFFSET={measure_offset} with NUM_QUERIES={nq} exceeds "
            f"{nq_total} available queries"
        )

    mgr = SPTAG.CreateTenantIndexManager(dim, "SPANN", value_type)
    t_load = time.perf_counter()
    if not mgr.LoadAll(str(index_dir)):
        raise RuntimeError(f"LoadAll failed: {index_dir}")
    load_s = time.perf_counter() - t_load
    for level in levels:
        gt = np.load(qdir / f"groundtruth_{level}_local_ids.npy")
        if level == "unfilter":
            qtags = None
        else:
            qtags = np.ascontiguousarray(qtags_all[:, LEVEL_COL[level]], dtype=np.uint32)

        def do_search(i):
            if qtags is None:
                return mgr.SearchWithACL(queries[i].tobytes(), tenant, topk, b"", 0)
            tagb = np.asarray([qtags[i]], dtype=np.uint32).tobytes()
            return mgr.SearchWithACL(queries[i].tobytes(), tenant, topk, tagb, 1)

        for i in range(min(warmup, nq_total)):
            do_search(i)

        hit = 0
        denom = 0
        read_postings = 0
        scanned_vectors = 0
        matched_vectors = 0
        primary_head_candidates = 0
        t0 = time.perf_counter()
        for measured_index in range(nq):
            i = measure_offset + measured_index
            result = do_search(i)
            read_postings += int(mgr.GetLastPostingReadCount())
            scanned_vectors += int(mgr.GetLastScannedVectors())
            matched_vectors += int(mgr.GetLastMatchedVectors())
            primary_head_candidates += int(mgr.GetLastPrimaryHeadCandidateCount())
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
            "maxcheck": maxcheck,
            "level": level,
            "nprobe": fixed_nprobe,
            "rerank_l": rerank_l,
            "fixed_nprobe": fixed_nprobe,
            "seed_max_check": seed_max_check,
            "configured_search_internal_result_num": configured_search_internal_result_num,
            "num_queries": int(nq),
            "measure_offset": int(measure_offset),
            "topk": int(topk),
            "value_type": value_type,
            "force_dense_tag_search": force_dense_tag_search,
            "recall": round(hit / denom if denom else 0.0, 4),
            "qps": round(nq / elapsed, 1) if elapsed > 0 else 0.0,
            "load_s": round(load_s, 1),
            "read_postings_per_q": round(read_postings / nq, 1),
            "scanned_vecs_per_q": round(scanned_vectors / nq, 1),
            "matched_vecs_per_q": round(matched_vectors / nq, 1),
            "primary_head_candidates_per_q": round(primary_head_candidates / nq, 1),
            "match_rate": round(matched_vectors / scanned_vectors, 6) if scanned_vectors else 0.0,
        }
        print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
