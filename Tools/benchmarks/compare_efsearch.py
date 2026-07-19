#!/usr/bin/env python3
"""Compare top-300 result-ID stability + recall across MaxCheck (efSearch) levels.

Reads top300_mc<MC>_np<NP>.npy dumps produced by efsearch_probe.py and, using the
LARGEST MaxCheck run as reference, reports for each level:
  - recall@100 (from results_np<NP>.jsonl)
  - mean per-query Jaccard of the top-300 ID set vs the reference run
  - mean # of top-300 slots that differ vs reference
  - fraction of queries whose top-300 set is IDENTICAL to reference
Answers: does raising efSearch actually change the (top-300) search output?
"""
import json
import sys
from pathlib import Path

import numpy as np


def load_results(path: Path):
    rows = {}
    for line in path.read_text().splitlines():
        if line.startswith("RESULT "):
            d = json.loads(line[len("RESULT "):])
            rows[int(d["maxcheck"])] = d
    return rows


def top_set(row):
    return set(int(x) for x in row if x >= 0)


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    nprobe = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    res = load_results(out_dir / f"results_np{nprobe}.jsonl")
    mcs = sorted(res)
    if not mcs:
        print("no results found")
        return
    ref_mc = mcs[-1]
    ref = np.load(out_dir / f"top300_mc{ref_mc}_np{nprobe}.npy")
    nq = ref.shape[0]

    print(f"nprobe={nprobe}  reference MaxCheck={ref_mc}  nq={nq}\n")
    hdr = f"{'MaxCheck':>10} {'recall@100':>11} {'QPS':>7} {'meanJaccard':>12} {'meanDiff/300':>13} {'%identical':>11}"
    print(hdr)
    print("-" * len(hdr))
    for mc in mcs:
        arr = np.load(out_dir / f"top300_mc{mc}_np{nprobe}.npy")
        jac = np.empty(nq)
        diff = np.empty(nq)
        identical = 0
        for i in range(nq):
            a = top_set(arr[i])
            b = top_set(ref[i])
            inter = len(a & b)
            union = len(a | b)
            jac[i] = inter / union if union else 1.0
            diff[i] = len(a ^ b) / 2.0
            if a == b:
                identical += 1
        d = res[mc]
        print(f"{mc:>10} {d['recall@100']:>11.4f} {d['qps']:>7.1f} "
              f"{jac.mean():>12.4f} {diff.mean():>13.2f} {100.0*identical/nq:>10.1f}%")


if __name__ == "__main__":
    main()
