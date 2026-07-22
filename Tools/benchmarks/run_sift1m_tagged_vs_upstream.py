#!/usr/bin/env python3
"""Measure matched unfiltered SIFT1M QPS/recall for tagged and vanilla SPANN."""

import argparse
import json
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/datadisk/yfcc_fast/sptag_sift1m_tagged_vs_upstream")
SUMMARY = re.compile(r"^\[\d+\]\s+0-(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")


def parse_nprobes(value: str) -> list[int]:
    nprobes = [int(part) for part in value.split(",") if part]
    if not nprobes or any(nprobe <= 0 for nprobe in nprobes):
        raise argparse.ArgumentTypeError("nprobes must be positive comma-separated integers")
    return nprobes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark" / "matched_unfilter",
    )
    parser.add_argument("--query-file", type=Path, default=ROOT / "tenant0_query_1000.fvecs")
    parser.add_argument("--truth-file", type=Path, default=ROOT / "tenant0_unfilter_top10_1000.truth.bin")
    parser.add_argument("--tagged-index", type=Path)
    parser.add_argument("--upstream-index", type=Path)
    parser.add_argument("--tagged-searcher", type=Path, default=Path("/home/v-mochengli/SPTAG/Release/indexsearcher"))
    parser.add_argument("--upstream-searcher", type=Path, default=Path("/home/v-mochengli/SPTAG-upstream/Release/indexsearcher"))
    parser.add_argument("--tagged-name", default="tagged_current")
    parser.add_argument("--upstream-name", default="vanilla_upstream")
    parser.add_argument("--nprobes", type=parse_nprobes, default=[32, 64, 96, 128, 192, 256])
    parser.add_argument("--warmup-nprobe", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-check", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def command(searcher: Path, index: Path, truth: Path, args: argparse.Namespace, nprobe: int) -> list[str]:
    return [
        str(searcher),
        "-v", "Float",
        "-d", "128",
        "-f", "XVEC",
        "-i", str(args.query_file),
        "-x", str(index),
        "-r", str(truth),
        "-k", str(args.top_k),
        "-tk", str(args.top_k),
        "-t", str(args.threads),
        "-b", str(args.batch_size),
        "-m", str(args.max_check),
        f"BuildSSDIndex.SearchInternalResultNum={nprobe}",
        "BuildSSDIndex.AsyncMergeInSearch=false",
    ]


def parse_summary(output: str, expected_count: int, expected_max_check: int) -> tuple[float, float]:
    for line in reversed(output.splitlines()):
        match = SUMMARY.match(line)
        if not match:
            continue
        count, max_check, _, _, _, recall, qps = match.groups()
        if int(count) == expected_count and int(max_check) == expected_max_check:
            return float(recall), float(qps)
    raise RuntimeError("indexsearcher output did not contain a final summary line")


def run_once(
    searcher: Path,
    index: Path,
    truth: Path,
    args: argparse.Namespace,
    nprobe: int,
    workdir: Path,
) -> tuple[float, float]:
    workdir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command(searcher, index, truth, args, nprobe),
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (workdir / "indexsearcher.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"indexsearcher failed for {index}; see {workdir / 'indexsearcher.log'}")
    return parse_summary(result.stdout, args.batch_size, args.max_check)


def write_results(path: Path, results: dict[str, object]) -> None:
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least one")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {args.output_dir}")
    if args.tagged_name == args.upstream_name:
        raise ValueError("benchmark case names must differ")

    root = args.root
    truth = args.truth_file
    cases = {
        args.tagged_name: {
            "searcher": args.tagged_searcher,
            "index": args.tagged_index or root / "index_tagged_current" / "tenant_0",
        },
        args.upstream_name: {
            "searcher": args.upstream_searcher,
            "index": args.upstream_index or root / "index_vanilla_upstream",
        },
    }
    for case in cases.values():
        for path in (case["searcher"], case["index"], truth, args.query_file):
            if not path.exists():
                raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "query": str(args.query_file),
        "truth": str(truth),
        "threads": args.threads,
        "max_check": args.max_check,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "nprobes": args.nprobes,
        "warmup_nprobe": args.warmup_nprobe,
        "repeats": args.repeats,
        "async_merge_in_search": False,
        "cases": {},
    }
    output = args.output_dir / "results.json"

    for name, case in cases.items():
        searcher = case["searcher"]
        index = case["index"]
        print(f"[{name}] warmup nprobe={args.warmup_nprobe}", flush=True)
        run_once(searcher, index, truth, args, args.warmup_nprobe, args.output_dir / name / "warmup")

        case_results: dict[str, object] = {
            "searcher": str(searcher),
            "index": str(index),
            "nprobes": {},
        }
        results["cases"][name] = case_results
        for nprobe in args.nprobes:
            samples = []
            for repeat in range(args.repeats):
                print(f"[{name}] nprobe={nprobe} repeat={repeat + 1}/{args.repeats}", flush=True)
                recall, qps = run_once(
                    searcher,
                    index,
                    truth,
                    args,
                    nprobe,
                    args.output_dir / name / f"nprobe_{nprobe}" / f"repeat_{repeat + 1}",
                )
                samples.append({"recall": recall, "qps": qps})
            case_results["nprobes"][str(nprobe)] = {
                "samples": samples,
                "recall_median": statistics.median(sample["recall"] for sample in samples),
                "qps_median": statistics.median(sample["qps"] for sample in samples),
                "qps_min": min(sample["qps"] for sample in samples),
                "qps_max": max(sample["qps"] for sample in samples),
            }
            write_results(output, results)

    write_results(output, results)
    print(f"results: {output}")


if __name__ == "__main__":
    main()
