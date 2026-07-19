#!/usr/bin/env bash
set -euo pipefail

cd /home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH="$PWD/Release:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/Release"

IDX="${IDX:-/mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_spann_pipepq32_r010_tail1}"
INI="$IDX/tenant_0/indexloader.ini"
QRY="${QRY:-/mnt/nvme/baotonglu/mocheng/datasets/sift1b/multitenant/query}"
OUT="${OUT:-/mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_postfilter_curves_r010_tail1_L500.jsonl}"
LOG="${LOG:-${OUT}.log}"
NQ="${NQ:-100}"
WARMUP="${WARMUP:-20}"
TOPK="${TOPK:-100}"
MEASURE_OFFSET="${MEASURE_OFFSET:-0}"
NPROBES="${NPROBES:-50 100 200 400 800 1600 3200 6400}"
LEVELS="unfilter,org,dept,team,project"
FORCE_DENSE_TAG_SEARCH="${FORCE_DENSE_TAG_SEARCH:-true}"
export SPTAG_UNFILTER_TAIL=1
export SPTAG_CROSSEDGE_UNFILTER=1

[[ -f "$INI" ]] || { echo "Missing index config: $INI" >&2; exit 1; }
[[ -s "$IDX/tenant_0/signatures_bitmask.bin" ]] ||
  { echo "Missing signatures_bitmask.bin: $IDX" >&2; exit 1; }
[[ -s "$IDX/tenant_0/HeadIndex/head_node_meta.bin" ]] ||
  { echo "Missing head_node_meta.bin: $IDX" >&2; exit 1; }
[[ -s "$IDX/tenant_0/HeadIndex/tag_node_index.bin" ]] ||
  { echo "Missing tag_node_index.bin: $IDX" >&2; exit 1; }
[[ -s "$IDX/tenant_0/HeadIndex/head_cross_edges.bin" ]] ||
  { echo "Missing head_cross_edges.bin: $IDX" >&2; exit 1; }
for expected in \
  'EnablePrimaryHeadBypass=false' \
  'RerankL=500' \
  'TailReplicaCount=8' \
  'UnfilterTailBufferLength=1' \
  'PostingQuantizer=PipePQ' \
  'PostingQuantM=32'; do
  grep -qx "$expected" "$INI" ||
    { echo "Expected $expected in $INI" >&2; exit 1; }
done

tmp="${OUT}.tmp"
rm -f "${LOG}".*.tmp "${tmp}".*.result
: > "$tmp"
: > "$LOG"
printf '# SIFT1B distance-first posting post-filter curve start %s nq=%s warmup=%s topk=%s measure_offset=%s nprobes=[%s]\n' \
  "$(date -u +%FT%TZ)" "$NQ" "$WARMUP" "$TOPK" "$MEASURE_OFFSET" "$NPROBES" | tee -a "$tmp" "$LOG"

for nprobe in $NPROBES; do
  printf '=== nprobe=%s %s ===\n' "$nprobe" "$(date -u +%FT%TZ)" | tee -a "$tmp" "$LOG"
  result_tmp="${tmp}.${nprobe}.result"
  run_log="${LOG}.${nprobe}.tmp"
  : > "$result_tmp"
  SPTAG_FIXED_NPROBE="$nprobe" \
  SPTAG_TAG_AWARE_HEAD_EXPANSION=1 \
  SPTAG_DISABLE_HIER_POSTING_FILTER=1 \
  FORCE_DENSE_TAG_SEARCH="$FORCE_DENSE_TAG_SEARCH" \
  INDEX_DIR="$IDX" \
  QUERY_DIR="$QRY" \
  NUM_QUERIES="$NQ" \
  WARMUP="$WARMUP" \
  TOPK="$TOPK" \
  MEASURE_OFFSET="$MEASURE_OFFSET" \
  LEVELS="$LEVELS" \
  SPTAG_VALUE_TYPE=UInt8 \
    python3 Tools/benchmarks/efsearch_probe_levels.py \
      > "$run_log" 2>&1
  cat "$run_log" >> "$LOG"
  awk '/^RESULT / { print }' "$run_log" > "$result_tmp"
  rm -f "$run_log"

  python3 - "$result_tmp" "$nprobe" <<'PY' >> "$tmp"
import json
import sys

path, nprobe = sys.argv[1], int(sys.argv[2])
expected = {"unfilter", "org", "dept", "team", "project"}
rows = []
for line in open(path, encoding="utf-8"):
    if line.startswith("RESULT "):
        rows.append(json.loads(line[7:]))
assert {row["level"] for row in rows} == expected, rows
assert len(rows) == len(expected), rows
assert all(row["nprobe"] == nprobe for row in rows), rows
assert all(row["primary_head_candidates_per_q"] == 0.0 for row in rows), rows
assert all(
    row["read_postings_per_q"] > 0.0
    for row in rows
    if row["level"] != "unfilter"
), rows
for row in rows:
    row["policy"] = "distance-first-post-filter"
    print("RESULT " + json.dumps(row, sort_keys=True))
PY
  rm -f "$result_tmp"
done

mv "$tmp" "$OUT"
printf '# SIFT1B distance-first posting post-filter curve done %s\n' "$(date -u +%FT%TZ)" | tee -a "$OUT" "$LOG"
