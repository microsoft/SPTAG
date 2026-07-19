#!/usr/bin/env bash
# Full recall/QPS curve across all 5 filter levels after the routing sidecar
# (tag_node_index.bin) was regenerated. One process per nprobe (env override);
# each process benchmarks every level at that fixed nprobe. Warm page cache =>
# reload is ~free. Emits one JSONL row per (nprobe, level).
set -u
ROOT=/home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH=$ROOT/Release:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT/Release
export INDEX_DIR=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/spacev1b_opq25
export QUERY_DIR=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/multitenant/query
export TENANT=0 TOPK=100 WARMUP=200 NUM_QUERIES=${NUM_QUERIES:-2000}
export LEVELS=${LEVELS:-unfilter,org,dept,team,project}

OUT=${OUT:-/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/routing_curve}
mkdir -p "$OUT"
JSONL="$OUT/curve_k100.jsonl"
: > "$JSONL"
GRID=${GRID:-"20 40 60 100 150 200 300 400"}

for np in $GRID; do
  echo "=== nprobe=$np $(date +%H:%M:%S) ===" | tee -a "$OUT/driver.log"
  SPTAG_FIXED_NPROBE=$np python3 "$ROOT/Tools/benchmarks/efsearch_probe_levels.py" \
      > "$OUT/np_${np}.log" 2>&1
  grep '^RESULT ' "$OUT/np_${np}.log" | sed "s/^RESULT //" >> "$JSONL"
  grep '^RESULT ' "$OUT/np_${np}.log" | tee -a "$OUT/driver.log"
done
echo "=== DONE $(date +%H:%M:%S) ===" | tee -a "$OUT/driver.log"
