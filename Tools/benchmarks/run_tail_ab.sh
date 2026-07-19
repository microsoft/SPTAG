#!/usr/bin/env bash
# A/B: does scanning the replica tail (SPTAG_UNFILTER_TAIL=0) fix org's recall
# deficit at fixed nprobe? One process per env config (static env reads).
set -euo pipefail
cd /home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH=$PWD/Release:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PWD/Release

INDEX_DIR=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/spacev1b_opq25
INI=$INDEX_DIR/tenant_0/indexloader.ini
QUERY_DIR=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/multitenant/query
OUT=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/tail_ab
mkdir -p "$OUT"
RES=$OUT/results.jsonl; : > "$RES"

cp -n "$INI" "$INI.tailbak"
sed -i 's/^RerankL=.*/RerankL=1000/' "$INI"
trap 'cp "$INI.tailbak" "$INI"; echo ini restored' EXIT

common="INDEX_DIR=$INDEX_DIR QUERY_DIR=$QUERY_DIR TENANT=0 TOPK=100 SPTAG_FIXED_NPROBE=400 NUM_QUERIES=2000 WARMUP=200 LEVELS=org TEST_MAXCHECK=4096"

echo "=== A: org default (tail skipped) $(date +%T) ==="
env $common SPTAG_LOG_PATH_STATS=1 python3 Tools/benchmarks/efsearch_probe_levels.py 2>&1 \
  | tee "$OUT/a_default.log" | grep '^RESULT' | sed 's/^RESULT /RESULT cfg=default /' >> "$RES" || true

echo "=== B: org SPTAG_UNFILTER_TAIL=0 (scan replica tail) $(date +%T) ==="
env $common SPTAG_UNFILTER_TAIL=0 python3 Tools/benchmarks/efsearch_probe_levels.py 2>&1 \
  | tee "$OUT/b_tail0.log" | grep '^RESULT' | sed 's/^RESULT /RESULT cfg=tail0 /' >> "$RES" || true

echo "=== C: org TAIL=0 + KEEP_UEXTRA=1 $(date +%T) ==="
env $common SPTAG_UNFILTER_TAIL=0 SPTAG_FILTER_KEEP_UEXTRA=1 python3 Tools/benchmarks/efsearch_probe_levels.py 2>&1 \
  | tee "$OUT/c_tail0_uextra.log" | grep '^RESULT' | sed 's/^RESULT /RESULT cfg=tail0_uextra /' >> "$RES" || true

echo "=== DONE ==="; cat "$RES"