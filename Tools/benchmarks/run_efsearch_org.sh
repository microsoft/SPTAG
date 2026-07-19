#!/usr/bin/env bash
# efSearch sweep for unfilter + org at fixed nprobe: is org head-nav-budget limited?
set -euo pipefail
cd /home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH=$PWD/Release:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PWD/Release

INDEX_DIR=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/spacev1b_opq25
INI=$INDEX_DIR/tenant_0/indexloader.ini
QUERY_DIR=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/multitenant/query
OUT_DIR=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/efsearch_org
mkdir -p "$OUT_DIR"

NPROBE=${NPROBE:-400}
NUM_QUERIES=${NUM_QUERIES:-2000}
MAXCHECKS=${MAXCHECKS:-"4096 16384 65536"}
LEVELS=${LEVELS:-"unfilter,org"}
RESULTS=$OUT_DIR/results_np${NPROBE}.jsonl
: > "$RESULTS"

cp -n "$INI" "$INI.orgbak"
sed -i 's/^RerankL=.*/RerankL=1000/' "$INI"
restore() { cp "$INI.orgbak" "$INI"; echo "ini restored"; }
trap restore EXIT

for MC in $MAXCHECKS; do
  awk -v mc="$MC" '
    /^\[/{sec=$0}
    sec=="[BuildSSDIndex]" && /^MaxCheck=/{print "MaxCheck=" mc; next}
    {print}
  ' "$INI" > "$INI.tmp" && mv "$INI.tmp" "$INI"
  echo "=== MaxCheck=$MC (nprobe=$NPROBE) $(date +%T) ==="
  INDEX_DIR=$INDEX_DIR QUERY_DIR=$QUERY_DIR TENANT=0 TOPK=100 \
    SPTAG_FIXED_NPROBE=$NPROBE TEST_MAXCHECK=$MC NUM_QUERIES=$NUM_QUERIES \
    WARMUP=200 LEVELS="$LEVELS" \
    python3 Tools/benchmarks/efsearch_probe_levels.py 2>&1 \
    | tee -a "$OUT_DIR/run_mc${MC}.log" | grep '^RESULT' >> "$RESULTS" || true
done
echo "=== ALL DONE ==="; cat "$RESULTS"
