#!/usr/bin/env bash
# efSearch (MaxCheck) sensitivity sweep for SPANN unfilter at a fixed nprobe.
# MaxCheck is consumed at LoadAll -> one index reload per value (~5 min each).
set -euo pipefail
cd /home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH=$PWD/Release:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PWD/Release

INDEX_DIR=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/spacev1b_opq25
INI=$INDEX_DIR/tenant_0/indexloader.ini
QUERY_DIR=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/multitenant/query
OUT_DIR=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/efsearch_probe
mkdir -p "$OUT_DIR"

NPROBE=${NPROBE:-400}
NUM_QUERIES=${NUM_QUERIES:-2000}
MAXCHECKS=${MAXCHECKS:-"4096 16384 65536 262144"}
RESULTS=$OUT_DIR/results_np${NPROBE}.jsonl
: > "$RESULTS"

# back up ini once
cp -n "$INI" "$INI.efbak"
# pin RerankL=1000 for all runs (isolate MaxCheck)
sed -i 's/^RerankL=.*/RerankL=1000/' "$INI"

restore() { cp "$INI.efbak" "$INI"; echo "ini restored"; }
trap restore EXIT

for MC in $MAXCHECKS; do
  # edit ONLY the [BuildSSDIndex] MaxCheck (line under [BuildSSDIndex])
  awk -v mc="$MC" '
    /^\[/{sec=$0}
    sec=="[BuildSSDIndex]" && /^MaxCheck=/{print "MaxCheck=" mc; next}
    {print}
  ' "$INI" > "$INI.tmp" && mv "$INI.tmp" "$INI"
  echo "=== MaxCheck=$MC (nprobe=$NPROBE) $(date +%T) ==="
  grep -nE '^MaxCheck=' "$INI"
  INDEX_DIR=$INDEX_DIR QUERY_DIR=$QUERY_DIR OUT_DIR=$OUT_DIR \
    TENANT=0 TOPK=300 SPTAG_FIXED_NPROBE=$NPROBE TEST_MAXCHECK=$MC \
    NUM_QUERIES=$NUM_QUERIES WARMUP=200 \
    python3 Tools/benchmarks/efsearch_probe.py 2>&1 | tee -a "$OUT_DIR/run_mc${MC}.log" | grep -E 'RESULT|Traceback|Error' \
    | grep '^RESULT' >> "$RESULTS" || true
done
echo "=== ALL DONE ==="; cat "$RESULTS"
