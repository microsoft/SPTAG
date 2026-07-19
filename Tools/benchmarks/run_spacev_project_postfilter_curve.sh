#!/usr/bin/env bash
set -euo pipefail

cd /home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH="$PWD/Release:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/Release"

IDX=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/spacev1b_opq25
QRY=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/multitenant/query
OUT="${OUT:-/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/curve_k100_L1000_postfilter_project.jsonl}"
NQ="${NQ:-500}"
WARMUP="${WARMUP:-100}"
NPROBES="${NPROBES:-100 200 400 800 1600 3200 6400 12800}"

if ! grep -qx 'RerankL=1000' "$IDX/tenant_0/indexloader.ini"; then
  echo "Expected RerankL=1000 in $IDX/tenant_0/indexloader.ini" >&2
  exit 1
fi

tmp="${OUT}.tmp"
: > "$tmp"
printf '# SPACEV project post-filter curve start %s nq=%s nprobes=[%s]\n' \
  "$(date -u +%FT%TZ)" "$NQ" "$NPROBES" | tee -a "$tmp"

for nprobe in $NPROBES; do
  printf '=== project nprobe=%s %s ===\n' "$nprobe" "$(date -u +%FT%TZ)" | tee -a "$tmp"
  SPTAG_FIXED_NPROBE="$nprobe" \
    python3 Tools/benchmarks/sweep_curve_spacev1b.py \
      --index-dir "$IDX" --query-dir "$QRY" \
      --topk 100 --num-queries "$NQ" --warmup "$WARMUP" --levels project \
      2>&1 | tee -a "$tmp"
done

mv "$tmp" "$OUT"
printf '# SPACEV project post-filter curve done %s\n' "$(date -u +%FT%TZ)" | tee -a "$OUT"
