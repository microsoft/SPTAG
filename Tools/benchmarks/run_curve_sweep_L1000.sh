#!/usr/bin/env bash
set -u
cd /home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH=$PWD/Release:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PWD/Release
export SPTAG_UNFILTER_TAIL=1

IDX=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/spacev1b_opq25
INI=$IDX/tenant_0/indexloader.ini
QRY=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/multitenant/query
OUT=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/curve_k100_L1000.jsonl
NQ=${NQ:-500}
TOPK=100
L=${L:-1000}
NPROBES="${NPROBES:-100 200 400 800 1600 3200 6400 12800}"

sed -i "s/^RerankL=.*/RerankL=$L/" "$INI"
echo "# curve(L=$L) start $(date) nq=$NQ topk=$TOPK $(grep '^RerankL=' $INI) nprobes=[$NPROBES]" | tee -a "$OUT"
for np in $NPROBES; do
  echo "=== nprobe=$np $(date) ===" >&2
  SPTAG_FIXED_NPROBE=$np timeout 3000 python3 Tools/benchmarks/sweep_curve_spacev1b.py \
    --index-dir "$IDX" --query-dir "$QRY" \
    --topk $TOPK --num-queries $NQ --warmup 100 2>>"$OUT.err" \
    | grep --line-buffered RESULT | sed "s/}$/, \"rerankL\": $L}/" | tee -a "$OUT"
done
sed -i "s/^RerankL=.*/RerankL=150/" "$INI"
echo "# curve(L=$L) done $(date), ini restored to RerankL=150" | tee -a "$OUT"
