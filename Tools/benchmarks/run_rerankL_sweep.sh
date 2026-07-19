#!/usr/bin/env bash
set -u
cd /home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH=$PWD/Release:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PWD/Release
export SPTAG_UNFILTER_TAIL=1

IDX=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/spacev1b_opq25
INI=$IDX/tenant_0/indexloader.ini
QRY=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/multitenant/query
OUT=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/rerankL_k100.jsonl
NQ=${NQ:-500}
TOPK=100
NPROBE=${NPROBE:-6400}
export SPTAG_FIXED_NPROBE=$NPROBE

LS="${LS:-100 150 300 500 1000 2000}"

echo "# rerankL sweep start $(date) nq=$NQ topk=$TOPK nprobe=$NPROBE Ls=[$LS]" | tee -a "$OUT"
for L in $LS; do
  sed -i "s/^RerankL=.*/RerankL=$L/" "$INI"
  cur=$(grep '^RerankL=' "$INI")
  echo "=== RerankL=$L ($cur) $(date) ===" >&2
  timeout 3000 python3 Tools/benchmarks/sweep_curve_spacev1b.py \
    --index-dir "$IDX" --query-dir "$QRY" \
    --topk $TOPK --num-queries $NQ --warmup 100 2>>"$OUT.err" \
    | grep --line-buffered RESULT \
    | sed "s/}$/, \"rerankL\": $L}/" | tee -a "$OUT"
done
# restore baseline
sed -i "s/^RerankL=.*/RerankL=150/" "$INI"
echo "# rerankL sweep done $(date), ini restored to RerankL=150" | tee -a "$OUT"
