#!/usr/bin/env bash
# 2D sweep (nprobe x rerankL) for SPANN unfilter, to test if a lower-nprobe +
# higher-rerankL operating point can push the recall-QPS frontier up to overtake
# PipeANN unfilter. Recall ceiling at a given nprobe is capped by the retrieved
# candidate pool, so we probe whether high rerankL recovers recall cheaply (QPS).
set -u
cd /home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH=$PWD/Release:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PWD/Release
export SPTAG_UNFILTER_TAIL=1

IDX=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/spacev1b_opq25
INI=$IDX/tenant_0/indexloader.ini
QRY=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/multitenant/query
OUT=/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/unfilter_2d_k100.jsonl
NQ=${NQ:-500}
TOPK=100

# Focus on the high-QPS regime where PipeANN currently dominates (95-138 QPS).
NPROBES="${NPROBES:-100 200 400 800}"
LS="${LS:-500 1000 2000 4000}"

: > "$OUT"
echo "# unfilter 2D sweep start $(date) nq=$NQ topk=$TOPK nprobes=[$NPROBES] Ls=[$LS]" | tee -a "$OUT"
for np in $NPROBES; do
  for L in $LS; do
    sed -i "s/^RerankL=.*/RerankL=$L/" "$INI"
    echo "=== nprobe=$np RerankL=$L $(date) ===" >&2
    SPTAG_FIXED_NPROBE=$np timeout 3000 python3 Tools/benchmarks/sweep_curve_spacev1b.py \
      --index-dir "$IDX" --query-dir "$QRY" \
      --topk $TOPK --num-queries $NQ --warmup 100 2>>"$OUT.err" \
      | grep --line-buffered RESULT \
      | sed "s/}$/, \"rerankL\": $L, \"nprobe_env\": $np}/" | tee -a "$OUT"
  done
done
sed -i "s/^RerankL=.*/RerankL=150/" "$INI"
echo "ALL DONE unfilter 2D $(date), ini restored to RerankL=150" | tee -a "$OUT"
