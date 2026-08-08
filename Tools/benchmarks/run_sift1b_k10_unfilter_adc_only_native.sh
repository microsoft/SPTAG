#!/usr/bin/env bash
set -euo pipefail

cd /home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH="$PWD/Release:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/Release"

SOURCE_INDEX=/mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_spann_pipepq32_r010_tail1
SOURCE_INI="$SOURCE_INDEX/tenant_0/indexloader.ini"
QUERY_DIR=/mnt/nvme/baotonglu/mocheng/datasets/sift1b/multitenant/query
OUT="${OUT:-/mnt/nvme/baotonglu/mocheng/pipeann/audits/sift1b_k10_unfilter_adc_only_native_20260720}"
OVERLAY="$OUT/overlay"
TENANT_DIR="$OVERLAY/tenant_0"
LOG="$OUT/run.log"
RESULTS="$OUT/curve.jsonl"

[[ -f "$SOURCE_INI" ]] || { echo "Missing source INI: $SOURCE_INI" >&2; exit 1; }
[[ -f "$SOURCE_INDEX/manifest.txt" ]] || { echo "Missing manifest: $SOURCE_INDEX" >&2; exit 1; }
[[ -d "$QUERY_DIR" ]] || { echo "Missing query directory: $QUERY_DIR" >&2; exit 1; }
if [[ -e "$RESULTS" && "${SPTAG_OVERWRITE:-0}" != "1" ]]; then
    echo "Refusing to overwrite $RESULTS; set SPTAG_OVERWRITE=1 to replace it." >&2
    exit 1
fi

mkdir -p "$TENANT_DIR"
ln -sfn "$SOURCE_INDEX/manifest.txt" "$OVERLAY/manifest.txt"
for entry in "$SOURCE_INDEX/tenant_0"/*; do
    name="$(basename "$entry")"
    [[ "$name" == indexloader.ini ]] && continue
    ln -sfn "$entry" "$TENANT_DIR/$name"
done

cp "$SOURCE_INI" "$TENANT_DIR/indexloader.ini"
for key in RerankL SearchInternalResultNum FixedNprobe ForceDenseTagSearch \
           LogPhaseTime QuantADCOnly MaxDistRatio; do
    sed -i "/^${key}=/d" "$TENANT_DIR/indexloader.ini"
done
cat >> "$TENANT_DIR/indexloader.ini" <<'EOF'
RerankL=500
SearchInternalResultNum=120
FixedNprobe=120
ForceDenseTagSearch=true
LogPhaseTime=true
QuantADCOnly=true
MaxDistRatio=8
EOF

grep -qx 'RerankL=500' "$TENANT_DIR/indexloader.ini"
grep -qx 'SearchInternalResultNum=120' "$TENANT_DIR/indexloader.ini"
grep -qx 'FixedNprobe=120' "$TENANT_DIR/indexloader.ini"
grep -qx 'ForceDenseTagSearch=true' "$TENANT_DIR/indexloader.ini"
grep -qx 'LogPhaseTime=true' "$TENANT_DIR/indexloader.ini"
grep -qx 'QuantADCOnly=true' "$TENANT_DIR/indexloader.ini"
grep -qx 'MaxDistRatio=8' "$TENANT_DIR/indexloader.ini"

mkdir -p "$OUT"
INDEX_DIR="$OVERLAY" \
QUERY_DIR="$QUERY_DIR" \
NUM_QUERIES=100 \
WARMUP=20 \
TOPK=10 \
MEASURE_OFFSET=20 \
LEVELS=unfilter \
SPTAG_VALUE_TYPE=UInt8 \
    python3 Tools/benchmarks/efsearch_probe_levels.py > "$LOG" 2>&1

grep -F '[OPQ prefilter] QuantADCOnly=true: returning ADC-ranked results without full-vector rerank' "$LOG" >/dev/null
grep '^RESULT ' "$LOG" > "$RESULTS"
grep -F 'OPQPhaseTime:' "$LOG" > "$OUT/phase.log"
grep '^RESULT ' "$LOG"
