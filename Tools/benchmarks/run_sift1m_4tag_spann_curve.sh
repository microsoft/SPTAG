#!/usr/bin/env bash
set -euo pipefail

if (($# < 2)); then
    echo "Usage: $0 <RerankL> <FixedNprobe ...>" >&2
    exit 2
fi

RERANK_L=$1
shift
NPROBES=("$@")

cd /home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH="$PWD/Release:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/Release"

SOURCE_INDEX=/mnt/nvme/baotonglu/mocheng/datasets/sift1m_4tag_demo/sift1m_spann_pipepq32_r010_tail1
SOURCE_INI="$SOURCE_INDEX/tenant_0/indexloader.ini"
QUERY_DIR=/mnt/nvme/baotonglu/mocheng/datasets/sift1m_4tag_demo/multitenant/query
OUT_ROOT=/mnt/nvme/baotonglu/mocheng/pipeann/audits/sift1m_4tag_scale_check_20260720
OUT_DIR="$OUT_ROOT/spann/rerank_l_${RERANK_L}"
CURVE="$OUT_DIR/curve.jsonl"

[[ "$RERANK_L" =~ ^[1-9][0-9]*$ ]] ||
    { echo "RerankL must be positive" >&2; exit 1; }
[[ -f "$SOURCE_INI" && -d "$QUERY_DIR" ]] ||
    { echo "Missing built SIFT1M SPANN index or query inputs" >&2; exit 1; }

mkdir -p "$OUT_DIR/overlays"
: > "$CURVE"

make_overlay() {
    local nprobe=$1
    local overlay="$OUT_DIR/overlays/nprobe_${nprobe}"
    local tenant="$overlay/tenant_0"
    rm -rf "$overlay"
    mkdir -p "$tenant"
    ln -s "$SOURCE_INDEX/manifest.txt" "$overlay/manifest.txt"
    for entry in "$SOURCE_INDEX/tenant_0"/*; do
        local name
        name=$(basename "$entry")
        [[ "$name" == "indexloader.ini" ]] && continue
        ln -s "$entry" "$tenant/$name"
    done
    cp "$SOURCE_INI" "$tenant/indexloader.ini"
    sed -i \
        -e "s/^RerankL=.*/RerankL=${RERANK_L}/" \
        -e 's/^ResultNum=.*/ResultNum=10/' \
        -e "s/^SearchInternalResultNum=.*/SearchInternalResultNum=${nprobe}/" \
        -e "s|^IndexDirectory=.*|IndexDirectory=${tenant}|" \
        -e "s/^FixedNprobe=.*/FixedNprobe=${nprobe}/" \
        -e 's/^LogPhaseTime=.*/LogPhaseTime=false/' \
        -e 's/^ForceDenseTagSearch=.*/ForceDenseTagSearch=true/' \
        "$tenant/indexloader.ini"
    grep -qx "RerankL=${RERANK_L}" "$tenant/indexloader.ini"
    grep -qx 'ResultNum=10' "$tenant/indexloader.ini"
    grep -qx "SearchInternalResultNum=${nprobe}" "$tenant/indexloader.ini"
    grep -qx "FixedNprobe=${nprobe}" "$tenant/indexloader.ini"
    grep -qx "IndexDirectory=${tenant}" "$tenant/indexloader.ini"
    grep -qx 'ForceDenseTagSearch=true' "$tenant/indexloader.ini"
    printf '%s\n' "$overlay"
}

for nprobe in "${NPROBES[@]}"; do
    [[ "$nprobe" =~ ^[1-9][0-9]*$ ]] ||
        { echo "FixedNprobe must be positive: $nprobe" >&2; exit 1; }
    overlay=$(make_overlay "$nprobe")
    log="$OUT_DIR/nprobe_${nprobe}.log"
    INDEX_DIR="$overlay" \
    QUERY_DIR="$QUERY_DIR" \
    NUM_QUERIES=100 \
    WARMUP=20 \
    MEASURE_OFFSET=20 \
    LEVELS=unfilter,org \
    SPTAG_VALUE_TYPE=UInt8 \
        python3 Tools/benchmarks/efsearch_probe_levels.py > "$log" 2>&1
    python3 - "$log" "$CURVE" "$nprobe" "$RERANK_L" <<'PY'
import json
import sys
from pathlib import Path

log, curve, nprobe, rerank_l = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
log_text = log.read_text(encoding="utf-8")
if "[ERROR] Failed to load tenant" in log_text:
    raise RuntimeError(f"Tenant load failed; rejecting benchmark log: {log}")
rows = [json.loads(line[7:]) for line in log_text.splitlines()
        if line.startswith("RESULT ")]
if {row["level"] for row in rows} != {"unfilter", "org"} or len(rows) != 2:
    raise RuntimeError(f"Unexpected SPANN rows: {rows}")
if any(row["nprobe"] != nprobe or row["rerank_l"] != rerank_l for row in rows):
    raise RuntimeError(f"INI values were not applied: {rows}")
with curve.open("a", encoding="utf-8") as handle:
    for row in rows:
        row.update(
            dataset="sift1m_4tag_demo",
            method="SPANN",
            measured_queries=100,
            warmup_queries=20,
            threads=1,
        )
        handle.write("RESULT " + json.dumps(row, sort_keys=True) + "\n")
PY
done
