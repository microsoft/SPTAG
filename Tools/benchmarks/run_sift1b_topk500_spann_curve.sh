#!/usr/bin/env bash
set -euo pipefail

if (($# < 3)); then
    echo "Usage: $0 <TopK> <RerankL> <FixedNprobe ...>" >&2
    exit 2
fi

TOPK=$1
RERANK_L=$2
shift 2
NPROBES=("$@")

cd /home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH="$PWD/Release:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/Release"

SOURCE_INDEX=/mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_spann_pipepq32_r010_tail1
SOURCE_INI="$SOURCE_INDEX/tenant_0/indexloader.ini"
AUDIT_ROOT=/mnt/nvme/baotonglu/mocheng/pipeann/audits/sift1b_topk500_scale_check_20260720
QUERY_DIR="$AUDIT_ROOT/inputs/spann_query"
OUT_DIR="$AUDIT_ROOT/spann/topk_${TOPK}_rerank_l_${RERANK_L}"
CURVE="$OUT_DIR/curve.jsonl"

[[ "$TOPK" =~ ^[1-9][0-9]*$ && "$RERANK_L" =~ ^[1-9][0-9]*$ ]] ||
    { echo "TopK and RerankL must be positive integers" >&2; exit 1; }
(( RERANK_L >= TOPK )) ||
    { echo "RerankL must be at least TopK" >&2; exit 1; }
[[ -f "$SOURCE_INI" && -f "$QUERY_DIR/groundtruth_unfilter_local_ids.npy" ]] ||
    { echo "Missing SIFT1B index or TopK input preparation" >&2; exit 1; }

mkdir -p "$OUT_DIR/overlays"
: > "$CURVE"

set_ini_value() {
    local ini=$1
    local key=$2
    local value=$3
    if grep -q "^${key}=" "$ini"; then
        sed -i "s|^${key}=.*|${key}=${value}|" "$ini"
    else
        sed -i "/^\[BuildSSDIndex\]$/a ${key}=${value}" "$ini"
    fi
}

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
    set_ini_value "$tenant/indexloader.ini" IndexDirectory "$tenant"
    set_ini_value "$tenant/indexloader.ini" ResultNum "$TOPK"
    set_ini_value "$tenant/indexloader.ini" RerankL "$RERANK_L"
    set_ini_value "$tenant/indexloader.ini" SearchInternalResultNum "$nprobe"
    set_ini_value "$tenant/indexloader.ini" FixedNprobe "$nprobe"
    set_ini_value "$tenant/indexloader.ini" LogPhaseTime false
    set_ini_value "$tenant/indexloader.ini" ForceDenseTagSearch true
    for expected in \
        "IndexDirectory=${tenant}" \
        "ResultNum=${TOPK}" \
        "RerankL=${RERANK_L}" \
        "SearchInternalResultNum=${nprobe}" \
        "FixedNprobe=${nprobe}" \
        'ForceDenseTagSearch=true'; do
        grep -qx "$expected" "$tenant/indexloader.ini"
    done
    printf '%s\n' "$overlay"
}

for nprobe in "${NPROBES[@]}"; do
    [[ "$nprobe" =~ ^[1-9][0-9]*$ ]] && (( nprobe >= TOPK )) ||
        { echo "FixedNprobe must be an integer at least TopK: $nprobe" >&2; exit 1; }
    overlay=$(make_overlay "$nprobe")
    log="$OUT_DIR/nprobe_${nprobe}.log"
    INDEX_DIR="$overlay" \
    QUERY_DIR="$QUERY_DIR" \
    NUM_QUERIES=100 \
    WARMUP=20 \
    MEASURE_OFFSET=20 \
    LEVELS=unfilter \
    SPTAG_VALUE_TYPE=UInt8 \
        python3 Tools/benchmarks/efsearch_probe_levels.py > "$log" 2>&1
    python3 - "$log" "$CURVE" "$TOPK" "$nprobe" "$RERANK_L" <<'PY'
import json
import sys
from pathlib import Path

log, curve = map(Path, sys.argv[1:3])
topk, nprobe, rerank_l = map(int, sys.argv[3:])
text = log.read_text(encoding="utf-8")
if "[ERROR] Failed to load tenant" in text:
    raise RuntimeError(f"Tenant load failed; rejecting benchmark log: {log}")
rows = [json.loads(line[7:]) for line in text.splitlines() if line.startswith("RESULT ")]
if len(rows) != 1 or rows[0]["level"] != "unfilter":
    raise RuntimeError(f"Unexpected SPANN rows: {rows}")
row = rows[0]
if row["topk"] != topk or row["nprobe"] != nprobe or row["rerank_l"] != rerank_l:
    raise RuntimeError(f"INI values were not applied: {row}")
row.update(
    dataset="sift1b",
    method="SPANN",
    measured_queries=100,
    warmup_queries=20,
    threads=1,
)
with curve.open("a", encoding="utf-8") as handle:
    handle.write("RESULT " + json.dumps(row, sort_keys=True) + "\n")
PY
done
