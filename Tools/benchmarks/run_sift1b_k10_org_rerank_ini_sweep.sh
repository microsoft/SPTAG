#!/usr/bin/env bash
set -euo pipefail

cd /home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH="$PWD/Release:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/Release"

SOURCE_INDEX="/mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_spann_pipepq32_r010_tail1"
SOURCE_INI="$SOURCE_INDEX/tenant_0/indexloader.ini"
QUERY_DIR="/mnt/nvme/baotonglu/mocheng/datasets/sift1b/multitenant/query"
OUT_DIR="${OUT_DIR:-/mnt/nvme/baotonglu/mocheng/pipeann/audits/sift1b_k10_org_rerank_ini_20260719}"
RERANK_LS=("$@")
if ((${#RERANK_LS[@]} == 0)); then
    RERANK_LS=(20 30 500)
fi

[[ -f "$SOURCE_INI" ]] || { echo "Missing source INI: $SOURCE_INI" >&2; exit 1; }
[[ -f "$SOURCE_INDEX/manifest.txt" ]] || { echo "Missing manifest: $SOURCE_INDEX" >&2; exit 1; }
[[ -d "$SOURCE_INDEX/tenant_0" ]] || { echo "Missing tenant index: $SOURCE_INDEX/tenant_0" >&2; exit 1; }
[[ -d "$QUERY_DIR" ]] || { echo "Missing query directory: $QUERY_DIR" >&2; exit 1; }

mkdir -p "$OUT_DIR/overlays"
OUT="$OUT_DIR/curve.jsonl"
LOG="$OUT_DIR/run.log"
: > "$OUT"
: > "$LOG"
printf '# SIFT1B org k10 native-INI rerank sweep start %s rerank_l=[%s]\n' \
    "$(date -u +%FT%TZ)" "${RERANK_LS[*]}" | tee -a "$OUT" "$LOG"

make_overlay() {
    local rerank_l="$1"
    local overlay="$OUT_DIR/overlays/rerank_l_${rerank_l}"
    local tenant_dir="$overlay/tenant_0"

    rm -rf "$overlay"
    mkdir -p "$tenant_dir"
    ln -s "$SOURCE_INDEX/manifest.txt" "$overlay/manifest.txt"
    for entry in "$SOURCE_INDEX/tenant_0"/*; do
        local name
        name="$(basename "$entry")"
        [[ "$name" == "indexloader.ini" ]] && continue
        ln -s "$entry" "$tenant_dir/$name"
    done

    cp "$SOURCE_INI" "$tenant_dir/indexloader.ini"
    sed -i \
        -e "s/^RerankL=.*/RerankL=${rerank_l}/" \
        -e 's/^ForceDenseTagSearch=.*/ForceDenseTagSearch=true/' \
        -e 's/^SearchInternalResultNum=.*/SearchInternalResultNum=64/' \
        "$tenant_dir/indexloader.ini"
    grep -qx "RerankL=${rerank_l}" "$tenant_dir/indexloader.ini"
    grep -qx 'ForceDenseTagSearch=true' "$tenant_dir/indexloader.ini"
    grep -qx 'SearchInternalResultNum=64' "$tenant_dir/indexloader.ini"
    printf '%s\n' "$overlay"
}

for rerank_l in "${RERANK_LS[@]}"; do
    [[ "$rerank_l" =~ ^[1-9][0-9]*$ ]] ||
        { echo "Invalid RerankL: $rerank_l" >&2; exit 1; }
    overlay="$(make_overlay "$rerank_l")"
    run_log="$OUT_DIR/rerank_l_${rerank_l}.log"
    printf '=== RerankL=%s %s ===\n' "$rerank_l" "$(date -u +%FT%TZ)" | tee -a "$OUT" "$LOG"

    INDEX_DIR="$overlay" \
    QUERY_DIR="$QUERY_DIR" \
    NUM_QUERIES=100 \
    WARMUP=20 \
    TOPK=10 \
    MEASURE_OFFSET=20 \
    LEVELS=org \
    SPTAG_VALUE_TYPE=UInt8 \
        python3 Tools/benchmarks/efsearch_probe_levels.py > "$run_log" 2>&1

    grep -F "[OPQ prefilter] rerank survivors L=${rerank_l} (RerankL from index config)" "$run_log" >/dev/null
    grep '^RESULT ' "$run_log" | tee -a "$OUT" "$LOG"
done

printf '# SIFT1B org k10 native-INI rerank sweep done %s\n' "$(date -u +%FT%TZ)" | tee -a "$OUT" "$LOG"
