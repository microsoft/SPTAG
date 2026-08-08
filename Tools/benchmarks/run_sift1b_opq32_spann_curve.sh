#!/usr/bin/env bash
set -euo pipefail

# Native-INI-only SIFT1B OPQ32 curve runner. Search parameters are written into
# isolated overlays; no SPTAG search setting is supplied through the environment.

if (($# < 4)); then
    echo "Usage: $0 <TopK> <QueryDir> <RerankL> <FixedNprobe ...>" >&2
    exit 2
fi

TOPK=$1
QUERY_DIR=$2
RERANK_L=$3
shift 3
NPROBES=("$@")

cd /home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH="$PWD/Release:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/Release"

SOURCE_INDEX=/mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_spann_opq32_r010_tail1
SOURCE_INI="$SOURCE_INDEX/tenant_0/indexloader.ini"
AUDIT_ROOT="${AUDIT_ROOT:-/mnt/nvme/baotonglu/mocheng/pipeann/audits/sift1b_opq32_nprobe_rerank_sweep_20260720}"
OUT_DIR="$AUDIT_ROOT/spann/topk_${TOPK}_rerank_l_${RERANK_L}"
CURVE="$OUT_DIR/curve.jsonl"

[[ "$TOPK" =~ ^[1-9][0-9]*$ && "$RERANK_L" =~ ^[1-9][0-9]*$ ]] ||
    { echo "TopK and RerankL must be positive integers" >&2; exit 1; }
(( RERANK_L >= TOPK )) ||
    { echo "RerankL must be at least TopK" >&2; exit 1; }
[[ -f "$SOURCE_INI" && -f "$QUERY_DIR/groundtruth_unfilter_local_ids.npy" ]] ||
    { echo "Missing OPQ index or native query inputs" >&2; exit 1; }

mkdir -p "$OUT_DIR/overlays"
: > "$CURVE"

set_ini_value() {
    local ini=$1
    local section=$2
    local key=$3
    local value=$4
    python3 - "$ini" "$section" "$key" "$value" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
section, key, value = sys.argv[2:]
lines = path.read_text(encoding="utf-8").splitlines()
header = f"[{section}]"

start = next(
    (i for i, line in enumerate(lines) if line.strip().lower() == header.lower()),
    None,
)
if start is None:
    if lines and lines[-1]:
        lines.append("")
    lines.extend((header, f"{key}={value}"))
else:
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("[")),
        len(lines),
    )
    for i in range(start + 1, end):
        if "=" in lines[i] and lines[i].split("=", 1)[0].strip().lower() == key.lower():
            lines[i] = f"{key}={value}"
            break
    else:
        lines.insert(end, f"{key}={value}")

path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

make_overlay() {
    local nprobe=$1
    local overlay="$OUT_DIR/overlays/nprobe_${nprobe}"
    local tenant="$overlay/tenant_0"
    rm -rf "$overlay"
    mkdir -p "$tenant"
    if [[ -e "$SOURCE_INDEX/manifest.txt" ]]; then
        ln -s "$SOURCE_INDEX/manifest.txt" "$overlay/manifest.txt"
    fi
    for entry in "$SOURCE_INDEX/tenant_0"/*; do
        local name
        name=$(basename "$entry")
        [[ "$name" == "indexloader.ini" ]] && continue
        ln -s "$entry" "$tenant/$name"
    done
    cp "$SOURCE_INI" "$tenant/indexloader.ini"
    set_ini_value "$tenant/indexloader.ini" Base IndexDirectory "$tenant"
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex isExecute true
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex BuildSsdIndex false
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex ResultNum "$TOPK"
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex RerankL "$RERANK_L"
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex SearchInternalResultNum "$nprobe"
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex FixedNprobe "$nprobe"
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex MaxCheck 1024
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex HashTableExponent 4
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex MaxDistRatio 8.0
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex SearchPostingPageLimit 3
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex EnableAdaptiveFilteredNprobe false
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex LogPhaseTime false
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex ForceDenseTagSearch true
    for expected in \
        "IndexDirectory=${tenant}" \
        'PostingQuantizer=OPQ' \
        'PostingQuantM=32' \
        '[SearchSSDIndex]' \
        "ResultNum=${TOPK}" \
        "RerankL=${RERANK_L}" \
        "SearchInternalResultNum=${nprobe}" \
        "FixedNprobe=${nprobe}" \
        'ForceDenseTagSearch=true'; do
        grep -Fqx "$expected" "$tenant/indexloader.ini"
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
    raise RuntimeError(f"Native INI values were not applied: {row}")
row.update(
    dataset="sift1b",
    method="SPANN-OPQ32",
    measured_queries=100,
    warmup_queries=20,
    threads=1,
)
with curve.open("a", encoding="utf-8") as handle:
    handle.write("RESULT " + json.dumps(row, sort_keys=True) + "\n")
PY
done
