#!/usr/bin/env bash
set -euo pipefail

# Native-INI-only FileIO request-batch sweep for the existing SIFT1B OPQ32
# index. Each point loads a separate overlay so SpdkBatchSize is consumed before
# the FileIO controller is initialized.

if (($# < 4)); then
    echo "Usage: $0 <TopK> <QueryDir> <RerankL> <FixedNprobe> [SpdkBatchSize ...]" >&2
    exit 2
fi

TOPK=$1
QUERY_DIR=$2
RERANK_L=$3
NPROBE=$4
shift 4
BATCH_SIZES=("$@")
if ((${#BATCH_SIZES[@]} == 0)); then
    BATCH_SIZES=(64 128 192 256)
fi

cd /home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH="$PWD/Release:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/Release"

SOURCE_INDEX=/mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_spann_opq32_r010_tail1
SOURCE_INI="$SOURCE_INDEX/tenant_0/indexloader.ini"
AUDIT_ROOT="${AUDIT_ROOT:-/mnt/nvme/baotonglu/mocheng/pipeann/audits/sift1b_opq32_fileio_batch_sweep_20260721}"
OUT_DIR="$AUDIT_ROOT/topk_${TOPK}_rerank_l_${RERANK_L}_nprobe_${NPROBE}"
CURVE="$OUT_DIR/curve.jsonl"

[[ "$TOPK" =~ ^[1-9][0-9]*$ && "$RERANK_L" =~ ^[1-9][0-9]*$ && "$NPROBE" =~ ^[1-9][0-9]*$ ]] ||
    { echo "TopK, RerankL, and FixedNprobe must be positive integers" >&2; exit 1; }
(( RERANK_L >= TOPK && NPROBE >= TOPK )) ||
    { echo "RerankL and FixedNprobe must be at least TopK" >&2; exit 1; }
[[ -f "$SOURCE_INI" && -f "$QUERY_DIR/groundtruth_unfilter_local_ids.npy" ]] ||
    { echo "Missing OPQ index or unfilter query inputs" >&2; exit 1; }

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
    local batch_size=$1
    local overlay="$OUT_DIR/overlays/spdk_batch_${batch_size}"
    local tenant="$overlay/tenant_0"
    rm -rf "$overlay"
    mkdir -p "$tenant"
    [[ ! -e "$SOURCE_INDEX/manifest.txt" ]] || ln -s "$SOURCE_INDEX/manifest.txt" "$overlay/manifest.txt"
    for entry in "$SOURCE_INDEX/tenant_0"/*; do
        local name
        name=$(basename "$entry")
        [[ "$name" == "indexloader.ini" ]] && continue
        ln -s "$entry" "$tenant/$name"
    done

    cp "$SOURCE_INI" "$tenant/indexloader.ini"
    set_ini_value "$tenant/indexloader.ini" Base IndexDirectory "$tenant"
    set_ini_value "$tenant/indexloader.ini" BuildSSDIndex SpdkBatchSize "$batch_size"
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex isExecute true
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex BuildSsdIndex false
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex ResultNum "$TOPK"
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex RerankL "$RERANK_L"
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex SearchInternalResultNum "$NPROBE"
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex FixedNprobe "$NPROBE"
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex MaxCheck 1024
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex HashTableExponent 4
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex MaxDistRatio 8.0
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex SearchPostingPageLimit 3
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex EnableAdaptiveFilteredNprobe false
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex ForceDenseTagSearch true
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex LogPhaseTime true

    python3 - "$tenant/indexloader.ini" "$tenant" "$batch_size" "$TOPK" "$RERANK_L" "$NPROBE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected_dir, batch, topk, rerank_l, nprobe = sys.argv[2:]
sections = {}
section = ""
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith((";", "#")):
        continue
    if line.startswith("[") and line.endswith("]"):
        section = line[1:-1].lower()
    elif "=" in line:
        key, value = line.split("=", 1)
        sections[(section, key.strip().lower())] = value.strip()

checks = {
    ("base", "indexdirectory"): expected_dir,
    ("buildssdindex", "spdkbatchsize"): batch,
    ("searchssdindex", "resultnum"): topk,
    ("searchssdindex", "rerankl"): rerank_l,
    ("searchssdindex", "searchinternalresultnum"): nprobe,
    ("searchssdindex", "fixednprobe"): nprobe,
    ("searchssdindex", "logphasetime"): "true",
    ("searchssdindex", "forcedensetagsearch"): "true",
}
for key, value in checks.items():
    if sections.get(key) != value:
        raise SystemExit(f"native INI mismatch for {key}: {sections.get(key)!r} != {value!r}")
PY
    printf '%s\n' "$overlay"
}

for batch_size in "${BATCH_SIZES[@]}"; do
    [[ "$batch_size" =~ ^[1-9][0-9]*$ ]] ||
        { echo "SpdkBatchSize must be a positive integer: $batch_size" >&2; exit 1; }
    overlay=$(make_overlay "$batch_size")
    log="$OUT_DIR/spdk_batch_${batch_size}.log"

    INDEX_DIR="$overlay" \
    QUERY_DIR="$QUERY_DIR" \
    NUM_QUERIES=100 \
    WARMUP=20 \
    MEASURE_OFFSET=20 \
    TOPK="$TOPK" \
    LEVELS=unfilter \
    SPTAG_VALUE_TYPE=UInt8 \
        python3 Tools/benchmarks/efsearch_probe_levels.py > "$log" 2>&1

    python3 - "$log" "$CURVE" "$TOPK" "$NPROBE" "$RERANK_L" "$batch_size" <<'PY'
import json
import sys
from pathlib import Path

log_path, curve_path = map(Path, sys.argv[1:3])
topk, nprobe, rerank_l, batch_size = map(int, sys.argv[3:])
text = log_path.read_text(encoding="utf-8")
if "[ERROR] Failed to load tenant" in text:
    raise RuntimeError(f"tenant load failed: {log_path}")
rows = [json.loads(line[7:]) for line in text.splitlines() if line.startswith("RESULT ")]
if len(rows) != 1 or rows[0].get("level") != "unfilter":
    raise RuntimeError(f"unexpected benchmark rows: {rows}")
row = rows[0]
if row.get("topk") != topk or row.get("nprobe") != nprobe or row.get("rerank_l") != rerank_l:
    raise RuntimeError(f"native settings were not applied: {row}")
row.update(
    dataset="sift1b",
    method="SPANN-OPQ32-FileIO",
    spdk_batch_size=batch_size,
    measured_queries=100,
    warmup_queries=20,
    threads=1,
    phase_log=str(log_path),
)
with curve_path.open("a", encoding="utf-8") as handle:
    handle.write("RESULT " + json.dumps(row, sort_keys=True) + "\n")
PY
done
