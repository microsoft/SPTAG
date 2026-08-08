#!/usr/bin/env bash
set -euo pipefail

# SIFT1B raw-STATIC STM1 Recall@10/QPS curves. Runtime controls belong only
# in each isolated native INI overlay; no SPTAG_* search environment overrides.

if (($# < 1)); then
    echo "Usage: $0 <InternalResultNum ...>" >&2
    exit 2
fi

NPROBES=("$@")

cd /home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH="$PWD/Release:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/Release"

SOURCE_INDEX=/mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_spann_static_signature_r08
SOURCE_INI="$SOURCE_INDEX/tenant_0/indexloader.ini"
QUERY_DIR=/mnt/nvme/baotonglu/mocheng/datasets/sift1b/multitenant/query
AUDIT_ROOT="${AUDIT_ROOT:-/mnt/nvme/baotonglu/mocheng/pipeann/audits/sift1b_static_stm1_curve_20260723}"
OUT_DIR="$AUDIT_ROOT/spann_k10"
CURVE="$OUT_DIR/curve.jsonl"
RUN_LOG="$OUT_DIR/run.log"

[[ -f "$SOURCE_INI" && -f "$SOURCE_INDEX/manifest.txt" ]] ||
    { echo "Missing STM1 static index: $SOURCE_INDEX" >&2; exit 1; }
[[ -s "$SOURCE_INDEX/tenant_0/SPTAGFullList.bin" ]] ||
    { echo "Missing STM1 postings" >&2; exit 1; }
[[ -s "$SOURCE_INDEX/tenant_0/HeadIndex/head_node_meta.bin" ]] ||
    { echo "Missing STM1 posting-mask metadata" >&2; exit 1; }
[[ -f "$QUERY_DIR/query_vectors.npy" && -f "$QUERY_DIR/query_tags.npy" ]] ||
    { echo "Missing SIFT1B query inputs: $QUERY_DIR" >&2; exit 1; }

mkdir -p "$OUT_DIR/overlays"
: > "$CURVE"
: > "$RUN_LOG"
printf '# SIFT1B raw STATIC STM1 k10 curve start %s internal_result_num=[%s]\n' \
    "$(date -u +%FT%TZ)" "${NPROBES[*]}" | tee -a "$RUN_LOG"

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
start = next((i for i, line in enumerate(lines) if line.strip().lower() == header.lower()), None)
if start is None:
    if lines and lines[-1]:
        lines.append("")
    lines.extend((header, f"{key}={value}"))
else:
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("[")), len(lines))
    for i in range(start + 1, end):
        if "=" in lines[i] and lines[i].split("=", 1)[0].strip().lower() == key.lower():
            lines[i] = f"{key}={value}"
            break
    else:
        lines.insert(end, f"{key}={value}")
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

remove_ini_key() {
    local ini=$1
    local key=$2
    python3 - "$ini" "$key" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
key = sys.argv[2].lower()
lines = path.read_text(encoding="utf-8").splitlines()
filtered = [
    line for line in lines
    if not ("=" in line and line.split("=", 1)[0].strip().lower() == key)
]
path.write_text("\n".join(filtered) + "\n", encoding="utf-8")
PY
}

make_overlay() {
    local nprobe=$1
    local maxcheck=$2
    local overlay="$OUT_DIR/overlays/internal_${nprobe}"
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

    # These controls were removed in the post-pull parameter cleanup. Do not
    # retain stale copies inherited from the persisted source INI.
    remove_ini_key "$tenant/indexloader.ini" FixedNprobe
    remove_ini_key "$tenant/indexloader.ini" PinnedPostingTarget
    remove_ini_key "$tenant/indexloader.ini" TagAwareHeadExpansion
    remove_ini_key "$tenant/indexloader.ini" SearchPostingPageLimit

    set_ini_value "$tenant/indexloader.ini" Base IndexDirectory "$tenant"
    # STATIC allocates its async reader queues from the BuildSSDIndex handler
    # count when loading. Keep it at one in the isolated runtime overlay: this
    # is required for the one-thread protocol and avoids a 45*nprobe uring
    # allocation at high budgets without changing the source index.
    set_ini_value "$tenant/indexloader.ini" BuildSSDIndex NumberOfThreads 1
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex isExecute true
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex BuildSsdIndex false
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex ResultNum 10
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex InternalResultNum "$nprobe"
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex MaxCheck "$maxcheck"
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex NumberOfThreads 1
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex HashTableExponent 4
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex MaxDistRatio 8.0
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex PostingPageLimit 3
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex IOThreadsPerHandler 1
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex ForceDenseTagSearch true
    set_ini_value "$tenant/indexloader.ini" SearchSSDIndex EnableHierPostingFilter true

    grep -Fqx "IndexDirectory=$tenant" "$tenant/indexloader.ini"
    grep -Fqx 'NumberOfThreads=1' "$tenant/indexloader.ini"
    grep -Fqx 'ResultNum=10' "$tenant/indexloader.ini"
    grep -Fqx "InternalResultNum=$nprobe" "$tenant/indexloader.ini"
    grep -Fqx "MaxCheck=$maxcheck" "$tenant/indexloader.ini"
    grep -Fqx 'IOThreadsPerHandler=1' "$tenant/indexloader.ini"
    grep -Fqx 'ForceDenseTagSearch=true' "$tenant/indexloader.ini"
    grep -Fqx 'EnableHierPostingFilter=true' "$tenant/indexloader.ini"
    ! grep -Eiq '^[[:space:]]*(FixedNprobe|PinnedPostingTarget|TagAwareHeadExpansion)[[:space:]]*=' \
        "$tenant/indexloader.ini"
    printf '%s\n' "$overlay"
}

for nprobe in "${NPROBES[@]}"; do
    [[ "$nprobe" =~ ^[1-9][0-9]*$ ]] ||
        { echo "InternalResultNum must be a positive integer: $nprobe" >&2; exit 1; }

    # MaxCheck is the graph traversal budget. Keep its legacy 1024 floor while
    # allowing every requested posting candidate to be reached at larger points.
    maxcheck=$(( nprobe > 1024 ? nprobe : 1024 ))
    overlay=$(make_overlay "$nprobe" "$maxcheck")
    point_log="$OUT_DIR/internal_${nprobe}.log"

    printf '=== InternalResultNum=%s MaxCheck=%s %s ===\n' \
        "$nprobe" "$maxcheck" "$(date -u +%FT%TZ)" | tee -a "$RUN_LOG"
    INDEX_DIR="$overlay" \
    QUERY_DIR="$QUERY_DIR" \
    NUM_QUERIES=100 \
    WARMUP=20 \
    MEASURE_OFFSET=20 \
    LEVELS=unfilter,org,dept,team,project \
    SPTAG_VALUE_TYPE=UInt8 \
    TEST_MAXCHECK="$maxcheck" \
        python3 Tools/benchmarks/efsearch_probe_levels.py > "$point_log" 2>&1

    grep -F "Setting SearchInternalResultNum with value $nprobe" "$point_log" >/dev/null
    grep -F "Setting MaxCheck with value $maxcheck" "$point_log" >/dev/null
    grep -F 'Setting EnableHierPostingFilter with value true' "$point_log" >/dev/null
    grep -E 'AsyncFileIO::InitializeFileIo: file .* threads=1 maxNumBlocks=' "$point_log" >/dev/null
    ! grep -F 'Cannot setup aio:' "$point_log"
    python3 - "$point_log" "$CURVE" "$nprobe" "$maxcheck" <<'PY'
import json
import sys
from pathlib import Path

log_path, curve_path = map(Path, sys.argv[1:3])
nprobe, maxcheck = map(int, sys.argv[3:])
rows = [
    json.loads(line[7:])
    for line in log_path.read_text(encoding="utf-8").splitlines()
    if line.startswith("RESULT ")
]
expected_levels = {"unfilter", "org", "dept", "team", "project"}
if {row["level"] for row in rows} != expected_levels or len(rows) != len(expected_levels):
    raise RuntimeError(f"Unexpected curve rows: {rows}")
if any(row["topk"] != 10 for row in rows):
    raise RuntimeError(f"Unexpected top-k: {rows}")
if any(row["nprobe"] != nprobe or row["configured_search_internal_result_num"] != nprobe
       for row in rows):
    raise RuntimeError(f"Native InternalResultNum was not applied: {rows}")
if any(row["maxcheck"] != maxcheck for row in rows):
    raise RuntimeError(f"Native MaxCheck was not applied: {rows}")
if any(not row["force_dense_tag_search"] for row in rows):
    raise RuntimeError(f"ForceDenseTagSearch was not applied: {rows}")
with curve_path.open("a", encoding="utf-8") as handle:
    for row in rows:
        row.update(
            dataset="sift1b",
            method="SPANN-STATIC-STM1",
            internal_result_num=nprobe,
            maxcheck=maxcheck,
            posting_prefilter=True,
            measured_queries=100,
            warmup_queries=20,
            threads=1,
            ssd_io_threads=1,
            io_threads_per_handler=1,
        )
        handle.write("RESULT " + json.dumps(row, sort_keys=True) + "\n")
PY
    grep '^RESULT ' "$point_log" | tee -a "$RUN_LOG"
done

printf '# SIFT1B raw STATIC STM1 k10 curve done %s\n' "$(date -u +%FT%TZ)" | tee -a "$RUN_LOG"
