#!/usr/bin/env bash
set -euo pipefail

# Clone the current SIFT1B four-node/tail index and replace only its 32-byte
# PipePQ posting codes with 32-byte OPQ codes. The copy preserves the head
# graph, posting memberships, record order, and pure/tail placement.

ROOT=/mnt/nvme/baotonglu/mocheng
SOURCE_INDEX="$ROOT/datasets/sift1b/sift1b_spann_pipepq32_r010_tail1"
TARGET_INDEX="$ROOT/datasets/sift1b/sift1b_spann_opq32_r010_tail1"
OPQ_BUILD_DIR="$ROOT/datasets/sift1b/sift1b_opq32_build"
OPQ_CODES="$OPQ_BUILD_DIR/opq_codes_m32.bin"
OPQ_QUANTIZER="$OPQ_BUILD_DIR/opq_quantizer.bin"
SPANN_BUILDER=/home/baotonglu/mocheng/SPTAG/Release/spannbuilder

SOURCE_TENANT="$SOURCE_INDEX/tenant_0"
TARGET_TENANT="$TARGET_INDEX/tenant_0"

for path in "$SOURCE_TENANT/indexloader.ini" "$SOURCE_TENANT/inpost_pipepq.bin" \
    "$SOURCE_TENANT/ssdmapping_postings" "$OPQ_CODES" "$OPQ_QUANTIZER" "$SPANN_BUILDER"; do
    if [[ ! -e "$path" ]]; then
        echo "Required path is missing: $path" >&2
        exit 1
    fi
done
if [[ -e "$TARGET_INDEX" ]]; then
    echo "Refusing to overwrite existing target index: $TARGET_INDEX" >&2
    exit 1
fi

read -r SOURCE_M SOURCE_STRIDE < <(
    python3 - "$SOURCE_TENANT/inpost_pipepq.bin" <<'PY'
import struct
import sys

with open(sys.argv[1], "rb") as marker:
    raw = marker.read(8)
if len(raw) != 8:
    raise SystemExit("PipePQ marker is truncated")
print(*struct.unpack("=ii", raw))
PY
)
if [[ "$SOURCE_M" != 32 || "$SOURCE_STRIDE" != 57 ]]; then
    echo "Expected a PipePQ32 [meta25|code32] source, got M=$SOURCE_M stride=$SOURCE_STRIDE" >&2
    exit 1
fi

EXPECTED_CODE_BYTES=32000000000
ACTUAL_CODE_BYTES=$(stat -c '%s' "$OPQ_CODES")
if [[ "$ACTUAL_CODE_BYTES" != "$EXPECTED_CODE_BYTES" ]]; then
    echo "OPQ code sidecar has $ACTUAL_CODE_BYTES bytes; expected $EXPECTED_CODE_BYTES" >&2
    exit 1
fi

MUTABLE_FILES=(
    indexloader.ini
    DeletedIDs.bin
    ssdmapping
    ssdmapping_postings
    ssdmapping_postings_blockpool
    ssdinfo
    checksum
    posting_pure_counts.bin
)
REQUIRED_BYTES=$(du -sB1 "$SOURCE_TENANT/ssdmapping_postings" \
    "$SOURCE_TENANT/DeletedIDs.bin" \
    "$SOURCE_TENANT/ssdmapping" \
    "$SOURCE_TENANT/ssdmapping_postings_blockpool" \
    "$SOURCE_TENANT/ssdinfo" \
    "$SOURCE_TENANT/checksum" \
    "$SOURCE_TENANT/posting_pure_counts.bin" | awk '{total += $1} END {print total}')
AVAILABLE_BYTES=$(df -B1 --output=avail "$(dirname "$TARGET_INDEX")" | tail -1 | tr -d ' ')
if (( AVAILABLE_BYTES < REQUIRED_BYTES + 107374182400 )); then
    echo "Insufficient space for isolated mutable postings: need at least $((REQUIRED_BYTES + 107374182400)) bytes, have $AVAILABLE_BYTES" >&2
    exit 1
fi

echo "[clone] linking immutable SIFT index structure"
cp -al "$SOURCE_INDEX" "$TARGET_INDEX"

isolate_file() {
    local name=$1
    local target="$TARGET_TENANT/$name"
    local temporary="$TARGET_TENANT/.${name}.isolated"
    cp --reflink=auto --sparse=always --preserve=mode,timestamps "$target" "$temporary"
    mv -f "$temporary" "$target"
    if [[ "$(stat -c '%i' "$SOURCE_TENANT/$name")" == "$(stat -c '%i' "$target")" ]]; then
        echo "Mutable file remains hard-linked to the source: $name" >&2
        exit 1
    fi
}

echo "[clone] isolating mutable posting-store files"
for name in "${MUTABLE_FILES[@]}"; do
    isolate_file "$name"
done

ln "$OPQ_CODES" "$TARGET_TENANT/opq_codes_m32.bin"
ln "$OPQ_QUANTIZER" "$TARGET_TENANT/opq_quantizer.bin"

python3 - "$TARGET_TENANT/indexloader.ini" "$TARGET_TENANT" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
target_tenant = sys.argv[2]
lines = path.read_text(encoding="utf-8").splitlines()

updates = {
    "Base": {
        "IndexDirectory": target_tenant,
    },
    "BuildSSDIndex": {
        "PostingQuantizer": "OPQ",
        "PostingQuantM": "32",
        "PostingQuantizerFile": "opq_codes_m32.bin",
        "PipePQPivotsFile": "",
        "RequantizeFromPipePQ": "true",
    },
}

output = []
section = None
seen = {name: set() for name in updates}
for line in lines:
    if line.startswith("[") and line.endswith("]"):
        section = line[1:-1]
        output.append(line)
        continue
    if section in updates and "=" in line and not line.lstrip().startswith(";"):
        key = line.split("=", 1)[0].strip()
        if key in updates[section]:
            if key not in seen[section]:
                output.append(f"{key}={updates[section][key]}")
                seen[section].add(key)
            continue
    output.append(line)

for section_name, values in updates.items():
    missing = [key for key in values if key not in seen[section_name]]
    if not missing:
        continue
    insertion = next(
        (index + 1 for index, line in enumerate(output) if line == f"[{section_name}]"),
        None,
    )
    if insertion is None:
        output.extend(["", f"[{section_name}]"])
        insertion = len(output)
    for key in missing:
        output.insert(insertion, f"{key}={values[key]}")
        insertion += 1

path.write_text("\n".join(output) + "\n", encoding="utf-8")
PY

if [[ -e "$TARGET_TENANT/inpost_opq.bin" ]]; then
    echo "Target unexpectedly contains an OPQ marker before conversion" >&2
    exit 1
fi

echo "[clone] rewriting only posting codes through the native INI"
"$SPANN_BUILDER" --inpost-opq-requantize \
    --index-dir "$TARGET_INDEX" \
    --dim 128 \
    --value-type UInt8 \
    --tenant 0 2>&1 | tee "$TARGET_INDEX/pipepq_to_opq32_requantize.log"

if [[ -e "$TARGET_TENANT/inpost_pipepq.bin" || ! -e "$TARGET_TENANT/inpost_opq.bin" ]]; then
    echo "OPQ requantization markers are invalid" >&2
    exit 1
fi
read -r TARGET_M TARGET_STRIDE < <(
    python3 - "$TARGET_TENANT/inpost_opq.bin" <<'PY'
import struct
import sys

with open(sys.argv[1], "rb") as marker:
    raw = marker.read(8)
if len(raw) != 8:
    raise SystemExit("OPQ marker is truncated")
print(*struct.unpack("=ii", raw))
PY
)
if [[ "$TARGET_M" != 32 || "$TARGET_STRIDE" != 57 ]]; then
    echo "Unexpected OPQ marker M=$TARGET_M stride=$TARGET_STRIDE" >&2
    exit 1
fi

echo "[clone] complete: $TARGET_INDEX"
