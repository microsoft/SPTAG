#!/bin/bash
# Migrate the pre-STM1 legacy raw STATIC snapshot to STM1, then build its
# posting-level ACL masks. Fresh STM1-aware builds embed tags directly. The tag
# sidecar is read only during these offline migration steps.
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT=$(pwd)
CFG="${1:-$ROOT/Tools/benchmarks/build_spann_static_signature_sift1b.ini}"
[ -f "$CFG" ] || { echo "config not found: $CFG" >&2; exit 2; }

ini() {
  awk -v wanted_section="$1" -v wanted_key="$2" '
    /^\[/ {
      section = $0
      sub(/^\[/, "", section)
      sub(/\][[:space:]]*$/, "", section)
      next
    }
    section == wanted_section {
      split($0, fields, "=")
      if (fields[1] == wanted_key) {
        value = substr($0, index($0, "=") + 1)
        sub(/[[:space:]]*;.*/, "", value)
        sub(/[[:space:]]+$/, "", value)
        print value
        exit
      }
    }
  ' "$CFG"
}

is_true() {
  case "$1" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
    *) return 1 ;;
  esac
}

OUT=$(ini Base IndexDirectory)
TAG_FILE=$(ini Tags TagFile)
VECTOR_COUNT=$(ini Base VectorCount)
DIM=$(ini Base Dim)
VECTOR_TYPE=$(ini Base VectorType)
TAG_COUNT=$(ini Tags NumTagsPerVec)
TENANT=$(ini Tags Tenant)
STORAGE=$(ini BuildSSDIndex Storage)
BUILD_SIGNATURES=$(ini Build BuildSignatures)

[ "$STORAGE" = "STATIC" ] ||
  { echo "refusing non-STATIC config: Storage=$STORAGE" >&2; exit 2; }
is_true "$BUILD_SIGNATURES" ||
  { echo "BuildSignatures=true is required after STM1 conversion" >&2; exit 2; }
[ -n "$OUT" ] && [ -n "$TAG_FILE" ] && [ -n "$VECTOR_COUNT" ] && [ -n "$DIM" ] &&
  [ -n "$VECTOR_TYPE" ] && [ -n "$TAG_COUNT" ] && [ -n "$TENANT" ] ||
  { echo "missing required Base/Tags values in $CFG" >&2; exit 2; }
[ "$VECTOR_TYPE" = "UInt8" ] ||
  { echo "this raw SIFT1B STM1 launcher requires VectorType=UInt8" >&2; exit 2; }
case "$OUT" in
  /mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_spann_static_signature_*) ;;
  *) echo "refusing unexpected output path: $OUT" >&2; exit 2 ;;
esac

POSTING="$OUT/tenant_$TENANT/SPTAGFullList.bin"
LEGACY="$POSTING.legacy"
TEMPORARY="$POSTING.stm1.tmp"
[ -f "$POSTING" ] || { echo "missing legacy STATIC posting file: $POSTING" >&2; exit 1; }
[ -f "$TAG_FILE" ] || { echo "missing offline tag source: $TAG_FILE" >&2; exit 1; }
[ ! -e "$LEGACY" ] && [ ! -e "$TEMPORARY" ] ||
  { echo "refusing to overwrite existing STM1 backup or temporary file" >&2; exit 1; }

python3 - "$POSTING" "$VECTOR_COUNT" "$DIM" "$TAG_COUNT" <<'PY'
import struct
import sys
from pathlib import Path

posting = Path(sys.argv[1])
expected_count, expected_dim, expected_tags = map(int, sys.argv[2:])
with posting.open("rb") as source:
    header = source.read(16)
if len(header) != 16:
    raise SystemExit("legacy STATIC posting header is truncated")
list_count, vector_count, dim, list_page_offset = struct.unpack("<4i", header)
if list_count <= 0 or vector_count != expected_count or dim != expected_dim or list_page_offset < 0:
    raise SystemExit(
        f"legacy STATIC header mismatch: lists={list_count} vectors={vector_count} "
        f"dim={dim} pages={list_page_offset}"
    )
if struct.unpack("<I", header[:4])[0] == 0x314D5453:
    raise SystemExit("posting file is already STM1")
if expected_tags <= 0:
    raise SystemExit("NumTagsPerVec must be positive for STM1")
print(
    f"[static-stm1] legacy header: lists={list_count} vectors={vector_count} "
    f"dim={dim} tags={expected_tags}"
)
PY

echo "[static-stm1] config=$CFG"
echo "[static-stm1] posting=$POSTING"
df -h "$OUT"
/usr/bin/time -v "$ROOT/Release/spannbuilder" -c "$CFG" --repack-static-stm1

python3 - "$POSTING" "$VECTOR_COUNT" "$DIM" "$TAG_COUNT" <<'PY'
import struct
import sys
from pathlib import Path

posting = Path(sys.argv[1])
expected_count, expected_dim, expected_tags = map(int, sys.argv[2:])
with posting.open("rb") as source:
    header = source.read(36)
if len(header) != 36:
    raise SystemExit("STM1 header is truncated")
magic, version, lists, vector_count, dim, record_bytes, tag_count, tail_pages, list_page_offset = (
    struct.unpack("<9i", header)
)
expected_record_bytes = 4 + expected_tags * 4 + expected_dim
if (magic, version, vector_count, dim, record_bytes, tag_count, tail_pages) != (
    0x314D5453, 1, expected_count, expected_dim, expected_record_bytes, expected_tags, 0
):
    raise SystemExit(
        f"STM1 header mismatch: magic={magic:#x} version={version} vectors={vector_count} "
        f"dim={dim} record={record_bytes} tags={tag_count} tail={tail_pages}"
    )
if lists <= 0 or list_page_offset < 0:
    raise SystemExit("STM1 list metadata is invalid")
print(
    f"[static-stm1] STM1 header verified: lists={lists} record_bytes={record_bytes} "
    f"list_page_offset={list_page_offset}"
)
PY

HEAD_META="$OUT/tenant_$TENANT/HeadIndex/head_node_meta.bin"
HEAD_META_MARKER=$(mktemp "$OUT/tenant_$TENANT/.head-node-meta-before.XXXXXX")
trap 'rm -f "$HEAD_META_MARKER"' EXIT HUP INT TERM
/usr/bin/time -v "$ROOT/Release/spannbuilder" -c "$CFG" --build-signatures-only
[ -s "$HEAD_META" ] ||
  { echo "missing STM1 posting-mask artifact after BuildSignatures: $HEAD_META" >&2; exit 1; }
[ "$HEAD_META" -nt "$HEAD_META_MARKER" ] ||
  { echo "head_node_meta.bin was not refreshed by STM1 BuildSignatures" >&2; exit 1; }
echo "[static-stm1] complete: STM1 postings and static posting masks are ready"
