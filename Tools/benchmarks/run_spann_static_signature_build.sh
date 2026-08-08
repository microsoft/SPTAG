#!/bin/bash
# Build an isolated SIFT1B raw-static SPANN control from its native INI.
# Current STM1-aware binaries embed tags during this fresh build; the separate
# repack launcher is only for the pre-STM1 legacy snapshot already on disk.
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT=$(pwd)
CFG="${1:-$ROOT/Tools/benchmarks/build_spann_static_signature_sift1b.ini}"
[ -f "$CFG" ] || { echo "config not found: $CFG" >&2; exit 2; }

ini() {
  sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*\([^;#]*\).*/\1/p" "$CFG" |
    head -1 | sed 's/[[:space:]]*$//'
}

is_true() {
  case "$1" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
    *) return 1 ;;
  esac
}

OUT=$(ini IndexDirectory)
TMPROOT=$(ini TmpDir)
STORAGE=$(ini Storage)
BUILD_SIGNATURES=$(ini BuildSignatures)
INPLACE_BUILD=$(ini InPlaceBuild)
PERSIST_SELECTHEAD=$(ini PersistSelectHead)
ORDERED_PAGE_START=$(ini EnableOrderedPageStart)

[ "$STORAGE" = "STATIC" ] ||
  { echo "refusing non-STATIC config: Storage=$STORAGE" >&2; exit 2; }
[ -n "$OUT" ] && [ -n "$TMPROOT" ] ||
  { echo "IndexDirectory and TmpDir are required" >&2; exit 2; }
case "$OUT" in
  /mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_spann_static_signature_*|\
  /mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_spann_ordered_page_*) ;;
  *) echo "refusing unexpected output path: $OUT" >&2; exit 2 ;;
esac
[ ! -e "$OUT" ] ||
  { echo "refusing to overwrite existing output: $OUT" >&2; exit 2; }

mkdir -p "$TMPROOT"
export TMPDIR="$TMPROOT"
export SPTAG_SPANN_WORK_DIR="$TMPROOT/work"
mkdir -p "$SPTAG_SPANN_WORK_DIR"

if is_true "$INPLACE_BUILD"; then
  export SPTAG_SPANN_INPLACE_DIR="$OUT"
fi
if is_true "$PERSIST_SELECTHEAD"; then
  export SPTAG_PERSIST_SELECTHEAD=1
fi

JEMALLOC_SO="${JEMALLOC_SO:-/usr/lib/x86_64-linux-gnu/libjemalloc.so.2}"
if [ -f "$JEMALLOC_SO" ]; then
  export LD_PRELOAD="$JEMALLOC_SO"
fi
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

PRIMARY_CFG=$(mktemp "$TMPROOT/spann-static-primary.XXXXXX.ini")
trap 'rm -f "$PRIMARY_CFG"' EXIT HUP INT TERM
awk '
  /^\[Build\][[:space:]]*$/ { in_build = 1 }
  /^\[/ && $0 !~ /^\[Build\][[:space:]]*$/ { in_build = 0 }
  in_build && /^[[:space:]]*BuildSignatures[[:space:]]*=/ {
    print "BuildSignatures=false"
    replaced = 1
    next
  }
  { print }
  END { exit replaced ? 0 : 1 }
' "$CFG" > "$PRIMARY_CFG"

echo "[static-signature] config=$CFG"
echo "[static-signature] output=$OUT"
echo "[static-signature] tmp=$TMPROOT"
echo "[static-signature] in-place=${SPTAG_SPANN_INPLACE_DIR:-off}"

/usr/bin/time -v "$ROOT/Release/spannbuilder" -c "$PRIMARY_CFG"

if is_true "$BUILD_SIGNATURES"; then
  /usr/bin/time -v "$ROOT/Release/spannbuilder" -c "$CFG" --build-signatures-only
  [ -s "$OUT/tenant_0/HeadIndex/head_node_meta.bin" ] ||
    { echo "missing static posting-mask artifact after BuildSignatures" >&2; exit 1; }
fi

[ -s "$OUT/tenant_0/indexloader.ini" ] ||
  { echo "missing runtime indexloader.ini" >&2; exit 1; }
if is_true "$ORDERED_PAGE_START"; then
  [ -s "$OUT/tenant_0/ordered_page_starts.bin" ] ||
    { echo "missing ordered page-start directory after ordered STATIC build" >&2; exit 1; }
else
  [ ! -e "$OUT/tenant_0/ordered_page_starts.bin" ] ||
    { echo "unexpected ordered page-start directory for no-order STATIC build" >&2; exit 1; }
fi

python3 - "$OUT/tenant_0/indexloader.ini" <<'PY'
import sys
from pathlib import Path

values = {}
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith((";", "#")) or "=" not in line:
        continue
    key, value = line.split("=", 1)
    values[key.strip().lower()] = value.strip()

if values.get("storage", "").upper() != "STATIC":
    raise SystemExit(f"runtime Storage is not STATIC: {values.get('storage')!r}")
for forbidden in ("postingquantizer", "postingquantm", "postingquantizerfile",
                  "pipepqpivotsfile", "fullvectorfile", "rerankl",
                  "tailreplicacount", "unfiltertailbufferlength"):
    if forbidden in values and values[forbidden].strip().lower() not in {"", "0", "none", "false"}:
        raise SystemExit(f"runtime config unexpectedly enables {forbidden}={values[forbidden]!r}")
print("[static-signature] runtime confirms STATIC raw postings without PQ/rerank/tail")
PY
