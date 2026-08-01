#!/bin/bash
# =============================================================================
# Thin launcher for the native-config attribute SPANN build.
#
#   ./Tools/benchmarks/run_spann_attr_build.sh [config.ini]
#
# All BUILD parameters live in the .ini (the single source of truth, read by
# spannbuilder -c). This launcher only carries what is NOT a build param:
#   * process-loader env (jemalloc, static-TLS) -- runtime, not config;
#   * the post-build cross-graph step (augmentheadgraph) -- a separate tool;
#   * copying the OPQ codebook into the tenant dir for search.
# Paths are derived FROM the ini so nothing is duplicated here.
# =============================================================================
set -e
cd "$(dirname "$0")/../.."          # -> repo root (SPTAG/)
ROOT=$(pwd)

CFG="${1:-$ROOT/Script_AE/iniFile/build_spann_attr_spacev_opq25.ini}"
[ -f "$CFG" ] || { echo "config not found: $CFG"; exit 2; }

# --- process loader (NOT build params) ---
# jemalloc reduces fragmentation/RSS at billion scale. Preload ONLY if present so
# the build doesn't spew "cannot be preloaded" on boxes without it; override the
# location via JEMALLOC_SO. (Drop libjemalloc.so.2 at this path to enable it.)
JEMALLOC_SO="${JEMALLOC_SO:-/usr/lib/x86_64-linux-gnu/libjemalloc.so.2}"
if [ -f "$JEMALLOC_SO" ]; then
  export LD_PRELOAD="$JEMALLOC_SO"
  echo "[launcher] LD_PRELOAD=$JEMALLOC_SO"
else
  echo "[launcher] jemalloc not found at $JEMALLOC_SO -- using system malloc (set JEMALLOC_SO to enable)"
fi
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

# --- derive paths from the ini (single source of truth) ---
ini() { sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*\([^#]*\).*/\1/p" "$CFG" | head -1 | sed 's/[[:space:]]*$//'; }
ini_section() {
  awk -F= -v section="$1" -v key="$2" '
    BEGIN {
      wanted_section = "[" tolower(section) "]";
      wanted_key = tolower(key);
    }
    /^\[/ {
      in_section = (tolower($0) == wanted_section);
      next;
    }
    in_section && NF >= 2 {
      name = $1;
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", name);
      if (tolower(name) == wanted_key) {
        value = substr($0, index($0, "=") + 1);
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value);
        print value;
        exit;
      }
    }
  ' "$CFG"
}
OUT=$(ini IndexDirectory)
STORAGE=$(ini Storage); [ -z "$STORAGE" ] && STORAGE=FILEIO
TMPROOT=$(ini TmpDir)
QFILE=$(ini PostingQuantizerFile)        # optional; e.g. .../opq_codes_m25.bin
SID=""
if [ -n "$QFILE" ]; then
  SID=$(dirname "$QFILE")
fi
PIPEPQ_PIVOTS=$(ini PipePQPivotsFile)

# (1) existing cross-graph knobs.
#   CrossEdges       : 1 = build/reuse head_cross_edges.bin. New STATIC bundle
#                      builds create it before Phase 4; this post-build fallback
#                      supports older/non-STATIC builders.
#   CrossExtraEdges  : -m, cross-subgraph edges kept per head (augmentheadgraph
#                      clamps <=0 back to 10). Default 10.
CROSS_EDGES=$(ini CrossEdges);            [ -z "$CROSS_EDGES" ] && CROSS_EDGES=1
CROSS_EXTRA_EDGES=$(ini CrossExtraEdges); [ -z "$CROSS_EXTRA_EDGES" ] && CROSS_EXTRA_EDGES=10
case "$CROSS_EXTRA_EDGES" in
  *[!0-9]*|'') CROSS_EXTRA_EDGES=10 ;;
esac
CROSS_EDGE_SEARCH_TOPK=$CROSS_EXTRA_EDGES
[ "$CROSS_EDGE_SEARCH_TOPK" -lt 15 ] && CROSS_EDGE_SEARCH_TOPK=15
CROSS_EDGE_BUILD_THREADS=$(ini_section BuildSSDIndex NumberOfThreads)
[ -z "$CROSS_EDGE_BUILD_THREADS" ] && CROSS_EDGE_BUILD_THREADS=1
ORDERED_PAGE_START=$(ini EnableOrderedPageStart); [ -z "$ORDERED_PAGE_START" ] && ORDERED_PAGE_START=false

# (2) SelectHead resume checkpoint knobs ([MultiTenant], single source of truth).
#   PersistSelectHead : 1 = after SelectHead, write head_select_state.bin and keep
#                       per-node head vector files, so a failed BuildHead/BuildSSDIndex
#                       can be retried WITHOUT re-running the BKT head selection.
#                       Default 0 (off; original behavior, smaller index).
#   ResumeBuild       : 1 = reuse an existing index dir + head_select_state.bin and
#                       skip the BKT (requires PersistSelectHead). Default 0.
PERSIST_SELECTHEAD=$(ini PersistSelectHead); [ -z "$PERSIST_SELECTHEAD" ] && PERSIST_SELECTHEAD=0
RESUME_BUILD=$(ini ResumeBuild);             [ -z "$RESUME_BUILD" ] && RESUME_BUILD=0

# (3) In-place build knob ([MultiTenant], single source of truth).
#   InPlaceBuild : 1 = build the SPANN index DIRECTLY into the final IndexDirectory
#                  (<OUT>/tenant_<id>) instead of staging in /tmp (or
#                  SPTAG_SPANN_WORK_DIR) and copying at the end. The SSD block pool
#                  is pre-allocated + incrementally flushed in the final dir, and
#                  SaveAll skips the copy. Avoids the transient 2x disk footprint
#                  and the copy time -- essential at billion scale. Default 0.
INPLACE_BUILD=$(ini InPlaceBuild);           [ -z "$INPLACE_BUILD" ] && INPLACE_BUILD=0

is_true() {
  case "$1" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
    *) return 1 ;;
  esac
}

if [ -n "$TMPROOT" ]; then
  mkdir -p "$TMPROOT/work"
  export TMPDIR="$TMPROOT"
  export SPTAG_SPANN_WORK_DIR="$TMPROOT/work"
  echo "[launcher] work dir = $SPTAG_SPANN_WORK_DIR"
fi

# BuildSignatures scans every posting and retains its filtering sidecars in
# memory. Run it in a fresh process after the memory-intensive index build.
BUILD_SIGNATURES=$(ini BuildSignatures); [ -z "$BUILD_SIGNATURES" ] && BUILD_SIGNATURES=false
PRIMARY_CFG="$CFG"
if is_true "$BUILD_SIGNATURES"; then
  PRIMARY_CFG=$(mktemp "${TMPDIR:-/tmp}/spann-primary-build.XXXXXX.ini")
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
fi

validate_runtime_config() {
  local runtime_ini="$OUT/tenant_0/indexloader.ini"
  [ -f "$runtime_ini" ] || { echo "[launcher] missing runtime config: $runtime_ini"; exit 1; }
  if [ "${STORAGE^^}" = "STATIC" ]; then
    [ -s "$OUT/tenant_0/SPTAGFullList.bin" ] ||
      { echo "[launcher] missing static posting snapshot"; exit 1; }
    if is_true "$ORDERED_PAGE_START"; then
      [ -s "$OUT/tenant_0/ordered_page_starts.bin" ] ||
        { echo "[launcher] missing ordered page-start directory"; exit 1; }
    else
      [ ! -e "$OUT/tenant_0/ordered_page_starts.bin" ] ||
        { echo "[launcher] unexpected ordered page-start directory"; exit 1; }
    fi
    local key expected actual
    for key in TailReplicaCount UnfilterTailBufferLength EnableUnfilterTail; do
      expected=$(ini "$key")
      [ -z "$expected" ] && continue
      actual=$(sed -n "s/^[[:space:]]*$key[[:space:]]*=[[:space:]]*//Ip" "$runtime_ini" | head -1)
      [ "$actual" = "$expected" ] ||
        { echo "[launcher] $key mismatch: expected $expected, got ${actual:-<missing>}"; exit 1; }
    done
    echo "[launcher] verified static posting snapshot"
    return
  fi
  python3 - "$CFG" "$runtime_ini" "$OUT/tenant_0/head_role.bin" <<'PY'
import math
import sys
from pathlib import Path

config_path, runtime_path, head_role_path = map(Path, sys.argv[1:])

def parse_ini(path):
    section = ""
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith((";", "#")):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif "=" in line:
            key, value = line.split("=", 1)
            values[(section, key.strip())] = value.strip()
    return values

config = parse_ini(config_path)
runtime = parse_ini(runtime_path)
required = {
    "TailReplicaCount": config.get(("BuildSSDIndex", "TailReplicaCount")),
    "UnfilterTailBufferLength": config.get(("BuildSSDIndex", "UnfilterTailBufferLength")),
    "PostingQuantizer": config.get(("BuildSSDIndex", "PostingQuantizer")),
    "PostingQuantM": config.get(("BuildSSDIndex", "PostingQuantM")),
    "RerankL": config.get(("BuildSSDIndex", "RerankL")),
}
for key, expected in required.items():
    if expected is None:
        continue
    actual = next((value for (_, candidate), value in runtime.items() if candidate == key), None)
    if actual is None:
        raise SystemExit(f"[launcher] runtime config is missing {key}")
    if actual != expected:
        raise SystemExit(f"[launcher] {key} mismatch: expected {expected}, got {actual}")

dual_pool = config.get(("MultiTenant", "DualPoolAugment"), "0").lower()
enabled = dual_pool in {"1", "true", "yes", "on"}
if enabled != head_role_path.exists():
    state = "present" if head_role_path.exists() else "absent"
    raise SystemExit(
        f"[launcher] DualPoolAugment={dual_pool} but head_role.bin is {state}"
    )

expected_ratio = config.get(("SelectHead", "Ratio"))
vector_count = config.get(("Base", "VectorCount"))
if expected_ratio is not None and vector_count is not None:
    import struct
    with (runtime_path.parent / "ssdinfo").open("rb") as f:
        total_heads = struct.unpack("<i", f.read(4))[0]
    h1_heads = total_heads
    if head_role_path.exists():
        h1_heads = 0
        with head_role_path.open("rb") as f:
            while chunk := f.read(1 << 20):
                h1_heads += chunk.count(0)
    achieved_ratio = h1_heads / int(vector_count)
    expected = float(expected_ratio)
    if not math.isclose(achieved_ratio, expected, rel_tol=0.1, abs_tol=0.005):
        raise SystemExit(
            f"[launcher] selected-head ratio mismatch: expected {expected:.6f}, "
            f"got {achieved_ratio:.6f} ({h1_heads}/{vector_count})"
        )
    print(
        f"[launcher] verified selected-head ratio={achieved_ratio:.6f} "
        f"({h1_heads}/{vector_count}), runtime tail/PQ settings, and U_extra state"
    )
else:
    print("[launcher] verified runtime tail/PQ settings and U_extra artifact state")
PY
}

if [ "$PERSIST_SELECTHEAD" = "1" ] || [ "$PERSIST_SELECTHEAD" = "true" ]; then
  export SPTAG_PERSIST_SELECTHEAD=1
fi
if [ "$RESUME_BUILD" = "1" ] || [ "$RESUME_BUILD" = "true" ]; then
  export SPTAG_RESUME_BUILD=1
fi
if [ "$INPLACE_BUILD" = "1" ] || [ "$INPLACE_BUILD" = "true" ]; then
  export SPTAG_SPANN_INPLACE_DIR="$OUT"
  echo "[launcher] IN-PLACE build: SPTAG_SPANN_INPLACE_DIR=$OUT (no final copy)"
fi

echo "[launcher] config = $CFG"
echo "[launcher] index  = $OUT"
if [ "${SPTAG_RESUME_BUILD:-0}" = "1" ]; then
  echo "[launcher] RESUME: keeping existing index dir, skipping BKT if head_select_state.bin present"
else
  rm -rf "$OUT"
fi

# --- build: ALL params from the ini ---
/usr/bin/time -v "$ROOT/Release/spannbuilder" -c "$PRIMARY_CFG" 2>&1
if is_true "$BUILD_SIGNATURES"; then
  echo "[launcher] BuildSignatures in a fresh process"
  /usr/bin/time -v "$ROOT/Release/spannbuilder" -c "$CFG" --build-signatures-only 2>&1
  if [ "${STORAGE^^}" != "STATIC" ]; then
    [ -s "$OUT/tenant_0/signatures_bitmask.bin" ] ||
      { echo "[launcher] missing signatures_bitmask.bin after BuildSignatures"; exit 1; }
  fi
  [ -s "$OUT/tenant_0/HeadIndex/head_node_meta.bin" ] ||
    { echo "[launcher] missing head_node_meta.bin after BuildSignatures"; exit 1; }
  # The full signature pass can lack the categorical-only routing projection
  # when numeric attributes are present. Rebuild only that small sidecar rather
  # than rescanning all SSD postings.
  ROUTING_INDEX="$OUT/tenant_0/HeadIndex/tag_node_index.bin"
  if [ ! -s "$ROUTING_INDEX" ]; then
    echo "[launcher] missing tag_node_index.bin; rebuilding routing sidecar"
    /usr/bin/time -v "$ROOT/Release/spannbuilder" -c "$CFG" --routing-only 2>&1
  fi
  [ -s "$ROUTING_INDEX" ] ||
    { echo "[launcher] missing tag_node_index.bin after routing rebuild"; exit 1; }
fi
validate_runtime_config

# --- (1) cross-graph: new STATIC bundle builds created this sidecar before
#     BuildSSD. Retain the post-build tool only as a fallback/rebuild path. ---
if [ "$CROSS_EDGES" = "1" ] || [ "$CROSS_EDGES" = "true" ]; then
  CROSS_EDGE_FILE="$OUT/tenant_0/HeadIndex/head_cross_edges.bin"
  CROSS_EDGE_DIRTY="$OUT/tenant_0/HeadIndex/head_cross_edges.dirty"
  if [ -s "$CROSS_EDGE_FILE" ] && [ ! -e "$CROSS_EDGE_DIRTY" ]; then
    echo "[launcher] reusing pre-BuildSSD cross-edge sidecar"
  else
    echo "[launcher] cross-graph fallback: augmentheadgraph -k $CROSS_EDGE_SEARCH_TOPK -m $CROSS_EXTRA_EDGES -t $CROSS_EDGE_BUILD_THREADS (CrossEdges=$CROSS_EDGES)"
    "$ROOT/Release/augmentheadgraph" \
      -d "$OUT/tenant_0/HeadIndex" \
      -k "$CROSS_EDGE_SEARCH_TOPK" -m "$CROSS_EXTRA_EDGES" \
      -t "$CROSS_EDGE_BUILD_THREADS" -w true
  fi
else
  echo "[launcher] cross-graph DISABLED (CrossEdges=$CROSS_EDGES) -- skipping augmentheadgraph; unfilter will use per-node fan-out"
fi

if [ -n "$SID" ] && [ -f "$SID/opq_quantizer.bin" ]; then
  cp "$SID/opq_quantizer.bin" "$OUT/tenant_0/opq_quantizer.bin"
  echo "[launcher] copied opq_quantizer.bin into tenant_0"
else
  echo "[launcher] no posting quantizer codebook to copy"
fi
if [ -n "$PIPEPQ_PIVOTS" ] && [ -f "$PIPEPQ_PIVOTS" ]; then
  cp "$PIPEPQ_PIVOTS" "$OUT/tenant_0/pipepq_pivots.bin"
  RUNTIME_INI="$OUT/tenant_0/indexloader.ini"
  [ -f "$RUNTIME_INI" ] ||
    { echo "[launcher] missing runtime config after PipePQ pivot copy"; exit 1; }
  # The build needs the external pivot source, while the deployed index must
  # resolve the immutable copied pivot sidecar from its own tenant directory.
  sed -i 's|^PipePQPivotsFile=.*$|PipePQPivotsFile=pipepq_pivots.bin|' "$RUNTIME_INI"
  grep -qx 'PipePQPivotsFile=pipepq_pivots.bin' "$RUNTIME_INI" ||
    { echo "[launcher] failed to repoint PipePQ pivots to tenant_0"; exit 1; }
  echo "[launcher] copied pipepq_pivots.bin into tenant_0"
fi
RUNTIME_TAIL=$(sed -n 's/^[[:space:]]*EnableUnfilterTail[[:space:]]*=[[:space:]]*//Ip' \
  "$OUT/tenant_0/indexloader.ini" | head -1)
echo "[launcher] done. SEARCH-time unfilter tail: EnableUnfilterTail=${RUNTIME_TAIL:-true} (persisted INI)"
