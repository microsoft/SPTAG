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
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

# --- derive paths from the ini (single source of truth) ---
ini() { sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*\([^#]*\).*/\1/p" "$CFG" | head -1 | sed 's/[[:space:]]*$//'; }
OUT=$(ini IndexDirectory)
QFILE=$(ini PostingQuantizerFile)        # .../opq_codes_m25.bin
SID=$(dirname "$QFILE")

# (1) cross-graph knobs -- read from the ini ([MultiTenant], single source of truth).
#   CrossEdges       : 1 = run augmentheadgraph (stitch per-node head bundles),
#                      0 = skip it entirely (no head_cross_edges.bin -> unfilter
#                      falls back to per-node fan-out). Default 1.
#   CrossExtraEdges  : -m, cross-subgraph edges kept per head (augmentheadgraph
#                      clamps <=0 back to 10). Default 10.
CROSS_EDGES=$(ini CrossEdges);            [ -z "$CROSS_EDGES" ] && CROSS_EDGES=1
CROSS_EXTRA_EDGES=$(ini CrossExtraEdges); [ -z "$CROSS_EXTRA_EDGES" ] && CROSS_EXTRA_EDGES=10

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
/usr/bin/time -v "$ROOT/Release/spannbuilder" -c "$CFG" 2>&1

# --- (1) cross-graph: stitch the per-node head bundles (REQUIRED for >1 bundle).
#     Without head_cross_edges.bin, unfilter falls back to per-node fan-out.
#     Gated by [MultiTenant] CrossEdges in the ini. ---
if [ "$CROSS_EDGES" = "1" ] || [ "$CROSS_EDGES" = "true" ]; then
  echo "[launcher] cross-graph: augmentheadgraph -m $CROSS_EXTRA_EDGES (CrossEdges=$CROSS_EDGES)"
  "$ROOT/Release/augmentheadgraph" \
    -d "$OUT/tenant_0/HeadIndex" \
    -k 15 -m $CROSS_EXTRA_EDGES -t 16 -w true
else
  echo "[launcher] cross-graph DISABLED (CrossEdges=$CROSS_EDGES) -- skipping augmentheadgraph; unfilter will use per-node fan-out"
fi

# --- search needs the OPQ codebook in the tenant dir ---
cp "$SID/opq_quantizer.bin" "$OUT/tenant_0/opq_quantizer.bin"
echo "[launcher] copied opq_quantizer.bin into tenant_0"
echo "[launcher] done. SEARCH-time unfilter tail read: SPTAG_UNFILTER_TAIL=1"
