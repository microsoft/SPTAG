#!/usr/bin/env bash
# Rebuild only the global SIFT1B BKT HeadIndex. STM1 postings remain untouched.
set -euo pipefail

[[ $# -eq 0 ]] || {
  echo "usage: $0" >&2
  exit 2
}

cd "$(dirname "$0")/../.."
ROOT=$(pwd)
CFG="$ROOT/Tools/benchmarks/build_spann_static_signature_sift1b.ini"

[[ -f "$CFG" ]] || { echo "missing native build config: $CFG" >&2; exit 2; }
grep -A12 '^\[BuildHead\]$' "$CFG" | grep -qx 'BKTLambdaFactor=-1.0' ||
  { echo "BuildHead.BKTLambdaFactor must be -1.0 in $CFG" >&2; exit 2; }

exec /usr/bin/time -v "$ROOT/Release/spannbuilder" -c "$CFG" --rebuild-head-index
