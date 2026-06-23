#!/bin/bash
# =============================================================================
# Deterministic prep of the SPACEV-1B builder inputs referenced by
#   Tools/benchmarks/build_spann_attr_spacev1b_opq25.ini
#
# Pure C++ (no Python): drives the spannbuilder subcommands that mimic the
# original AnnService/src/Quantizer/main.cpp, so the artifacts are byte-for-byte
# what the SPANN build/search trust (validated against the 3M sidecar).
#
# Produces, under $OUT (default /datadisk/yfcc_fast/spacev1b_build):
#   spacev1b_tags5.u32       [N,5] uint32 = [org,dept,team,project | price]   (--merge-tags5)
#   spacev1b_group_tags.txt  org column, one int/line (PerTagBKT routing key)  (--merge-tags5)
#   opq_codes_m25.bin        [N,25] uint8 raw OPQ codes                        (--gen-opq-codes)
#   opq_quantizer.bin        the 3M-trained OPQ codebook (search-time ADC)     (copied)
#
# Usage:  Tools/benchmarks/prep_spacev1b_inputs.sh [N]
#         (N optional: limit #vectors for a smoke run; default = all 1e9)
# =============================================================================
set -e
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
SB="$ROOT/Release/spannbuilder"

DS=/home/v-mochengli/datasets/big-ann/MSSPACEV1B
BASE=$DS/spacev1b_base.i8bin
TAGS=$DS/multitenant/tags.npy
NUM=$DS/multitenant/num_attr.npy
CODEBOOK=/datadisk/yfcc_fast/spacev_opq25_sidecar/opq_quantizer.bin   # 3M-trained OPQ codebook
OUT=/datadisk/yfcc_fast/spacev1b_build
N="${1:--1}"                                                          # -1 = all

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
mkdir -p "$OUT"
echo "[prep] OUT=$OUT  N=$N"

# (1)+(2) tag sidecar + routing-key column (C++ .npy reader)
"$SB" --merge-tags5 \
  --tags-npy "$TAGS" --num-npy "$NUM" \
  --out-tags5 "$OUT/spacev1b_tags5.u32" --out-group "$OUT/spacev1b_group_tags.txt" \
  --acl-cols 4 --group-col 0 --n "$N"

# (3) in-posting OPQ codes (C++ encoder, raw widen + ADC=false, headerless N*M)
"$SB" --gen-opq-codes \
  --vectors "$BASE" --vec-offset 8 --dim 100 --value-type Int8 \
  --quantizer "$CODEBOOK" --out "$OUT/opq_codes_m25.bin" --n "$N"

# (4) codebook for search-time ADC
cp "$CODEBOOK" "$OUT/opq_quantizer.bin"
echo "[prep] copied opq_quantizer.bin"
echo "[prep] done. Build with: Tools/benchmarks/run_spann_attr_build.sh Tools/benchmarks/build_spann_attr_spacev1b_opq25.ini"
