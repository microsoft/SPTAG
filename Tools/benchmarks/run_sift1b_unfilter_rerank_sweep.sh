#!/usr/bin/env bash
# Sweep the unfiltered SIFT1B frontier without changing the canonical L=500 curve.
set -euo pipefail

cd /home/baotonglu/mocheng/SPTAG
export LD_LIBRARY_PATH="$PWD/Release:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/Release"
export SPTAG_UNFILTER_TAIL=1

IDX=/mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_spann_pipepq32_tail8_page1
INI="$IDX/tenant_0/indexloader.ini"
QRY=/mnt/nvme/baotonglu/mocheng/datasets/sift1b/multitenant/query
OUT="${OUT:-/mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_unfilter_rerank_sweep.jsonl}"
LOG="${LOG:-${OUT}.log}"
NQ="${NQ:-100}"
WARMUP="${WARMUP:-20}"
TOPK="${TOPK:-100}"
NPROBES="${NPROBES:-100 150 200 250 300 400}"
RERANK_LS="${RERANK_LS:-100 150 200 300 500}"
MAX_CHECKS="${MAX_CHECKS:-4096}"
TAIL_MODE="${TAIL_MODE:-full}"

case "$TAIL_MODE" in
    full)
        unset SPTAG_UNFILTER_PURE_PAGES SPTAG_UNFILTER_EXTRA_TAIL_PAGES
        unset SPTAG_ABLATE_TAIL SPTAG_ABLATE_UEXTRA
        ;;
    pure-pages)
        export SPTAG_UNFILTER_PURE_PAGES=1
        unset SPTAG_UNFILTER_EXTRA_TAIL_PAGES SPTAG_ABLATE_TAIL SPTAG_ABLATE_UEXTRA
        ;;
    no-tail)
        export SPTAG_ABLATE_TAIL=1
        export SPTAG_ABLATE_UEXTRA=1
        unset SPTAG_UNFILTER_PURE_PAGES SPTAG_UNFILTER_EXTRA_TAIL_PAGES
        ;;
    *)
        echo "TAIL_MODE must be one of: full, pure-pages, no-tail" >&2
        exit 1
        ;;
esac

[[ -f "$INI" ]] || { echo "Missing index config: $INI" >&2; exit 1; }
[[ "$NQ" -gt 0 && "$WARMUP" -ge 0 && "$TOPK" -gt 0 ]] ||
    { echo "NQ and TOPK must be positive; WARMUP must be non-negative" >&2; exit 1; }
for rerank_l in $RERANK_LS; do
    [[ "$rerank_l" -ge "$TOPK" ]] ||
        { echo "RerankL ($rerank_l) must be at least TOPK ($TOPK)" >&2; exit 1; }
done
for max_check in $MAX_CHECKS; do
    [[ "$max_check" -gt 0 ]] ||
        { echo "MaxCheck must be positive: $max_check" >&2; exit 1; }
done

exec 9>/tmp/sptag-sift1b-spann-indexloader.lock
flock -n 9 || { echo "Another SIFT1B indexloader sweep is active" >&2; exit 1; }

backup="$(mktemp "${INI}.rerank-sweep.XXXXXX")"
cp -p "$INI" "$backup"
restore_ini() {
    cp -p "$backup" "$INI"
    rm -f "$backup"
}
trap restore_ini EXIT HUP INT TERM

set_maxcheck() {
    awk -v max_check="$1" '
        /^\[/ { section = $0 }
        section == "[BuildSSDIndex]" && /^MaxCheck=/ {
            print "MaxCheck=" max_check
            found++
            next
        }
        { print }
        END { exit found == 1 ? 0 : 1 }
    ' "$INI" > "${INI}.tmp"
    mv "${INI}.tmp" "$INI"
}

mkdir -p "$(dirname "$OUT")"
tmp="${OUT}.tmp"
rm -f "${LOG}".*.tmp "${tmp}".*.result
: > "$tmp"
: > "$LOG"
printf '# SIFT1B unfilter RerankL/MaxCheck sweep start %s nq=%s warmup=%s topk=%s nprobes=[%s] rerankLs=[%s] maxChecks=[%s] tailMode=%s\n' \
    "$(date -u +%FT%TZ)" "$NQ" "$WARMUP" "$TOPK" "$NPROBES" "$RERANK_LS" "$MAX_CHECKS" "$TAIL_MODE" | tee -a "$tmp" "$LOG"

for max_check in $MAX_CHECKS; do
    set_maxcheck "$max_check"
    grep -A48 '^\[BuildSSDIndex\]$' "$INI" | grep -qx "MaxCheck=${max_check}" ||
        { echo "Failed to set MaxCheck=${max_check}" >&2; exit 1; }

    for rerank_l in $RERANK_LS; do
        sed -i "s/^RerankL=.*/RerankL=${rerank_l}/" "$INI"
        grep -qx "RerankL=${rerank_l}" "$INI" ||
            { echo "Failed to set RerankL=${rerank_l}" >&2; exit 1; }

        for nprobe in $NPROBES; do
            printf '=== MaxCheck=%s RerankL=%s nprobe=%s %s ===\n' \
                "$max_check" "$rerank_l" "$nprobe" "$(date -u +%FT%TZ)" | tee -a "$tmp" "$LOG"
            result_tmp="${tmp}.${max_check}.${rerank_l}.${nprobe}.result"
            run_log="${LOG}.${max_check}.${rerank_l}.${nprobe}.tmp"

            SPTAG_FIXED_NPROBE="$nprobe" \
            SPTAG_TAG_AWARE_HEAD_EXPANSION=1 \
            SPTAG_DISABLE_HIER_POSTING_FILTER=1 \
            INDEX_DIR="$IDX" \
            QUERY_DIR="$QRY" \
            TOPK="$TOPK" \
            NUM_QUERIES="$NQ" \
            WARMUP="$WARMUP" \
            LEVELS=unfilter \
            SPTAG_VALUE_TYPE=UInt8 \
            TEST_MAXCHECK="$max_check" \
                timeout 3000 python3 Tools/benchmarks/efsearch_probe_levels.py \
                > "$run_log" 2>&1
            cat "$run_log" >> "$LOG"
            awk '/^RESULT / { print }' "$run_log" > "$result_tmp"
            rm -f "$run_log"

            python3 - "$result_tmp" "$nprobe" "$rerank_l" "$max_check" "$TAIL_MODE" <<'PY' >> "$tmp"
import json
import sys

path, nprobe, rerank_l, max_check, tail_mode = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
)
rows = [json.loads(line[7:]) for line in open(path, encoding="utf-8")
        if line.startswith("RESULT ")]
assert len(rows) == 1 and rows[0]["level"] == "unfilter", rows
row = rows[0]
assert row["nprobe"] == nprobe, row
row["policy"] = "distance-first-post-filter"
row["rerankL"] = rerank_l
row["unfilter_tail"] = True
row["maxcheck"] = max_check
row["tail_mode"] = tail_mode
print("RESULT " + json.dumps(row, sort_keys=True))
PY
            rm -f "$result_tmp"
        done
    done
done

mv "$tmp" "$OUT"
printf '# SIFT1B unfilter RerankL/MaxCheck sweep done %s\n' "$(date -u +%FT%TZ)" | tee -a "$OUT" "$LOG"
