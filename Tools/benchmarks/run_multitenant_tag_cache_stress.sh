#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
bench_script="$script_dir/multitenant_tag_cache_stress.py"

scenario_file=${SPTAG_STRESS_SCENARIO_FILE:-/home/v-mochengli/test/tenant_tag_scenario_1m.json}
query_file=${SPTAG_STRESS_QUERY_FILE:-/home/v-mochengli/dataset/sift/sift_query.fvecs}
output_root=${SPTAG_STRESS_OUTPUT_ROOT:-/tmp}
python_bin=${SPTAG_STRESS_PYTHON:-/home/v-mochengli/anaconda3/envs/py310/bin/python}
ld_preload=${SPTAG_STRESS_LD_PRELOAD:-/lib/x86_64-linux-gnu/libjemalloc.so.2}

num_queries=${SPTAG_STRESS_NUM_QUERIES:-1000}
batch_size=${SPTAG_STRESS_BATCH_SIZE:-100}
topk=${SPTAG_STRESS_TOPK:-10}
tenant_range=${SPTAG_STRESS_TENANT_RANGE:-0,1,2,3,4,5,6,7,8,9}
seed=${SPTAG_STRESS_SEED:-20260413}
cache_limit_mb=${SPTAG_STRESS_CACHE_LIMIT_MB:-auto}
rss_high_water_mb=${SPTAG_STRESS_RSS_HIGH_WATER_MB:-}
rss_high_water_sweep_mb=${SPTAG_STRESS_RSS_HIGH_WATER_SWEEP_MB:-}
drop_page_cache_on_evict=${SPTAG_STRESS_DROP_PAGE_CACHE_ON_EVICT:-true}

force_dense_tag_search=${SPTAG_STRESS_FORCE_DENSE_TAG_SEARCH:-false}
direct_sparse_max_postings=${SPTAG_STRESS_DIRECT_SPARSE_MAX_POSTINGS:-320}
filtered_search_nprobe_safety=${SPTAG_STRESS_FILTERED_SEARCH_NPROBE_SAFETY:-1.0}
filtered_search_target_recall=${SPTAG_STRESS_FILTERED_SEARCH_TARGET_RECALL:-1.0}
filtered_search_coverage_exponent=${SPTAG_STRESS_FILTERED_SEARCH_COVERAGE_EXPONENT:-0.5}

ts=$(date +%Y%m%d_%H%M%S)
out_dir="${output_root%/}/multitenant_tag_cache_stress_${ts}"
mkdir -p "$out_dir"
printf '%s\n' "$out_dir" > "${output_root%/}/multitenant_tag_cache_stress_latest.txt"

log_file="$out_dir/benchmark.log"
status_file="$out_dir/status.txt"
meta_file="$out_dir/meta.txt"

git_commit=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || printf 'unknown')
if [[ -n $(git -C "$repo_root" status --porcelain 2>/dev/null || true) ]]; then
  git_dirty=true
else
  git_dirty=false
fi

cat <<EOF > "$meta_file"
timestamp=$ts
repo_root=$repo_root
runner=$0
script=$bench_script
python=$python_bin
scenario_file=$scenario_file
query_file=$query_file
output_dir=$out_dir
num_queries=$num_queries
batch_size=$batch_size
topk=$topk
tenant_range=$tenant_range
seed=$seed
cache_limit_mb=$cache_limit_mb
rss_high_water_mb=$rss_high_water_mb
rss_high_water_sweep_mb=$rss_high_water_sweep_mb
drop_page_cache_on_evict=$drop_page_cache_on_evict
force_dense_tag_search=$force_dense_tag_search
direct_sparse_max_postings=$direct_sparse_max_postings
filtered_search_nprobe_safety=$filtered_search_nprobe_safety
filtered_search_target_recall=$filtered_search_target_recall
filtered_search_coverage_exponent=$filtered_search_coverage_exponent
ld_preload=$ld_preload
git_commit=$git_commit
git_dirty=$git_dirty
EOF

cmd=(
  "$python_bin" -u "$bench_script"
  --scenario-file "$scenario_file"
  --query-file "$query_file"
  --output-dir "$out_dir"
  --num-queries "$num_queries"
  --batch-size "$batch_size"
  --topk "$topk"
  --tenant-range "$tenant_range"
  --seed "$seed"
  --direct-sparse-max-postings "$direct_sparse_max_postings"
  --filtered-search-nprobe-safety "$filtered_search_nprobe_safety"
  --filtered-search-target-recall "$filtered_search_target_recall"
  --filtered-search-coverage-exponent "$filtered_search_coverage_exponent"
)

if [[ "$cache_limit_mb" != "auto" ]]; then
  cmd+=(--cache-limit-mb "$cache_limit_mb")
fi

if [[ -n "$rss_high_water_mb" ]]; then
  cmd+=(--rss-high-water-mb "$rss_high_water_mb")
fi

if [[ -n "$rss_high_water_sweep_mb" ]]; then
  cmd+=(--rss-high-water-sweep-mb "$rss_high_water_sweep_mb")
fi

if [[ "$drop_page_cache_on_evict" == "true" ]]; then
  cmd+=(--drop-page-cache-on-evict)
fi

if [[ "$force_dense_tag_search" == "true" ]]; then
  cmd+=(--force-dense-tag-search)
fi

printf 'running\n' > "$status_file"

run_benchmark() {
  export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
  if [[ -n "$ld_preload" ]]; then
    export LD_PRELOAD="$ld_preload"
  fi
  cd "$repo_root"
  "${cmd[@]}"
}

if run_benchmark > "$log_file" 2>&1; then
  printf 'success\n' > "$status_file"
else
  printf 'failed\n' > "$status_file"
  exit 1
fi

printf '%s\n' "$out_dir"