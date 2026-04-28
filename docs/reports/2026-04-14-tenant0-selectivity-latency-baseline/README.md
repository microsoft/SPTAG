# Tenant 0 Selectivity Latency Baseline

Date: 2026-04-14

This file snapshots a validated baseline so future code changes can be compared against a fixed reference.

## Code Revision

- Git commit: `ca138730a7932d9574857a9799534bff40b97f99`
- Dirty working tree file count at run time: `16`

## Benchmark Inputs

- Scenario file: `/home/v-mochengli/test/tenant_tag_scenario_1m.json`
- Index dir: `/home/v-mochengli/test/tenant_index_tags_1m`
- Query file: `/home/v-mochengli/dataset/sift/sift_query.fvecs`
- Tenant: `0` (internal id `0`)
- Num queries: `100`
- TopK: `10`
- ForceDenseTagSearch: `false`
- DirectSparseMaxPostings: `320`
- FilteredSearchNprobeSafety: `1.0`
- FilteredSearchTargetRecall: `1.0`
- FilteredSearchCoverageExponent: `0.5`

## Baseline Metrics

| Level | Tag | Selectivity | Match Count | Recall | QPS | Avg Latency | P95 | P99 | Avg Nprobe(PostingRead) | Avg Valid | FP Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| org | 1001 | 25.0285% | 101320 | 0.9840 | 112.41 | 8.85 ms | 12.62 ms | 23.53 ms | 127.59 | 10.00 | 0.70% |
| dept | 2007 | 6.3204% | 25586 | 0.9860 | 57.40 | 17.39 ms | 30.22 ms | 37.02 ms | 254.19 | 10.00 | 19.71% |
| team | 3062 | 1.6153% | 6539 | 0.9930 | 29.32 | 34.07 ms | 47.03 ms | 53.03 ms | 502.51 | 10.00 | 63.11% |
| project | 4117 | 0.4133% | 1673 | 0.9880 | 14.70 | 68.00 ms | 88.57 ms | 96.92 ms | 993.10 | 10.00 | 88.18% |

## Artifact Files Used

- Report markdown: `/home/v-mochengli/test/tenant0_selectivity_latency_20260414/report.md`
- Result json: `/home/v-mochengli/test/tenant0_selectivity_latency_20260414/result.json`
- Workload jsonl: `/home/v-mochengli/test/tenant0_selectivity_latency_20260414/query_workload.jsonl`

## Artifact SHA256

- `report.md`: `fd099a8d6836f2187084622ef7821991cfd79c9b39e3f80db71a2e1dfbdfd2ac`
- `result.json`: `8b2801a0bf137a062ee788041aebf5a648dca8f660c1a7c3aef0137fc0ad89fc`
- `query_workload.jsonl`: `1e401f35a23cae52c34e45b684539d599b181771a54f36660b437df05770a4e8`

## Reproduce Command (py310)

```bash
cd /home/v-mochengli/SPTAG
export LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2
/home/v-mochengli/anaconda3/envs/py310/bin/python Tools/benchmarks/report_tenant0_selectivity_latency.py \
  --num-queries 100 \
  --topk 10 \
  --output-dir /home/v-mochengli/test/tenant0_selectivity_latency_20260414
```

## Notes

- Use this file as the baseline anchor for regression checks after search-path or cache-policy code changes.
- For future reruns, keep query count and runtime params consistent before comparing latency and QPS.
