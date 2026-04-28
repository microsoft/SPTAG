# Current Performance Comparison

Date: 2026-04-13

This note captures the current validated performance snapshot for the tenant + tag filtered-search system on the working branch.

## Scope

- Multi-tenant cache stress source artifact: `/tmp/multitenant_tag_cache_stress_20260413_071849/summary.md`
- Top-k sweep source artifact: `/tmp/filter_perf_1m_topk_sweep_20260413_062209/summary.md`
- Default filtered-search params used in the validated runs:
  - `DirectSparseMaxPostings=320`
  - `FilteredSearchNprobeSafety=1.0`
  - `FilteredSearchTargetRecall=1.0`
  - `FilteredSearchCoverageExponent=0.5`
  - `ForceDenseTagSearch=false`

## Multi-Tenant Cache Stress

Workload:

- 10 tenants
- 1000 queries
- 10 batches x 100 queries
- single-tag ACL filter per query
- `topk=10`
- HeadIndex cache limit: `111 MB`
- Total HeadIndex size across tenants: about `136.04 MB`
- Cache budget ratio: about `81.59%` of total HeadIndex size

Overall comparison:

| Scenario | Avg Latency | P95 | P99 | QPS | Avg Recall | Avg Selectivity | FP Rate | Cache End |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sequential tenant batches | 31.87 ms | 74.85 ms | 84.85 ms | 31.38 | 0.9953 | 8.50% | 67.66% | 72.65 MB |
| random mixed tenants | 47.27 ms | 122.46 ms | 288.86 ms | 21.15 | 0.9945 | 8.35% | 68.31% | 100.54 MB |
| random vs sequential delta | +48.32% | +63.61% | +240.44% | -32.60% | -0.0008 | -0.15 pp | +0.65 pp | +27.89 MB |

Key observations:

- Correctness stayed stable under cache pressure: both scenarios had `avg_valid=10.00`, `avg_expected_valid=10.00`, and `shortfall_queries=0`.
- Random tenant interleaving mainly hurts tail latency and throughput, not recall.
- Random mixing ends with a much fuller HeadIndex cache footprint, which is consistent with reduced locality.
- This benchmark is useful for evaluating cache scheduling and tenant mixing effects, not for proving worst-case latency bounds.

## SIFT1M Top-k Sweep

The following compact table keeps the same method labels as the recorded artifact and shows how recall and latency move as `k` grows.

| Filter | K=10 | K=20 | K=50 | K=100 |
| --- | --- | --- | --- | --- |
| no_filter | recall 0.980, 4.69 ms | 0.960, 4.81 ms | 0.950, 4.57 ms | 0.950, 6.76 ms |
| 1_org | recall 0.980, 8.88 ms | 0.975, 10.06 ms | 0.958, 10.07 ms | 0.955, 16.50 ms |
| 1_dept | recall 0.980, 17.79 ms | 0.975, 18.06 ms | 0.948, 18.23 ms | 0.930, 28.34 ms |
| 1_team | recall 0.990, 35.89 ms | 0.980, 34.15 ms | 0.872, 42.74 ms | 0.926, 86.66 ms |
| 1_project | recall 0.980, 69.85 ms | 0.900, 68.23 ms | 0.926, 171.29 ms | 0.940, 363.36 ms |

High-level reading:

- Broad filters such as `1_org` remain relatively stable as `k` grows, with moderate latency growth.
- Medium filters such as `1_dept` stay usable but show a visible recall drop by `k=100`.
- Narrow filters such as `1_team` and `1_project` are not stable across `k`; latency grows sharply and recall is no longer monotonic.
- For fixed selectivity, false-positive rate is largely determined by the filter regime rather than by `k`.

## What The Current Snapshot Says

- The multi-tenant cache design is operational and keeps recall near 1.0 under realistic mixed-tenant pressure.
- The filtered search defaults are acceptable for `topk=10` and still reasonable for broader filters at larger `k`.
- The current system is not yet `topk`-general for narrow filters. This is the main performance-quality tradeoff still exposed by the sweep.

## Recommended Interpretation For Optimization Work

- If the next round focuses on multi-tenant serving, target cache locality and tenant scheduling first.
- If the next round focuses on filtered-search quality, target dense-path coverage and `k` scaling first.
- The best next experiment is a cache-budget sweep at fixed workload plus a `topk=10/50/100` rerun under the same benchmark harness.