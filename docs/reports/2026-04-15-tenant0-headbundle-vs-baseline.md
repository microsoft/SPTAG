# Tenant 0 Head-Bundle Vs Baseline

Date: 2026-04-15

This note compares the tenant-0 selectivity benchmark from the pre-head-bundle baseline with the current node-split head-bundle implementation.

## Inputs

- Baseline result: `/home/v-mochengli/test/tenant0_selectivity_latency_20260414/result.json`
- Current result: `/home/v-mochengli/test/tenant0_selectivity_latency_20260415_headbundle/result.json`
- Query file: `/home/v-mochengli/dataset/sift/sift_query.fvecs`
- Num queries: `100`
- TopK: `10`
- Search params stayed aligned:
  - `ForceDenseTagSearch=false`
  - `DirectSparseMaxPostings=320`
  - `FilteredSearchNprobeSafety=1.0`
  - `FilteredSearchTargetRecall=1.0`
  - `FilteredSearchCoverageExponent=0.5`

## Per-Level Delta

| Level | Recall Old | Recall New | Recall Delta | QPS Old | QPS New | QPS Delta | QPS Delta % | Avg Lat Old | Avg Lat New | Lat Delta | Lat Delta % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| org | 0.9840 | 0.9990 | +0.0150 | 112.41 | 39.48 | -72.93 | -64.88% | 8.85 ms | 25.30 ms | +16.45 ms | +185.82% |
| dept | 0.9860 | 0.9870 | +0.0010 | 57.40 | 20.66 | -36.74 | -64.01% | 17.39 ms | 48.37 ms | +30.98 ms | +178.14% |
| team | 0.9930 | 0.9800 | -0.0130 | 29.32 | 9.17 | -20.15 | -68.72% | 34.07 ms | 108.99 ms | +74.92 ms | +219.88% |
| project | 0.9880 | 0.9910 | +0.0030 | 14.70 | 5.49 | -9.20 | -62.62% | 68.00 ms | 181.98 ms | +113.98 ms | +167.61% |

## Aggregate Direction

- Simple average recall moved from `0.9878` to `0.9893` (`+0.0015`).
- Simple average QPS moved from `53.46` to `18.70` (`-65.02%`).
- Simple average latency moved from `32.08 ms` to `91.16 ms` (`+184.17%`).

## Readout

- The current head-bundle implementation did **not** improve tenant-0 QPS on this workload.
- Recall is roughly flat overall and slightly better on `org`, `dept`, and `project`, but `team` regressed.
- The dominant change is performance cost: every level lost about `63%` to `69%` of QPS, with latency increasing by about `168%` to `220%`.
- The 2026-04-15 run did confirm that the routed node head-bundle graph-search path was actually exercised during benchmark queries.

## Caveat

- The 1M build harness currently has a manifest-count verification bug after save/load for non-identity tenant mappings. The benchmark above still ran on the saved index successfully, so the benchmark numbers are valid for the saved artifact, but the helper script needs a follow-up cleanup.