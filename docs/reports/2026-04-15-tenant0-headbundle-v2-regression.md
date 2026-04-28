# Tenant 0 Head-Bundle V2 Rebuild Regression

Date: 2026-04-15

This note compares the previously validated tenant-0 head-bundle benchmark against the rebuilt manifest-v2 index after removing manifest-v1 compatibility and rebuilding the 1M head-bundle artifact.

## Inputs

- Previous validated head-bundle result: `/home/v-mochengli/test/tenant0_selectivity_latency_20260415_headbundle/result.json`
- Rebuilt v2 head-bundle result: `/home/v-mochengli/test/tenant0_selectivity_latency_20260415_headbundle_v2/result.json`
- Rebuilt v2 scenario: `/home/v-mochengli/test/tenant_tag_scenario_1m_headbundle_20260415_v2.json`
- Rebuilt v2 index: `/home/v-mochengli/test/tenant_index_tags_1m_headbundle_20260415_v2`

## Per-Level Delta

| Level | Recall Old | Recall New | Recall Delta | QPS Old | QPS New | QPS Delta | Avg Lat Old | Avg Lat New | Lat Delta | Avg Posting Read Old | Avg Posting Read New |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| org | 0.9990 | 0.9980 | -0.0010 | 39.48 | 0.46 | -39.02 | 25.30 ms | 2166.27 ms | +2140.97 ms | 127.88 | 127.88 |
| dept | 0.9870 | 0.9910 | +0.0040 | 20.66 | 1.67 | -18.99 | 48.37 ms | 598.42 ms | +550.05 ms | 254.85 | 254.85 |
| team | 0.9800 | 0.9910 | +0.0110 | 9.17 | 0.92 | -8.25 | 108.99 ms | 1086.03 ms | +977.04 ms | 503.83 | 503.83 |
| project | 0.9910 | 0.9970 | +0.0060 | 5.49 | 5.47 | -0.03 | 181.98 ms | 182.89 ms | +0.92 ms | 995.26 | 995.26 |

## Readout

- Removing manifest-v1 support and rebuilding the 1M index did not improve performance.
- The rebuilt v2 index is dramatically slower on `org`, `dept`, and `team`, while `project` stayed essentially flat.
- Average posting-read counts stayed unchanged across all four levels, which means the current node-aware adaptive-nprobe change did not materially reduce posting fanout on this workload.
- The rebuilt tenant-0 artifact was confirmed to use manifest version `2` with `5` head-bundle nodes.

## Runtime Symptom

- The v2 benchmark log again showed many `AsyncFileReader::ReadBlocks ... timeout, continue for next batch...` lines during the slow run.
- The benchmark log also showed routed head-bundle graph search executing across `1` node for the rebuilt tenant-0 workload.

## Current Conclusion

- Deleting v1 compatibility is fine operationally, but it is not the reason performance was low.
- The first measured run after rebuilding the v2 index is not representative of steady-state performance.

## Steady-State Check

- A second benchmark run against the same rebuilt v2 index, without rebuilding, recovered to:
	- `org`: `39.78 QPS`, `25.11 ms`
	- `dept`: `20.78 QPS`, `48.09 ms`
	- `team`: `9.51 QPS`, `105.16 ms`
	- `project`: `5.51 QPS`, `181.41 ms`
- A formal warm benchmark with `warmup_queries=100` produced essentially the same steady-state result:
	- `org`: `39.87 QPS`, `25.05 ms`
	- `dept`: `21.10 QPS`, `47.35 ms`
	- `team`: `9.47 QPS`, `105.51 ms`
	- `project`: `5.49 QPS`, `182.03 ms`
- This means the severe `org/dept/team` collapse in the first v2 report was a cold-start IO artifact immediately after rebuild, not a persistent algorithmic regression of the rebuilt index.
- The benchmark harness now supports explicit warmup via `--warmup-queries` so future reports can measure steady-state behavior directly.