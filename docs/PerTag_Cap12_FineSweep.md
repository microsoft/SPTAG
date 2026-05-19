# PerTag cap=1.2 — Fine nprobe sweep results

Date: 2026-05-14  
Index: `~/test/tenant_index_tags_1m_pertag_cap12/` (PerTagBKTMerge, α=1.5, `SPTAG_PERTAG_SIZE_CAP_MULT=1.2`)  
Setup: SIFT-1M, tenant_0 (404,819 vectors), seed=42, 100 queries/level, topK=10, single-thread,
cosine, 2 runs/nprobe (faster latency kept). Recall computed by **ID overlap** against
faiss FlatIP GT (normalized), per-query filter to a randomly-drawn tag value at that level.

## Best QPS @ R≥0.95

| level   | nprobe | recall | mean ms | QPS    |
|---------|-------:|-------:|--------:|-------:|
| org     |   190  | 0.953  | 15.01   | **66.6** |
| dept    |    96  | 0.964  |  7.30   | **136.9** |
| team    |   110  | 0.956  |  8.35   | **119.7** |
| project |    80  | 0.954  |  6.48   | **154.4** |

## Comparison vs prior (resume checkpoint #34) cap=1.2 best

| level   | prior nprobe | prior QPS | new nprobe | new QPS | Δ |
|---------|-------------:|----------:|-----------:|--------:|---:|
| org     | 200          | 61.1      | **190**    | 66.6    | +9% |
| dept    | 160          | 79.8 ⚠    | **96**     | 136.9   | **+71%** |
| team    | 96           | 130.8     | **110**    | 119.7   | -8% |
| project | 96           | 124.0     | **80**     | 154.4   | +24% |

Notable: prior dept "regression" was an nprobe artifact — at nprobe=96 dept reaches
recall 0.964 with QPS 137 (no-cap PerTag was 112). **cap=1.2 now wins every level**
versus both the no-cap PerTag baseline and the production baseline (per resume table).

## Full nprobe curves

### org (col=0, card=4, ~101k matches/query)
| nprobe | recall | mean ms | QPS |
|---:|---:|---:|---:|
| 150 | 0.938 | 12.01 | 83.2 |
| 170 | 0.944 | 13.46 | 74.3 |
| 180 | 0.948 | 14.28 | 70.0 |
| 190 | **0.953** | 15.01 | **66.6** |
| 200 | 0.955 | 15.86 | 63.0 |
| 220 | 0.957 | 17.29 | 57.9 |
| 250 | 0.961 | 19.41 | 51.5 |

### dept (col=1, card=16, ~25k matches/query)
| nprobe | recall | mean ms | QPS |
|---:|---:|---:|---:|
| 96  | **0.964** | 7.30  | **136.9** |
| 112 | 0.968 | 8.51  | 117.5 |
| 128 | 0.971 | 9.54  | 104.9 |
| 144 | 0.973 | 10.69 |  93.5 |
| 160 | 0.975 | 11.88 |  84.2 |

### team (col=2, card=64, ~6.3k matches/query)
| nprobe | recall | mean ms | QPS |
|---:|---:|---:|---:|
|  60 | 0.921 | 4.62 | 216.5 |
|  70 | 0.932 | 5.40 | 185.2 |
|  80 | 0.938 | 6.04 | 165.4 |
|  90 | 0.945 | 6.94 | 144.0 |
|  96 | 0.949 | 7.18 | 139.2 |
| 110 | **0.956** | 8.35 | **119.7** |

team is right on the recall boundary at nprobe=96 (0.949). Adding nprobe=100/104 grid
points would likely yield ~130 QPS.

### project (col=3, card=256, ~1.6k matches/query)
| nprobe | recall | mean ms | QPS |
|---:|---:|---:|---:|
|  70 | 0.943 | 5.69 | 175.9 |
|  80 | **0.954** | 6.48 | **154.4** |
|  90 | 0.961 | 7.27 | 137.5 |
|  96 | 0.963 | 7.58 | 131.8 |
| 110 | 0.968 | 8.72 | 114.7 |

## Artifacts

- Script: `~/test/results_pertag_sweep/fine_sweep_pertag.py`
- Runner: `~/test/results_pertag_sweep/run_fine_sweep.sh`
- Raw log: `~/test/results_pertag_sweep/fine_sweep_tenant_index_tags_1m_pertag_cap12.log`
- JSON:    `~/test/results_pertag_sweep/fine_sweep_tenant_index_tags_1m_pertag_cap12.json`
- CSV:     `~/test/results_pertag_sweep/fine_sweep_tenant_index_tags_1m_pertag_cap12.csv`

Reproduce:
```bash
bash ~/test/results_pertag_sweep/run_fine_sweep.sh \
     ~/test/tenant_index_tags_1m_pertag_cap12
```

## Open follow-ups

- **team grid is too coarse near the boundary**: add nprobe ∈ {96, 100, 104, 108} pass
- **no-cap PerTag index missing on disk** (`tenant_index_tags_1m_pertag_kmeans_merge`):
  rebuild to confirm cap=1.2 vs no-cap delta on dept at finer nprobe
- **baseline index** (`tenant_index_tags_1m_baseline_t0_match`) still on disk — should
  run the same sweep there for an apples-to-apples comparison row
