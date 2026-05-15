# SPFresh SIFT1B v10 Metric Report

Date: 2026-05-07

Scope: analysis of the live benchmark run `benchmark_20260507_032733.log` and `evaluation/2026-04-23/output.json`. At the latest collection time, `output.json` contained six completed batches and the log had started batch 7. Batch 7 is not included because it had not completed.

## 1. Executive Summary

The first six completed batches show stable-ish search quality and post-batch latency, but insert throughput keeps dropping as layer0 split/reassign work grows. The strongest evidence still points to layer0 split/reassign amplification. TiKV latency rises as the request volume grows, but TiKV health signals do not show write stall or compaction backlog.

Key conclusions:

- Insert throughput drops from `579.4 vec/s` in batch 1 to `229.3 vec/s` in batch 6, a `60.42%` drop.
- Post-batch search remains usable: mean latency moves from `7.06 ms` to `8.05 ms`, and recall remains `0.973-0.982`.
- During-insert search latency is higher than batch 1 but not monotonic: mean latency is `7.14 ms` in batch 1, peaks at `10.75 ms` in batch 3, dips in batch 5, and is `10.67 ms` in batch 6.
- TiKV raw operation averages have increased versus the earlier reads: raw get `0.333-0.508 ms`, raw put `0.836-1.034 ms`, raw batch get `0.310-0.445 ms` across the three TiKV nodes.
- TiKV pressure signals are clean: write stall is `0`, pending compaction is `0`, stores are `Up`, and `slow_score=1`.
- DIAG shows layer0 split/reassign explosion by batch 6: `498,514` splits and `13,977,020` reassign submissions, with split max latency `8.70 s`.
- `BatchSplitPostingVectors avg` stays at about `247.2` in batches 3-6, while runtime posting size limit is `246`, so postings are splitting almost exactly at the configured page limit.
- Layer1 pressure continues to grow: batch 4 has `1` layer1 split, batch 5 has `27`, and batch 6 has `57` layer1 splits with `7,106` layer1 reassign submissions.

## 2. Runtime Configuration Evidence

Active config values from `benchmark_spfresh_sift1b_v10_multichunk.ini`:

| Setting | Value |
| --- | ---: |
| `UseMultiChunkPosting` | `false` |
| `PostingPageLimit` | `8` |
| `PostingCountCacheCapacity` | `1000000` |
| `DistributedVersionMap` | `true` |
| `ReassignK` | `64` |
| `AsyncMergeInSearch` | `true` |
| `VersionCacheMaxChunks` | `100000` |

Runtime log confirmation:

| Runtime item | Value |
| --- | ---: |
| Posting page limit | `8` |
| Posting size limit | `246` vectors |
| Version map | TiKV distributed version map |
| Version chunk size | `4096` |
| Version cache max chunks | `100000` |
| Multi-chunk posting | disabled |

Interpretation: this run is the single-key posting path, not the multi-chunk RawScan path. Version cache is enabled by `VersionCacheMaxChunks > 0`.

## 3. Completed Batch Metrics

Source: `evaluation/2026-04-23/output.json`, completed batch entries 1-6.

| Batch | Loaded vectors | Insert time (s) | Insert throughput (vec/s) | During-insert mean (ms) | During p95 (ms) | During p99 (ms) | Post-search mean (ms) | Post p95 (ms) | Post p99 (ms) | Post QPS | Recall@5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,000,000 | 1,725.9071 | 579.4055 | 7.1368 | 8.5670 | 35.2620 | 7.0611 | 8.4010 | 8.7490 | 562.4550 | 0.9770 |
| 2 | 2,000,000 | 1,793.2889 | 557.6346 | 8.3366 | 9.9530 | 39.0620 | 7.1031 | 9.0230 | 9.6070 | 560.4235 | 0.9820 |
| 3 | 3,000,000 | 2,492.2107 | 401.2502 | 10.7519 | 15.0000 | 38.9840 | 7.5841 | 9.2200 | 9.6530 | 525.6794 | 0.9820 |
| 4 | 4,000,000 | 3,371.4475 | 296.6085 | 10.0975 | 12.6910 | 34.8000 | 7.8953 | 10.1240 | 11.1430 | 502.8613 | 0.9790 |
| 5 | 5,000,000 | 3,895.7402 | 256.6906 | 9.0705 | 13.9000 | 36.2240 | 8.7304 | 9.9100 | 10.4390 | 455.4906 | 0.9770 |
| 6 | 6,000,000 | 4,360.8887 | 229.3111 | 10.6737 | 17.1710 | 39.5830 | 8.0503 | 10.1180 | 10.8690 | 494.8609 | 0.9730 |

Derived changes:

| Metric | Batch 1 -> Batch 6 |
| --- | ---: |
| Insert throughput | `-60.42%` |
| Insert time | `+152.67%` |
| During-insert mean latency | `+49.56%` |
| Post-search mean latency | `+14.01%` |
| Post-search QPS | `-12.02%` |

Interpretation: insertion is the part that deteriorates sharply. Search after each batch degrades only mildly by comparison, and recall remains stable.

## 3.1 Vector / Head Counts After Each Completed Batch

There are three different count concepts in the logs and checkpoint files:

- `SaveIndex(... GetNumSamples())` / `Current Vector Num` is the layer0 data-vector universe exposed by `Index::GetNumSamples()`, i.e. total inserted data vectors.
- `Save Vector (...,128)` is the saved in-memory top `HeadIndex` vector count.
- `SPTAGHeadVectorIDs.bin_layer0` / `SPTAGHeadVectors.bin_layer0` represent the layer0 head-vector set used as the input universe for the next layer. From file size, this is `200,928` rows in the available saved snapshots.

Directly observed checkpoint counts:

| Completed batch | Checkpoint path | Layer0 data vectors (`GetNumSamples`) | Deleted | Top `HeadIndex` vectors (`Save Vector`) | Layer0 head-vector file rows |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `spann_index_0` | `2,000,000` | `0` | `40,428` | not available now; copied from base |
| 2 | `spann_index_1` | `3,000,000` | `0` | `40,428` | not available now; copied from previous checkpoint |
| 3 | `spann_index_2` | `4,000,000` | `0` | `40,428` | `200,928` |
| 4 | `spann_index_3` | `5,000,000` | `0` | `40,429` | not printed in log |
| 5 | `spann_index_4` | `6,000,000` | `0` | `40,445` | not printed in log |
| 6 | `spann_index_5` | `7,000,000` | `0` | `40,482` | not printed in log |

Important limitation: this running process does not include the later checkpoint instrumentation that prints dynamic `GetNumSamples(layer)` and `GetHeadIndexMapping(layer).size()`. Therefore, if "layer-1 vector count" means the dynamic layer1 extra-searcher contained-vector/sample count after split-time `AddHeadIndex(..., tolayer=1)`, that exact per-batch value cannot be reliably reconstructed from this log. The next run should use the new checkpoint layer stats to capture it directly.

## 4. DIAG Evidence

### 4.1 Per-Batch Split / Reassign Counts

Counter semantics check: the per-batch values below were extracted by segmenting the log from `========== BATCH n/10` to `Checkpoint saved: batch n/10`, then taking the last `total_submitted` line for each layer inside that segment. These are per-batch counters, not global cumulative counters.

Important metric granularity note: `total_submitted reassign` is the comparable per-batch reassign submission/record counter available in every batch progress block. `ReassignJobUs count` is a full-DIAG histogram count of grouped reassign jobs; it is emitted in the full DIAG blocks but not consistently emitted for every checkpoint, so it should not be used as the primary cross-batch comparison metric.

| Batch | Layer | Log segment | First split/reassign | Final split | Final reassign | Reassign / split | Split avg (ms) | Split max (ms) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | lines `606-1528` | `0 / 0` | `566` | `68,168` | `120.44` | `236.2` | `3,790.3` |
| 1 | 1 | lines `606-1528` | `0 / 0` | `0` | `1,600` | N/A | `0.0` | `0.0` |
| 2 | 0 | lines `1529-2613` | `0 / 0` | `8,878` | `701,471` | `79.01` | `119.8` | `3,420.4` |
| 2 | 1 | lines `1529-2613` | `0 / 0` | `0` | `0` | N/A | `0.0` | `0.0` |
| 3 | 0 | lines `2614-3919` | `0 / 0` | `82,888` | `4,108,434` | `49.57` | `83.2` | `6,587.5` |
| 3 | 1 | lines `2614-3919` | `0 / 0` | `0` | `0` | N/A | `0.0` | `0.0` |
| 4 | 0 | lines `3920-5194` | `0 / 0` | `260,914` | `8,878,868` | `34.03` | `64.9` | `9,820.7` |
| 4 | 1 | lines `3920-5194` | `0 / 0` | `1` | `68` | `68.00` | `181.0` | `181.0` |
| 5 | 0 | lines `5195-6355` | `0 / 0` | `402,015` | `11,851,068` | `29.48` | `59.9` | `7,585.4` |
| 5 | 1 | lines `5195-6355` | `0 / 0` | `27` | `2,306` | `85.41` | `92.9` | `350.1` |
| 6 | 0 | lines `6356-7451` | `0 / 0` | `498,514` | `13,977,020` | `28.04` | `63.3` | `8,696.4` |
| 6 | 1 | lines `6356-7451` | `0 / 0` | `57` | `7,106` | `124.67` | `116.9` | `500.5` |

Layer0 split/reassign growth across completed batches:

| Comparison | Split growth | Reassign growth |
| --- | ---: | ---: |
| Batch 1 -> Batch 2 | `15.69x` | `10.29x` |
| Batch 2 -> Batch 3 | `9.34x` | `5.86x` |
| Batch 3 -> Batch 4 | `3.15x` | `2.16x` |
| Batch 4 -> Batch 5 | `1.54x` | `1.33x` |
| Batch 5 -> Batch 6 | `1.24x` | `1.18x` |
| Batch 1 -> Batch 6 | `880.77x` | `205.04x` |

Layer1 note: layer1 had `reassign=1600` in batch 1, then stayed quiet in batches 2-3. It starts showing pressure in batch 4 (`1` split, `68` reassign), grows in batch 5 (`27` splits, `2,306` reassign), and grows again in batch 6 (`57` splits, `7,106` reassign). The maintenance explosion is still dominated by layer0, but upward propagation is now a repeated trend.

Interpretation: the absolute reassign-per-split ratio decreases from batch 1 to batch 6 in layer0, but the split count grows so aggressively that total reassign submissions still increase from `68,168` to `13,977,020`. The clearest per-batch evidence is split growth and total reassign growth.

### 4.2 Initial Load Baseline

After initial load, layer0 did not split:

| DIAG item | Value |
| --- | ---: |
| layer0 split | `0` |
| layer0 merge | `41` |
| layer0 reassign | `332` |
| layer0 append | `1313` |
| `AppendGetUs` avg | `2300.33 us` |
| `AppendPutUs` avg | `2581.51 us` |
| `AppendPostBytes` avg | `4851.11 B` |
| `AppendOutcome triggeredSplit` | `0` |
| `BatchSplitPostingVectors` | `0` |
| `MultiGetPageBuffer` | `1973 waits`, `3884.36 us avg`, `50.59 avgBatch` |

This phase had some append IO but no split cascade.

### 4.3 Batch 3-6 Complete Layer0 DIAG

Batches 3-6 emit full layer0 DIAG blocks, which show both the posting-full trigger and the reassign fanout getting worse:

| Batch | Append count | AppendGet avg | AppendPut avg | AppendPostBytes avg | SplitLockWait avg | ReassignJobUs count / avg | Barrier wait |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | `2,432` | `764.32 us` | `991.24 us` | `19,723.48 B` | `32.20 us` | `296 / 82,247.46 us` | `0.301145 s` |
| 4 | `5,207` | `1204.60 us` | `1438.07 us` | `23,991.37 B` | `52.47 us` | `808 / 111,757.51 us` | `1.043934 s` |
| 5 | `1,602` | `1065.85 us` | `1315.21 us` | `24,007.99 B` | `71.87 us` | `64 / 96,055.75 us` | `2.990529 s` |
| 6 | `3,732` | `1547.00 us` | `1758.46 us` | `25,195.20 B` | `84.11 us` | `401 / 144,872.67 us` | `1.826470 s` |

Batch split fanout details:

| Batch | Fanout count | PostingVectors avg | NewHeads avg | ReassignVectors avg | ReassignRecords avg | ReassignTargetHeads avg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | `16,023` | `247.23` | `0.98` | `36.80` | `256.39` | `180.40` |
| 4 | `27,662` | `247.19` | `0.99` | `44.43` | `320.95` | `218.66` |
| 5 | `32,419` | `247.20` | `1.00` | `49.87` | `365.56` | `243.14` |
| 6 | `36,113` | `247.25` | `1.00` | `52.72` | `387.02` | `252.77` |

Interpretation: the expensive part is not append get/put latency itself. The expensive part is that postings keep splitting at the configured limit, and each split fans out reassign work to an increasing number of target heads.

### 4.4 Async / TiKV Path DIAG

Layer0 async DIAG across batches 3-6:

| Batch | `MultiGetPageBuffer` waits / avg / avgBatch | `AddIndexSingleKeyGet` waits / avg / avgBatch | `AddIndexSingleKeyPut` waits / avg / avgBatch | `MultiGetString` waits / avg |
| ---: | ---: | ---: | ---: | ---: |
| 3 | `2,404,753 / 1239.84 us / 50.53` | `1,030,540 / 651.30 us / 10.18` | `1,030,540 / 1048.96 us / 10.18` | `46 / 992.67 us` |
| 4 | `3,717,001 / 2094.73 us / 50.55` | `1,053,993 / 1171.80 us / 13.08` | `1,053,993 / 1629.35 us / 13.08` | `39 / 970.67 us` |
| 5 | `4,439,836 / 2491.87 us / 50.57` | `1,063,577 / 1498.07 us / 14.69` | `1,063,577 / 1969.97 us / 14.69` | `68 / 770.26 us` |
| 6 | `5,024,066 / 2980.45 us / 50.61` | `1,071,199 / 1912.99 us / 15.77` | `1,071,199 / 2394.17 us / 15.77` | `65 / 760.55 us` |

Interpretation:

- Single-key get/put waits rise materially by batch 6, which means TiKV is paying for the amplified maintenance workload.
- The request volume and waits rise together with split/reassign fanout, so TiKV looks like the cost sink rather than the root amplifier.
- `MultiGetString` remains tiny compared with page-buffer and single-key get/put waits, so version-cache miss reads are not the main bottleneck.
- All `[DIAG-MC]` counters are zero, confirming the run is not using the multi-chunk posting path.

## 5. TiKV / PD Evidence

TiKV raw gRPC metrics are cumulative since TiKV start, so they are not exact per-batch rates. They are still useful for checking whether TiKV latency is obviously unhealthy.

| TiKV status port | raw_get count | raw_get avg (ms) | raw_put count | raw_put avg (ms) | raw_batch_get count | raw_batch_get avg (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20181 | 35,602,019 | 0.333 | 35,736,024 | 0.836 | 327,064,809 | 0.310 |
| 20182 | 27,117,083 | 0.419 | 36,113,121 | 0.866 | 330,867,817 | 0.335 |
| 20183 | 30,526,325 | 0.508 | 34,479,780 | 1.034 | 346,596,939 | 0.445 |

PD store distribution at collection time:

| Store ID | Address | State | Leaders | Regions | slow_score | region_size |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `127.0.0.1:20162` | Up | 47 | 47 | 1 | not sampled |
| 4 | `127.0.0.1:20163` | Up | 55 | 55 | 1 | not sampled |
| 5 | `127.0.0.1:20161` | Up | 44 | 44 | 1 | not sampled |

Pressure/error signals:

| Signal | Value |
| --- | ---: |
| `tikv_engine_write_stall` | `0` on all nodes |
| `tikv_engine_write_stall_reason` | `0` on all nodes |
| `tikv_scheduler_pending_compaction_bytes` | `0` on all nodes |
| `region_error` log lines | `35` |
| gRPC/final failure log lines | `0` |
| `Split: new head VID ... already exists` lines | batch counts: `16`, `204`, `392`, `351`, `241`, `174` for batches 1-6 |

Interpretation: TiKV is doing much more work and cumulative raw operation averages are higher than in the earlier three-batch read. However, there is still no visible write-stall, compaction, or retry-failure signal. The evidence is consistent with TiKV being loaded by amplified SPFresh maintenance traffic rather than TiKV independently stalling.

## 6. Root Cause Assessment

The evidence chain is:

1. Runtime posting size limit is `246` vectors.
2. Batches 3-6 have `BatchSplitPostingVectors avg` around `247.2`, almost exactly the limit.
3. Batch6 split count reaches `498,514` and reassign submissions reach `13,977,020`.
4. `BatchSplitReassignTargetHeads avg` rises from `180.40` in batch 3 to `252.77` in batch 6, so each split fans out to more heads over time.
5. TiKV single get/put waits rise by batch 6, but TiKV stall/compaction signals are still zero.
6. Search after the batch remains stable enough, while insertion throughput drops sharply.

Therefore, the likely bottleneck is SPFresh layer0 maintenance amplification: frequent split at the posting limit followed by high-fanout reassign. TiKV is a participant in the cost because every split/reassign causes many single-key RMWs, but the root amplifier is the application-level split/reassign policy and posting threshold.

### 6.1 Posting-Full / Split / Merge-Existing-Head Hypothesis

The current data supports most of this hypothesis, but one part still needs the new checkpoint instrumentation to become conclusive.

Hypothesis: as batches are inserted, many postings approach the posting limit. Appends then trigger splits. Split produces high-fanout reassign work. Because the split path often finds that the selected new head already exists in the next-layer head index, it merges into the existing posting instead of growing the head/posting universe proportionally. This can keep visible head counts relatively flat while split/reassign work explodes.

Evidence already present in this run:

| Claim | Evidence | Strength |
| --- | --- | --- |
| Postings are splitting near full | Runtime posting size limit is `246`; batch3-6 `BatchSplitPostingVectors avg` is about `247.2` | Strong |
| Splits explode with batch number | Layer0 split count grows `566 -> 8,878 -> 82,888 -> 260,914 -> 402,015 -> 498,514` across completed batches | Strong |
| Split creates large reassign fanout | Batch6 `BatchSplitReassignRecords avg=387.02`, `BatchSplitReassignTargetHeads avg=252.77` | Strong |
| Head/posting growth is not proportional to split count | Top `Save Vector` grows only `40,428 -> 40,482` by batch6 while layer0 split reaches `498,514` in batch6 | Moderate |
| Split often targets an already-existing head | `Split: new head VID ... already exists in head index. Do merging...` appears frequently | Strong as a path signal, not yet a complete rate |

Existing-head merge log count by segment at latest read:

| Segment | Existing-head merge log lines |
| --- | ---: |
| Batch 1 | `16` |
| Batch 2 | `204` |
| Batch 3 | `392` |
| Batch 4 | `351` |
| Batch 5 | `241` |
| Batch 6 | `174` |

Code-path evidence:

- In split, if `m_headIndex->ContainSample(newHeadVID, m_layer + 1)` is true, the code logs `already exists in head index. Do merging...`, reads the existing posting, deduplicates/merges the new split posting into it, writes it back, and can call `SplitAsync(newHeadVID, currentLength)` again when the merged posting is still above `m_postingSizeLimit`.
- If the head does not exist, only then does the split path call `AddHeadIndex(..., m_layer + 1, ...)`, which is the path that grows the next-layer head mapping.
- During search, `AsyncMergeInSearch=true` can also enqueue `MergeAsync(curPostingID)` when a posting's real count is at or below `m_mergeThreshold`, which further works against visible posting/head growth.

Important caveat: the old log does not print `GetNumSamples(layer)` or `GetHeadIndexMapping(layer).size()` at checkpoint time, so it cannot conclusively prove the exact per-layer dynamic head/posting growth. Code was updated after this analysis to print checkpoint layer stats, and additional DIAG counters were added to log the split head path directly:

```text
Checkpoint layer stats: batch X/Y layer=N samples=... deleted=... headMapping=...
[DIAG] layer N SplitHeadPath sameHead=... existingHeadMerge=... existingHeadResplit=... newHead=... (batch)
```

The next run should use these two lines to test the hypothesis directly. The expected confirmation pattern is: high `BatchSplitPostingVectors` near `246`, high `existingHeadMerge` and possibly non-zero `existingHeadResplit`, high reassign target-head fanout, but relatively slow growth in `headMapping` compared with split count.

## 7. Recommended Next Checks / Experiments

1. Run an A/B test with larger `PostingPageLimit`, for example `16` or `32`, keeping other settings fixed. If this diagnosis is correct, split count and reassign count should fall sharply.
2. Run a second A/B with lower `ReassignK` than `64`. The most suspicious DIAG value is `BatchSplitReassignTargetHeads avg=252.77` by batch 6; reducing reassign fanout should directly reduce single-key RMW count.
3. Capture TiKV counter deltas over a fixed window instead of cumulative counters. The current TiKV numbers are healthy, but delta sampling would quantify write/read QPS during the slow part.
4. Emit full layer0 and layer1 DIAG after every batch. The later layer1 activity is still small, but batch 4-6 show it is no longer zero and is increasing.
5. Use the newly added checkpoint layer stats and `SplitHeadPath` DIAG counters in the next run to compare head mapping growth against split/new-head/existing-head-merge counts.

## 8. Bottom Line

Search quality and post-batch latency are still acceptable after six batches, though recall has dipped to `0.973`. Version cache is working. TiKV is busier and raw operation averages have risen, but TiKV still looks healthy from stall/compaction/store-state signals. The insert slowdown is best explained by layer0 split/reassign amplification caused by postings hitting the `246` vector limit and then fanning out reassign work to many target heads. Batch 4-6 add one new warning: layer1 is now participating and growing, so the same maintenance pressure may propagate upward in later batches.