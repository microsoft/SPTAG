# SPFresh SIFT1B v10 Metric Report

Date: 2026-05-07

Scope: analysis of the live benchmark run `benchmark_20260507_032733.log` and `evaluation/2026-04-23/output.json`. At collection time the process was still running (`PID 3779889`), and `output.json` contained three completed batches. Batch 4 observations are treated as live trend only, not completed-batch results.

## 1. Executive Summary

The first three completed batches show stable search quality and latency, but insert throughput drops materially after batch 2. The strongest evidence points to layer0 split/reassign amplification, not TiKV single-operation latency.

Key conclusions:

- Insert throughput drops from `579.4 vec/s` in batch 1 to `401.3 vec/s` in batch 3, a `30.75%` drop.
- Post-batch search is stable: mean latency moves only from `7.06 ms` to `7.58 ms`, and recall remains `0.977-0.982`.
- During-insert search latency worsens: mean latency grows from `7.14 ms` to `10.75 ms`, a `50.65%` increase. This indicates insert maintenance work is contending with search.
- TiKV raw operation averages are low: raw get `0.159-0.170 ms`, raw put `0.466-0.576 ms`, raw batch get `0.169-0.187 ms` across the three TiKV nodes.
- TiKV pressure signals are clean: write stall is `0`, pending compaction is `0`, stores are `Up`, and `slow_score=1`.
- DIAG shows layer0 split/reassign explosion by batch 3: `82,888` splits and `4,108,434` reassign submissions, with split max latency `6.59 s`.
- `BatchSplitPostingVectors avg=247.23`, while runtime posting size limit is `246`, so postings are splitting almost exactly at the configured page limit.

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

Source: `evaluation/2026-04-23/output.json`, completed batch entries 1-3.

| Batch | Loaded vectors | Insert time (s) | Insert throughput (vec/s) | During-insert mean (ms) | During p95 (ms) | During p99 (ms) | Post-search mean (ms) | Post p95 (ms) | Post p99 (ms) | Post QPS | Recall@5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,000,000 | 1,725.9071 | 579.4055 | 7.1368 | 8.5670 | 35.2620 | 7.0611 | 8.4010 | 8.7490 | 562.4550 | 0.9770 |
| 2 | 2,000,000 | 1,793.2889 | 557.6346 | 8.3366 | 9.9530 | 39.0620 | 7.1031 | 9.0230 | 9.6070 | 560.4235 | 0.9820 |
| 3 | 3,000,000 | 2,492.2107 | 401.2502 | 10.7519 | 15.0000 | 38.9840 | 7.5841 | 9.2200 | 9.6530 | 525.6794 | 0.9820 |

Derived changes:

| Metric | Batch 1 -> Batch 3 |
| --- | ---: |
| Insert throughput | `-30.75%` |
| Insert time | `+44.40%` |
| During-insert mean latency | `+50.65%` |
| Post-search mean latency | `+7.41%` |
| Post-search QPS | `-6.54%` |

Interpretation: search after each batch remains healthy; insertion is the part that deteriorates.

## 3.1 Vector / Head Counts After Each Completed Batch

There are three different count concepts in the logs and checkpoint files:

- `SaveIndex(... GetNumSamples())` / `Current Vector Num` is the layer0 data-vector universe exposed by `Index::GetNumSamples()`, i.e. total inserted data vectors.
- `Save Vector (40428,128)` is the saved in-memory top `HeadIndex` vector count.
- `SPTAGHeadVectorIDs.bin_layer0` / `SPTAGHeadVectors.bin_layer0` represent the layer0 head-vector set used as the input universe for the next layer. From file size, this is `200,928` rows in the available saved snapshots.

Directly observed checkpoint counts:

| Completed batch | Checkpoint path | Layer0 data vectors (`GetNumSamples`) | Deleted | Top `HeadIndex` vectors (`Save Vector`) | Layer0 head-vector file rows |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `spann_index_0` | `2,000,000` | `0` | `40,428` | not available now; copied from base |
| 2 | `spann_index_1` | `3,000,000` | `0` | `40,428` | not available now; copied from previous checkpoint |
| 3 | `spann_index_2` | `4,000,000` | `0` | `40,428` | `200,928` |

Available saved snapshot file-size checks:

| Snapshot | `SPTAGHeadVectorIDs.bin_layer0` rows | `SPTAGHeadVectors.bin_layer0` rows | `SPTAGHeadVectorIDs.bin` rows |
| --- | ---: | ---: | ---: |
| `spann_index` | `200,928` | `200,928` | `40,428` |
| `spann_index_2` | `200,928` | `200,928` | `40,428` |
| `spann_index_3` | `200,928` | `200,928` | `40,428` |

Important limitation: the current log does not directly print dynamic `GetNumSamples(1)` or `GetHeadIndexMapping(1).size()` after each batch. Therefore, if "layer-1 vector count" means the dynamic layer1 extra-searcher contained-vector/sample count after split-time `AddHeadIndex(..., tolayer=1)`, that exact per-batch value cannot be reliably reconstructed from the existing logs. The next run should log `GetNumSamples(layer)` and contained head count for every layer at checkpoint time.

## 4. DIAG Evidence

### 4.1 Per-Batch Split / Reassign Counts

Counter semantics check: the per-batch values below were extracted by segmenting the log from `========== BATCH n/10` to `Checkpoint saved: batch n/10`, then taking the last `total_submitted` line for each layer inside that segment. In all three completed segments, the first layer0 and layer1 `total_submitted` lines start from `split=0` and `reassign=0`, so these are per-batch counters, not global cumulative counters.

Important metric granularity note: `total_submitted reassign` is the comparable per-batch reassign submission/record counter available in every batch progress block. `ReassignJobUs count` is a full-DIAG histogram count of grouped reassign jobs; it is emitted for batch 3 but not consistently emitted for batch 1/2 checkpoints, so it should not be used for the three-batch comparison.

| Batch | Layer | Log segment | First split/reassign | Final split | Final reassign | Reassign / split | Split avg (ms) | Split max (ms) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | lines `606-1528` | `0 / 0` | `566` | `68,168` | `120.44` | `236.2` | `3,790.3` |
| 1 | 1 | lines `606-1528` | `0 / 0` | `0` | `1,600` | N/A | `0.0` | `0.0` |
| 2 | 0 | lines `1529-2613` | `0 / 0` | `8,878` | `701,471` | `79.01` | `119.8` | `3,420.4` |
| 2 | 1 | lines `1529-2613` | `0 / 0` | `0` | `0` | N/A | `0.0` | `0.0` |
| 3 | 0 | lines `2614-3919` | `0 / 0` | `82,888` | `4,108,434` | `49.57` | `83.2` | `6,587.5` |
| 3 | 1 | lines `2614-3919` | `0 / 0` | `0` | `0` | N/A | `0.0` | `0.0` |

Layer0 split/reassign growth across completed batches:

| Comparison | Split growth | Reassign growth |
| --- | ---: | ---: |
| Batch 1 -> Batch 2 | `15.69x` | `10.29x` |
| Batch 2 -> Batch 3 | `9.34x` | `5.86x` |
| Batch 1 -> Batch 3 | `146.45x` | `60.27x` |

Layer1 note: layer1 had `reassign=1600` in batch 1, then `0` split and `0` reassign in batch 2 and batch 3. There were no layer1 splits in the completed three batches. The insert maintenance explosion is therefore concentrated in layer0.

Interpretation: the absolute reassign-per-split ratio decreases from batch 1 to batch 3, but the split count grows so aggressively that total reassign submissions still increase from `68,168` to `4,108,434`. The clearest per-batch evidence is split growth and total reassign growth.

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

### 4.3 Batch 2 End Progress

The log does not print the full layer0 DIAG block at batch 2 checkpoint, but the final progress line before checkpoint shows:

| Batch 2 layer0 progress item | Value |
| --- | ---: |
| split | `8,878` |
| reassign | `701,471` |
| append | `1,194` |
| split avg latency | `119.8 ms` |
| split max latency | `3,420.4 ms` |
| barrier wait | `0.000001 s` |

### 4.4 Batch 3 Complete DIAG

Batch 3 emits the full layer0 DIAG block:

| DIAG item | Value |
| --- | ---: |
| layer0 split | `82,888` |
| layer0 reassign | `4,108,434` |
| layer0 append | `2,432` |
| split avg latency | `83.2 ms` |
| split max latency | `6,587.5 ms` |
| `AppendLockWait` avg | `1553.38 us` |
| `AppendGetUs` avg | `764.32 us` |
| `AppendPutUs` avg | `991.24 us` |
| `AppendPostBytes` avg | `19,723.48 B` |
| `SplitLockWait` count / avg | `82,888 / 32.20 us` |
| `ReassignJobUs` count / avg | `296 / 82,247.46 us` |
| `ReassignSrc fromSplitBatch` | `4,108,138` |
| `PoolQueueDepth` avg | `0.00 jobs` |
| `PoolRunning` avg | `1.43 workers` |
| `BatchBarrierWait` | `0.301145 s` |

Batch split fanout details:

| Split fanout item | Count | Average |
| --- | ---: | ---: |
| `BatchSplitPostingVectors` | `16,023` | `247.23 vectors` |
| `BatchSplitNewHeads` | `16,023` | `0.98 heads` |
| `BatchSplitReassignVectors` | `16,023` | `36.80 vectors` |
| `BatchSplitReassignRecords` | `16,023` | `256.39 records` |
| `BatchSplitReassignTargetHeads` | `16,023` | `180.40 heads` |

Important ratios:

| Ratio | Value |
| --- | ---: |
| Batch3 reassign / split | `49.57` |
| Batch3 split vs batch2 split | `9.34x` |
| Batch3 reassign vs batch2 reassign | `5.86x` |

Interpretation: the expensive part is not append get/put latency itself. The expensive part is that split generates a very large amount of reassign work, and each split touches many target heads.

### 4.5 Async / TiKV Path DIAG

Batch 3 async DIAG:

| Async DIAG item | Waits | Avg latency | Avg batch |
| --- | ---: | ---: | ---: |
| `MultiGetPageBuffer` | `2,404,753` | `1239.84 us` | `50.53` |
| `MultiGetString` | `46` | `992.67 us` | `1.00` |
| `AddIndexSingleKeyGet` | `1,030,540` | `651.30 us` | `10.18` |
| `AddIndexSingleKeyPut` | `1,030,540` | `1048.96 us` | `10.18` |

Interpretation:

- Single-key get/put waits are around `0.65-1.05 ms`, which is not high enough to explain the throughput collapse by itself.
- `MultiGetString waits=46` is the version-cache miss path. Only 46 waits means version cache is working; version-map miss is not the bottleneck.
- All `[DIAG-MC]` counters are zero, confirming the run is not using the multi-chunk posting path.

## 5. TiKV / PD Evidence

TiKV raw gRPC metrics are cumulative since TiKV start, so they are not exact per-batch rates. They are still useful for checking whether TiKV latency is obviously unhealthy.

| TiKV status port | raw_get count | raw_get avg (ms) | raw_put count | raw_put avg (ms) | raw_batch_get count | raw_batch_get avg (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20181 | 11,818,819 | 0.170 | 11,938,123 | 0.576 | 100,713,295 | 0.178 |
| 20182 | 10,617,756 | 0.164 | 13,429,524 | 0.521 | 97,512,889 | 0.169 |
| 20183 | 12,075,564 | 0.159 | 16,002,220 | 0.466 | 91,922,612 | 0.187 |

PD store distribution at collection time:

| Store ID | Address | State | Leaders | Regions | slow_score | region_size |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4 | `127.0.0.1:20163` | Up | 56 | 56 | 1 | 2390 |
| 5 | `127.0.0.1:20161` | Up | 41 | 41 | 1 | 2173 |
| 1 | `127.0.0.1:20162` | Up | 46 | 46 | 1 | 3895 |

Pressure/error signals:

| Signal | Value |
| --- | ---: |
| `tikv_engine_write_stall` | `0` on all nodes |
| `tikv_engine_write_stall_reason` | `0` on all nodes |
| `tikv_scheduler_pending_compaction_bytes` | `0` on all nodes |
| `region_error` log lines | `35` |
| gRPC/final failure log lines | `0` |
| `Split: new head VID ... already exists` lines | `956` at latest read, including live batch 4 |

Interpretation: TiKV is doing work, but there is no visible write-stall, compaction, or retry-failure signal. Region errors are recoverable and do not dominate the run.

## 6. Root Cause Assessment

The evidence chain is:

1. Runtime posting size limit is `246` vectors.
2. Batch3 `BatchSplitPostingVectors avg=247.23`, almost exactly the limit.
3. Batch3 split count reaches `82,888` and reassign submissions reach `4,108,434`.
4. Batch3 `BatchSplitReassignTargetHeads avg=180.40`, so each split fans out to many heads.
5. TiKV single get/put waits stay near `0.65-1.05 ms`, and TiKV stall/compaction signals are zero.
6. Search after the batch remains stable, while insertion throughput drops sharply.

Therefore, the likely bottleneck is SPFresh layer0 maintenance amplification: frequent split at the posting limit followed by high-fanout reassign. TiKV is a participant in the cost because every split/reassign causes many single-key RMWs, but the root amplifier is the application-level split/reassign policy and posting threshold.

### 6.1 Posting-Full / Split / Merge-Existing-Head Hypothesis

The current data supports most of this hypothesis, but one part still needs the new checkpoint instrumentation to become conclusive.

Hypothesis: as batches are inserted, many postings approach the posting limit. Appends then trigger splits. Split produces high-fanout reassign work. Because the split path often finds that the selected new head already exists in the next-layer head index, it merges into the existing posting instead of growing the head/posting universe proportionally. This can keep visible head counts relatively flat while split/reassign work explodes.

Evidence already present in this run:

| Claim | Evidence | Strength |
| --- | --- | --- |
| Postings are splitting near full | Runtime posting size limit is `246`; batch3 `BatchSplitPostingVectors avg=247.23` | Strong |
| Splits explode with batch number | Layer0 split count grows `566 -> 8,878 -> 82,888` across completed batches | Strong |
| Split creates large reassign fanout | Batch3 `BatchSplitReassignRecords avg=256.39`, `BatchSplitReassignTargetHeads avg=180.40` | Strong |
| Head/posting growth is not proportional to split count | Top `Save Vector` remains `40,428` after batches 1-3 while layer0 split reaches `82,888` in batch3 | Moderate |
| Split often targets an already-existing head | `Split: new head VID ... already exists in head index. Do merging...` appears frequently | Strong as a path signal, not yet a complete rate |

Existing-head merge log count by segment at latest read:

| Segment | Existing-head merge log lines |
| --- | ---: |
| Batch 1 | `16` |
| Batch 2 | `204` |
| Batch 3 | `392` |
| Batch 4 live | `345` |

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

## 7. Live Batch 4 Note

Batch 4 was running during collection. It should not be mixed with completed-batch metrics, but it reinforces the same trend. By `26%` progress in batch 4, layer0 had already reached:

| Batch4 live progress | Value |
| --- | ---: |
| split | `45,562` |
| reassign | `1,789,450` |
| append | `4,420` |
| split avg latency | `74.4 ms` |
| split max latency | `9,820.7 ms` |

This suggests split/reassign pressure continues and may worsen in later batches.

## 8. Recommended Next Checks / Experiments

1. Run an A/B test with larger `PostingPageLimit`, for example `16` or `32`, keeping other settings fixed. If this diagnosis is correct, split count and reassign count should fall sharply.
2. Run a second A/B with lower `ReassignK` than `64`. The most suspicious DIAG value is `BatchSplitReassignTargetHeads avg=180.40`; reducing reassign fanout should directly reduce single-key RMW count.
3. Capture TiKV counter deltas over a fixed window instead of cumulative counters. The current TiKV numbers are healthy, but delta sampling would quantify write/read QPS during the slow part.
4. Emit full layer0 DIAG after every batch, not only batch3. Batch2 has progress counters but not the full split fanout block, which makes cross-batch comparison less precise.
5. Use the newly added checkpoint layer stats and `SplitHeadPath` DIAG counters in the next run to compare head mapping growth against split/new-head/existing-head-merge counts.

## 9. Bottom Line

Search quality and post-batch latency are fine. Version cache is working. TiKV looks healthy. The insert slowdown is best explained by layer0 split/reassign amplification caused by postings hitting the `246` vector limit and then fanning out reassign work to many target heads.