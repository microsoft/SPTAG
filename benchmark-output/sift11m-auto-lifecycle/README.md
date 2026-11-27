# SIFT 9M -> 11M -> 9M automatic layer lifecycle

## Status

### Current policy: existing ordinary merge only

### 2026-11-26: unlimited-duration 5M result -- shrinking not accepted

Run `sift-20261125-5m-ordinary32-unlimited` used fresh full-size growth
calibration, `MergeThreshold=32`, `DisableReassign=false`, and
`--phase-timeout-seconds 0`. The formal workload was
4.5M -> 5.5M -> 4.5M, deleting 20 fixed-seed random batches of 50K users.
No manual merge or resize was used. Search budgets and Recall@5 >=0.95
acceptance were unchanged.

The frozen head-memory thresholds were high **269779125**, target
**264383542**, and low **253592377** bytes.

| Formal checkpoint | Live users | Layers | Routing heads | Recall@5 round 1 / 2 |
| --- | ---: | ---: | ---: | --- |
| Initial, before query maintenance | 4500000 | 1 | 905124 | 0.96726 / 0.96670 |
| Insert batch 10 | 5000000 | 2 | 779104 | 0.96618 / 0.96614 |
| Insert batch 20 | 5500000 | 2 | 783877 | 0.96576 / 0.96582 |
| Delete batch 10 | 5000000 | 2 | 782219 | 0.96464 / 0.96480 |
| Delete batch 20 | 4500000 | 2 | 779134 | 0.96430 / 0.96424 |
| Final convergence sweep 112 | 4500000 | 2 | 775071 | Not separately evaluated |

Each recall checkpoint used two rounds of 10000 native queries, Top5, with
zero deleted-result hits. All deletion batches completed, but the final
15-minute convergence window still ended with two layers. The test exited
**201** with:

```text
Random deletion to the final count did not trigger shrink with the frozen memory thresholds
```

The formal phase ran about **19h 47m 35s**. This was neither an eight-hour
wall-clock stop nor a resource-guard stop: the monitored minimum free disk
was about **30.13 GiB** and peak sampled RSS about **6.46 GiB**. Routing heads
fell by only **8806 (1.12%)** from the insertion peak to the last convergence
snapshot. Longer execution completed the deletions, but did not establish
automatic shrinking under this workload and frozen configuration.

**Accepted observations:** automatic growth 1 -> 2, completed random deletion,
and recall above 0.95. **Not accepted:** shrinking 2 -> 1 and final one-layer
reload; the reload block was not reached. Disabling the outer phase deadline
does not disable internal maintenance/resize/convergence deadlines.

The retained local run directory contains `metrics.jsonl`, `status.json`,
`formal-resources.csv`, `formal.log`, calibration/configuration snapshots and
the failed formal index. These generated files are not committed. The
runner's unlimited/explicit deadline behavior and retained disk guard are
covered by the 13 passing focused Python tests.

### Running the ordinary-merge lifecycle

To run the 5M lifecycle without the default eight-hour phase deadline:

```bash
python3 run_experiment.py --run-id UNIQUE_5M_RUN_ID --pivot-count 5000000 \
  --merge-threshold 32 --growth-calibration --discard-growth-index \
  --phase-timeout-seconds 0
```

`--phase-timeout-seconds` overrides each calibration/formal phase's wall-clock
limit; zero means unlimited. Omitting it preserves the existing defaults.
Disk/RSS guards and the driver's maintenance, resize and convergence deadlines
remain active. This starts fresh, not from an interrupted deletion checkpoint.

For a bounded real-SIFT mechanism check, use:

```bash
python3 run_experiment.py --run-id UNIQUE_SMALL_RUN_ID --pivot-count 16384 \
  --merge-threshold 32 --growth-calibration --discard-growth-index
```

This workload is 4096 -> 32768 -> 1024 users, with fixed-seed random deletion,
all 10000 native queries, Top5, unchanged search budgets and reassignment.
It is a strong-deletion-pressure test, not evidence for the matched 5M lifecycle.
Calibration and formal phases are each capped at ten minutes. Small-only
resource overrides are recorded: 1 GiB file growth, one-second resize cooldown,
60-second resize/maintenance waits, three-minute convergence, 24 GiB admission
and 20 GiB disk reserve. Full-size resource guards are unchanged.

The dedicated routing-compaction strategy described in the historical records
below is removed entirely, including its executor, cursor, journal recovery,
compatibility parameters and dedicated tests. Old dedicated-compaction indexes
are not supported and must not be resumed. New runs use only the existing
`MergeAsync` / `MergePostings` / reassignment path, with no manual resize/merge
calls or runtime threshold changes.

`--merge-threshold {10,24,32,48}` freezes the existing setting before build.
Ten is the archived baseline; higher values are explicit, audited deviations,
not a new index policy. Keep all other non-dynamic settings matched, including
`DisableReassign=false`, and require Recall@5 >=0.95 unchanged.

First test a real 450K -> 550K -> 450K lifecycle with fresh same-scale growth
calibration and final reload:

```bash
python3 run_experiment.py --run-id sift-ordinary-merge32-pilot \
  --pivot-count 5000000 --merge-threshold 32 \
  --growth-calibration --merge-pilot-only --discard-growth-index
```

The candidate threshold is experimental, not accepted on synthetic tests alone.
For thresholds above ten, growth calibration evaluates the same initial and
insertion checkpoints so ordinary query-driven maintenance is not omitted
from the measured growth trajectory.
If routing counts do not fall enough, report failure instead of forcing shrink.
Compare candidates with separate frozen runs; do not silently select a higher
threshold or change recall/search budgets. A successful small run does not
guarantee the 5M outcome. Full-size calibration must be repeated at 5M.

New runs monitor free space every second and stop below 30 GiB. The small
workflow requires 50 GiB at admission; full-size 5M requires 80 GiB.
The discard flag removes only this run's successful calibration index and
preserves its configuration and metrics. It does not authorize deletion of
previous experimental indexes or failed runs.

When validating a separate build during an active experiment, set
`SPTAG_TEST_BINARY` to that build's test executable for configuration tests;
leave the active experiment's executable unchanged.

### Historical experiments (not the current recipe)

The unmatched runs have been stopped. Do not resume their incomplete indexes.
The next experiment must recalibrate using the matched profile below; no new
large experiment is started automatically by editing this recipe.

The matched retry subsequently reached 11M with Recall@5 about 0.958 at all
three insertion checkpoints, but never crossed the pilot-derived high
watermark: 537941262 bytes at 11M versus a 546576240-byte threshold. It therefore
failed the two-layer assertion and did not test deletion/shrinking. Absolute
head memory at 1/10 scale is not sufficiently accurate for this narrow trigger.
The next calibration uses the observed full-scale growth trajectory instead.

## Matched configuration

The driver loads **every** parameter in `[Base]`, `[SelectHead]`, `[BuildHead]` and
`[BuildSSDIndex]` from the historical
[effective configuration](../../evaluation/2026-11-09-sift10m-fileio-layer1-matched/saved-indexloader.ini).
It does not rely on today's defaults or copy only a few selected settings.
The recorded file is read-only and a snapshot/hash is kept with each new run.

| Parameter | Matched value |
| --- | --- |
| SelectHeadType / Ratio | BKT / 0.2 |
| ReplicaCount | 8 |
| InternalResultNum / SearchInternalResultNum | 64 / 64 |
| MaxCheck (head and SSD) | 8192 |
| NeighborhoodSize / RefineIterations | 32 / 2 |
| AddCountForRebuild | 10000 |
| PostingVectorLimit | 118 |
| PostingPageLimit / SearchPostingPageLimit / BufferLength | 4 / 4 / 4 |
| ReassignK / MergeThreshold | 64 / 10 |
| LatencyLimit / HashTableExponent | 50 / 4 |
| IOThreadsPerHandler | 4 |
| Build threads (selection/head/postings) | 80 / 80 / 80 |
| AppendThreadNum / ReassignThreadNum | 48 / 0 |
| Search threads / insertion threads | 32 / 16 |
| Concurrent searches during insertion | 0 |
| Evaluation queries / TopK | 10000 / 5 |

Only `Base.VectorPath`, `Base.VectorSize` and `Base.IndexDirectory` are
replaced to accommodate in-memory input, the new initial count and the current
output location. All other `Base` fields, including UInt8, BKT and L2, are loaded
unchanged. Input is supplied directly to the public build/insert APIs.
Each of the 16 insertion workers sends one vector per API call, as in the
historical harness. Initial and inserted global/source IDs coincide
only before concurrent insertion; a persisted source-by-VID mapping tracks the
16 insertion workers and is used by exact truth and distance verification.

The explicit index-setting differences are limited to automatic resizing:

- `EnableDynamicLayers=true`, `EnableDynamicLayerShrink=true`.
- Low/target/high head-memory thresholds are recalibrated with this profile and
  frozen before the formal build. Unmatched pilot thresholds are rejected.
- `MaxLayers=2`, cooldown 60 seconds and resize timeout 600 seconds.
- `AsyncMergeInSearch=true` rather than false, so ordinary public queries can
  retire unnecessary routing postings after user deletion. Normal evaluation
  and deletion-phase searches can therefore perform maintenance; this is not a
  read-only performance comparison with the historical benchmark.
- Original `MaxID=1000000000`, initial `Layers=1`, storage and other settings
  remain unchanged.

## Workload and acceptance

1. Prepare the first 11M real SIFT1B vectors (128D UInt8) and the 10000 public
   SIFT queries. Their first 10M prefix and query SHA-256 hashes were checked
   against the historical experiment and match exactly. `prepare_data.py`
   downloads only a 1.408 GB prefix, not the full 128 GB dataset.
2. Calibrate at 1/10 scale: build 900K and insert twenty 10K batches to 1.1M.
   No manual resizing or runtime threshold changes are allowed.
3. Before formal build, freeze `high=10*pilot head bytes at 1M`,
   `target=0.98*high`, `low=0.94*high`. This is a hypothesis, not a count-based
   switching guarantee.
4. Build 9M with one posting layer; insert twenty 100K batches to 11M, expecting
   automatic growth near 10M.
5. Shuffle all 11M IDs using `std::mt19937(1729)` and delete the first 2M in
   twenty 100K batches to 9M. Persist exact deleted IDs and use ordinary searches
   to provide merge opportunities. Expect automatic shrink below 10M; report
   failure instead of forcing it if random deletion does not retire enough
   routing postings.
6. At initial 9M, inserted 10M/11M and surviving 10M/9M, independently compute
   exact live-set Top5 boundaries for all 10000 queries. Run two measured query
   passes with 32 workers, each preceded by a warmup, and require tie-aware
   Recall@5 >=0.95. Check complete live-ID membership, no duplicate/deleted
   results, and correct independently computed L2 distances.
7. Save at key checkpoints and finally reload, recheck recall/membership/layers
   and index integrity. Exact truth over 10000 queries is substantially more
   expensive than the abandoned 64-query smoke-style evaluation.

User counts, batch sizes and random deletion intentionally follow the new
requested lifecycle rather than the historical 9.9M + ten 10K insertions with
no deletions. Retired layer files are not physically reclaimed by shrinking.

## Running and safety

Build `SPTAGTest` with FILEIO enabled, then:

```bash
python3 benchmark-output/sift11m-auto-lifecycle/test_profile.py
python3 benchmark-output/sift11m-auto-lifecycle/prepare_data.py
python3 benchmark-output/sift11m-auto-lifecycle/run_experiment.py --run-id UNIQUE_RUN_ID
```

Set the appropriate `LD_LIBRARY_PATH` for the installed Boost/TBB libraries.
`test_profile.py` executes configuration-only mode without building an index and
checks all archived effective parameters, permitting only the declared merge
override, plus the historical query/thread settings.

Each run records driver/runner/profile snapshots, the profile SHA-256, source
commit, input hashes, frozen thresholds, JSONL measurements, logs and process
RSS/peak RSS/free disk. It requires 35 GiB free to start, stops below 20 GiB or
above 64 GiB RSS, and caps calibration at two hours/formal execution at eight
hours. Maintenance waits are capped at twenty minutes and final merge
convergence at fifteen minutes (an in-progress operation can overrun a check).
These limits may stop the heavier matched configuration; they are not relaxed
automatically. On this shared host, restarting the matched 80-build/48-maintenance
thread workload must be explicit.

Run directories are never overwritten and failed writes are not replayed from
an older checkpoint. The unmatched initial attempt failed its 9M Recall@10
baseline (0.890625); the 512-candidate retry was stopped at user request. Neither
is evidence for the matched profile or the requested automatic lifecycle.

## Full-scale threshold calibration

The original pilot formula above is retained as a record of the failed
calibration, not as the recommended next formal configuration. With an intact,
completed 11M checkpoint from the matched retry:

```bash
python3 benchmark-output/sift11m-auto-lifecycle/calibrate_fullscale.py \
  --source-run sift-20261120-matched-disk-retry --run-id UNIQUE_CALIBRATION_ID
```

This **mutates that source index** from 11M to 9M using the same fixed-seed
random deletions and ordinary public searches. It preserves the pre-calibration
configuration and growth metrics. Fixed, unreachable growth thresholds and
disabled automatic shrinking are configured before loading; this is explicitly
a one-layer calibration, not evidence that dynamic resizing works. No manual
grow/shrink/merge API is called. It keeps the 20 GiB/64 GiB resource guards.

The high watermark is the measured Head memory at inserted 10M. The low
watermark is halfway between estimated compact Head memory after evaluation at
deleted 10M and the next batch at 9.9M, with target halfway between low and high.
The compact estimate is Head bytes times live routing count divided by allocated
Head slots, matching the shrink precheck. Final 9M must also fit below low.
If these measurements cannot satisfy `0 < low < target < high`, calibration
fails explicitly instead of forcing resizing or changing unrelated settings.
The one-layer trace only establishes a candidate interval; a fresh actual
two-layer lifecycle must still verify it.

The completed full-scale deletion calibration found **no admissible interval**
under the current estimator and unchanged matched settings:

| Measurement | Result |
| --- | --- |
| Growth high from inserted 10M | 534486978 bytes |
| Compact estimate after deleted 10M evaluation | 536568291 bytes |
| Compact estimate at deleted 9.9M | 536565321 bytes |
| Compact estimate at final 9M | 536385636 bytes |
| Live routing heads, 11M -> 9M | 1806935 -> 1806012 |

Deleting 2M users retired only 923 routing heads (about 0.051%). Ten additional
10000-query sweeps did not reduce that count further. Even the final compact
estimate remains above the desired growth high, so lowering/raising watermarks
alone is not justified by this calibration. No new formal run was started.
This is a fixed-topology measurement, not an actual two-layer shrink result.
The source index was mutated to 9M and saved with calibration-only resize
settings; it must not be mistaken for the original 11M acceptance checkpoint.

Only a successful matched calibration can be supplied to a fresh formal run:

```bash
python3 benchmark-output/sift11m-auto-lifecycle/run_experiment.py \
  --run-id UNIQUE_FORMAL_ID --fullscale-calibration-run UNIQUE_CALIBRATION_ID
```

## Deletion-pressure compaction validation

The FILEIO implementation now provides opt-in
`LayerShrinkCompactLiveRatio` (default `0`, disabled). A value such as `0.9`
permits automatic shrinking only after live users fall to at most 90% of their
persisted peak. If routing memory remains above the low watermark, it first
merges neighboring postings whose deduplicated live union fits the existing
capacity, without changing `MergeThreshold=10` or advancing record versions.
Destination-first checkpoints and a checksummed intent with metadata backups
support interrupted-batch replay. This pauses queries/writes and adds IO; it
does not reclaim retired files or guarantee unchanged physical replica counts.
Both estimated-low and actual-target memory checks still apply.

Focused FILEIO recovery/capacity tests passed, including recovery from torn
mapping/block-pool/version-map checkpoints and refusal of corrupt intents.
Synthetic lifecycles used fixed `MergeThreshold=10` and compaction ratio `0.9`,
with no manual grow/shrink/merge calls:

| Users (initial -> peak -> final) | Posting layers | Recall@10 at peak / shrunk / reloaded |
| --- | --- | --- |
| 1024 -> 16384 -> 256 | 1 -> 2 -> 1 | 1.0 / 1.0 / 1.0 |
| 4096 -> 65536 -> 1024 | 1 -> 2 -> 1 | 0.990625 / 1.0 / 1.0 |

Both checked complete live-ID membership, independently computed result
distances, no duplicate/deleted results and zero concurrent query errors.
These clustered UInt8 synthetic tests are not the matched real-SIFT benchmark.
Compaction attempts are spaced at least five seconds apart to avoid a full
checkpoint after every deletion.

All 21 self-contained FILEIO layer regressions passed, including repeated small
and larger lifecycles; the separately provisioned 10M checkpoint test was
excluded because its index/query/truth environment variables were not set:

```bash
Release/SPTAGTest \
  '--run_test=FileIOLayerTest:!FileIOLayerTest/Sift10MCheckpointReload' \
  --log_level=test_suite --color_output=no
```

The SIFT driver leaves this option disabled unless explicitly requested.
The failed full-scale calibration above is not valid evidence for this new
policy, and no new 9M run has been started. The fresh calibration workflow
below records the dynamic-only option, freezes new thresholds and requires
enough disk before a formal run.

## Matched 5M-pivot experiment with compaction

The driver now accepts an explicit `SPTAG_SIFT_PIVOT` of 5M or 10M and
`SPTAG_SIFT_COMPACT_RATIO` (default 0/off). The matched configuration audit
checks both the original profile and the 5M/0.9 variant; all non-resize index
settings and the 10000-query Top5/32-search-thread evaluation stay matched.
The 5M lifecycle builds 4.5M, inserts twenty 50K batches to 5.5M, then randomly
deletes twenty 50K batches back to 4.5M with seed 1729. No manual resizing or
runtime threshold/ratio changes are used.

```bash
python3 benchmark-output/sift11m-auto-lifecycle/test_profile.py
python3 benchmark-output/sift11m-auto-lifecycle/run_experiment.py \
  --run-id UNIQUE_5M_RUN_ID --pivot-count 5000000 --compact-ratio 0.9 \
  --growth-calibration --discard-growth-index
```

First, a separate single-layer **full-size** 4.5M -> 5.5M growth calibration
runs with unreachable fixed resize budgets. Its measured 5M Head bytes become
the formal high watermark, without extrapolating small-pilot memory.
Target is 98% and low is 94% of high: these deletion-side ratios are hypotheses,
not measured guarantees. The fresh formal build freezes all three thresholds.
It must start with one posting layer, reach two by 5.5M, shrink automatically
back to one after deletion, and pass recall/membership/distance/reload checks.
Random deletion may still leave insufficient compactable routing capacity;
that is reported as failure, never bypassed.

The explicit discard flag removes **only this new run's successful temporary
growth-calibration index**, after retaining its metrics and configuration.
Existing experiment indexes, input, logs and evidence are untouched.
Failed calibration indexes are retained for diagnosis. Startup and the fresh
formal build each require 80 GiB free at the 5M pivot (100 GiB at 10M), with
a 64 GiB RSS cap. Compaction runs check disk every second and stop below
30 GiB: the original 20 GiB reserve plus one 10 GiB FILEIO expansion chunk.
Without the discard flag, keeping both new indexes requires 140/180 GiB
respectively. FILEIO expansion
is chunked, so these are admission guards, not guaranteed peak-space bounds.
Growth calibration and formal execution each have an eight-hour limit.

The first 5M attempt (`sift-20261122-matched-5m-compaction-growth-calibrated`)
stopped during growth calibration after the 5.25M measurement, not during
automatic shrink or recall evaluation. Its temporary index occupied 42 GiB.
A 10 GiB expansion took free disk from approximately 23 GiB to 14 GiB,
triggering the old 20 GiB monitor. The original 50 GiB admission estimate was
too optimistic. The stricter admission and expansion headroom above address
this resource failure without changing the matched index settings. Logs and
measurements remain preserved; incomplete calibration is not reusable as a
successful threshold calibration or a formal acceptance checkpoint.

The disk retry (`sift-20261122-matched-5m-compaction-disk-retry`) completed
automatic 1 -> 2 -> 1 topology changes: growth occurred near 4.9M and shrink
near 4.95M during the next delete batch. Initial/inserted-5M/inserted-5.5M/
deleted-5M Recall@5 were approximately 0.96724/0.96116/0.96224/0.96008.
Final deleted-4.5M Recall@5 was **0.94674**, below the unchanged 0.95 requirement,
so this run failed and did not reach final reload acceptance. No deleted,
duplicate or distance-invalid results were reported by the failed verification.

The original compaction always kept the first capacity-compatible neighbor,
regardless of how it represented the union's live points. The revised merge
chooses the lower-total-distance centroid from the two existing heads to limit
geometric distortion. This is a quality-oriented correction to validate, not
proof that final SIFT recall now passes; no search/replica parameters or recall
thresholds are changed. Source-by-VID mappings are now saved at insertion
evaluation checkpoints, and deletion checkpoints are saved before evaluation,
so a recall failure retains the exact state needed for diagnosis.

The centroid retry stalled at about 725662 routing heads before reaching the
shrink low watermark. Compaction repeatedly restarted scanning from the first
candidate after a timeout. The revised coordinator retains a process-local
candidate snapshot and cursor, yields after at most 512 source candidates or
five seconds of scanning, and continues incomplete sweeps even without a
productive slice. This permits waiting writers and queries between attempts.
A fully scanned, unproductive sweep waits for further deletion rather than
repeating empty work. Durable merge intent/recovery semantics are unchanged;
reload restarts the non-durable scan snapshot safely. This correction still
requires fresh SIFT acceptance; it does not guarantee enough merge capacity
or passing recall.
