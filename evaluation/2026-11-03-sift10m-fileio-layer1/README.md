# SIFT10M FILEIO: single layer, 80 build threads

[Evaluation index](../README.md) | [Results](results.json) |
[All query rounds](query-rounds.csv) | [Insertions](insertions.csv) |
[Checkpoint counts](checkpoint-counts.csv)

**Status: completed; strict count and final reload validation passed.**
This report contains only the FILEIO run completed on November 3, 2026.
Other local experiments and source changes are not published with this report.
For the newer four-profile comparison, use the separate
[November9 matched-binary FILEIO1 run](../2026-11-09-sift10m-fileio-layer1-matched/README.md)
and its [reproduction guide](../2026-11-09-sift10m-fileio-layer1-matched/Reproduce.md).
This historical run and its recorded metrics/provenance remain unchanged.

## Workload and configuration

| Setting | Value |
| --- | --- |
| Backend / posting layers | FILEIO / 1, plus the in-memory BKT head index |
| Build | Release, FOLLY=ON, TIKV=ON; runtime storage is FILEIO |
| Dataset | First 10M SIFT vectors, UInt8, 128 dimensions, squared L2 |
| Initial corpus / updates | 9,900,000 vectors + ten batches of 10,000 inserts |
| Query corpus | 10,000 native SIFT queries, TopK=5 |
| Query rounds | Two warmup and two measured passes per phase; 22 measured passes total |
| Build threads | 80 in SelectHead, BuildHead and BuildSSDIndex |
| Insert / maintenance / search threads | 16 / 48 / 32 |
| Search during insertion | Disabled; queries and foreground insertions are separate phases |
| MaxCheck | 8192 in BuildHead and BuildSSDIndex |
| PostingPageLimit / SearchPostingPageLimit / BufferLength | 4 / 4 / 4 |
| SearchInternalResultNum / LatencyLimit | 64 / 50 |
| SearchCheckVersionMapOnlyLayer0 | true, retained in the configuration; TiKV-only and inactive for FILEIO |

The [portable-path configuration](benchmark.ini) preserves the tuning values.
The [exact recorded configuration](fileio-recorded.ini) retains the original
machine paths. The raw [benchmark JSON](fileio-output.json) and
[benchmark log](fileio-run.log) are preserved without modification.

The shared host exposes 96 CPUs, two sockets, four NUMA nodes and an AMD EPYC
7V12 processor. No active vLLM processes were observed before this run;
the [startup host record](host-state-start.json) reports idle GPUs.
This was not a dedicated-host measurement.

## Completion and membership acceptance

- The benchmark exited 0; all ten ordered insertion checkpoints completed.
- Initial local VersionMap count was 9,900,000, not zero.
- Every checkpoint reported the expected bottom-layer sample and contained-ID
  counts, increasing by exactly 10,000, with no deleted bottom-layer vectors.
- Each batch reloaded the correct pre-insertion count.
- An independent final reload reported exactly 10,000,000 samples.
- Final contained IDs were exactly `[0,10000000)`.
- All 100,000 inserted metadata entries formed a source-ID bijection within
  their respective batches; concurrent allocation does not imply VID equals
  source row.
- All 22 measured passes used 10,000 real queries and phase-specific exact truth.
  No random-input fallback was used.

See [count validation](count-validation.json) and the independent
[final reload log](final-count-check.log). The 10M checker validates logical
membership and metadata, **not an exhaustive physical payload scan**.
Completion/count acceptance is not a new recall or latency release threshold.

## Observed performance

| Metric | Observation |
| --- | --- |
| Initial build | 1086.4506 seconds, about 18.1 minutes |
| Logged select / head build / SSD build | 156 / 527 / 402 seconds |
| Benchmark process wall time | 30:56.32 |
| Maximum resident set | 6,321,232 KiB, about 6.03 GiB |
| Recall@5 across all 22 measured rounds | 95.82% to 95.86% |
| QPS across all 22 measured rounds | 5538.0093 to 6766.8521 |

Process-wide measurements are in [resources.txt](resources.txt).

### Queries: second measured pass at every phase

Both measured passes are retained in [query-rounds.csv](query-rounds.csv);
this table consistently selects round 2, not the faster round.

| Batch | Active vectors | Recall@5 | QPS | Mean ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| Initial | 9,900,000 | 95.86% | 6617.9458 | 4.8178 | 5.878 | 6.266 |
| 1 | 9,910,000 | 95.84% | 6600.5469 | 4.8305 | 5.858 | 6.821 |
| 2 | 9,920,000 | 95.84% | 6306.5776 | 5.0569 | 6.117 | 7.452 |
| 3 | 9,930,000 | 95.83% | 6026.0967 | 5.2925 | 7.471 | 12.271 |
| 4 | 9,940,000 | 95.83% | 6300.0854 | 5.0612 | 6.251 | 9.847 |
| 5 | 9,950,000 | 95.82% | 5590.6899 | 5.7041 | 8.003 | 12.032 |
| 6 | 9,960,000 | 95.82% | 5539.9668 | 5.7588 | 7.179 | 11.808 |
| 7 | 9,970,000 | 95.82% | 5632.7397 | 5.6604 | 6.850 | 8.552 |
| 8 | 9,980,000 | 95.83% | 5788.6987 | 5.5084 | 6.671 | 8.514 |
| 9 | 9,990,000 | 95.84% | 6399.6929 | 4.9812 | 6.050 | 6.406 |
| 10 | 10,000,000 | 95.84% | 6341.2988 | 5.0277 | 6.094 | 8.875 |

### Insertions and benchmark overhead

The first 10K batch took 11.2554 seconds (888.463 vectors/s), including
6.4638 seconds waiting for background work. Batches 2-10 took
1.9161-2.1586 seconds each (4632.6454-5219.0242 vectors/s).

The sum of insertion durations was 29.8175 seconds for 100,000 vectors:
aggregate insertion throughput was 3353.7352 vectors/s. This excludes build,
query, index cloning, loading and saving; it is not end-to-end throughput.
The measured insertion timer includes the batch background-work barrier.

Snapshot cloning took 354.8254 seconds in total; loading and saving took
29.5964 and 30.5584 seconds respectively. Per-batch values are retained in
[insertions.csv](insertions.csv); barrier times are in the raw JSON.

## Provenance and reproducibility limits

**This is a results-only commit, not a source-fix release or a complete
byte-for-byte reproduction bundle.** The measured executable included local,
uncommitted FILEIO initialization and TiKV count-handling fixes. Those source
changes, their patch, the binaries, input vectors, truth payload and index
files are intentionally excluded. Building this report's repository revision
alone does not reproduce that executable and can retain the FILEIO
initialization regression.

The recorded base commit is `a6c00c6b281e2930558f480d57f67461848f4716`;
it was a local report commit subsequently removed from the publication branch.
It is a provenance identifier, not a promised remotely available source revision.
The measured executable SHA256 is
`fdcbef6e008c9be8f22bc6d1fdb1d651c9b22e32653bb2cdf064067f86ea41ec`.
The archived local patch SHA256 is
`05345344bd155c182b1b276a7fe579f83060fd4c843d0619d38b44a573fe13fa`.

[Binary provenance](binary-build-provenance.json) is the original local record:
its small-validation and build-snapshot paths refer to local-only material,
not missing files promised by this report. [Input hashes](input-checksums.json)
and [truth verification](truth-verification.json) identify the reused inputs.
[Artifact hashes](artifacts.sha256) cover the published report files.

This is one independent build with no fixed random seed or cache flush.
Variation across query phases is observational, not proof of a performance
regression. No TiKV measurements or cancelled/failed FILEIO runs are included.
