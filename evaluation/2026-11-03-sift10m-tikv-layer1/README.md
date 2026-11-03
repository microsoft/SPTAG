# SIFT10M TiKV: single layer, 80 build threads

[Evaluation index](../README.md) | [Results](results.json) |
[All query rounds](query-rounds.csv) | [Insertions](insertions.csv) |
[FILEIO reference](../2026-11-03-sift10m-fileio-layer1/README.md)

**Status: completed; strict count and independent final reload validation passed.**
Only this TiKV experiment is included. Earlier local TiKV results are excluded.

## Configuration and scope

- Release build, FOLLY=ON, TIKV=ON; actual runtime backend TIKVIO.
- One posting layer plus the in-memory BKT head index.
- UInt8 SIFT, 128 dimensions, squared L2, TopK=5.
- 9,900,000 initial vectors plus ten batches of 10,000 inserts.
- 10,000 native queries per pass; two warmup and two measured passes per phase.
- Build threads 80 in all three stages; insert 16, maintenance 48, search 32.
- MaxCheck=8192; PostingPageLimit/SearchPostingPageLimit/BufferLength=4/4/4.
- DistributedVersionMap=true; SearchCheckVersionMapOnlyLayer0=true.
- No simultaneous foreground queries during insertion.
- Isolated three-PD/three-TiKV cluster; replication factor 3.
- Per store: 16GB memory limit, 4GB block cache, gRPC concurrency 8.

[Recorded configuration](tikv-recorded.ini) retains the machine paths and unique
TiKV prefix. [benchmark.ini](benchmark.ini) uses replacement paths and a placeholder
prefix; set all of them before attempting a new experiment.
[tikv.toml](tikv.toml) preserves the server settings.

## Completion and performance

All ten ordered checkpoints had the exact expected counts, ending at 10,000,000.
An independent final reload enumerated exactly `[0,10000000)` and checked all
100,000 inserted metadata entries as per-batch source-ID bijections. This is
logical membership validation, not an exhaustive physical payload scan.
All six owned containers stopped with exit 0 and no OOM indication.
See [count validation](count-validation.json), [reload evidence](final-count-check.log)
and [cluster shutdown](cluster-shutdown.json).

- Initial build: 2454.4685 seconds.
- All 22 measured rounds: Recall@5 95.79% to
  95.80%; QPS
  630.8964 to 835.0540.
- Insertion timer total: 713.4700 seconds for 100,000 vectors;
  aggregate 140.1601 vectors/s.
  This includes the batch background-work barrier but excludes build, query,
  loading, snapshot cloning and saving; it is not end-to-end throughput.

### Second measured query pass at every phase

Both rounds are retained in the CSV; this table does not select the faster round.

| Batch | Active vectors | Recall@5 | QPS | Mean ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 9900000 | 95.80% | 630.8964 | 50.6766 | 61.684 | 65.615 |
| 1 | 9910000 | 95.80% | 787.2518 | 40.6036 | 48.472 | 50.658 |
| 2 | 9920000 | 95.80% | 746.9030 | 42.7994 | 51.521 | 55.937 |
| 3 | 9930000 | 95.80% | 797.4769 | 40.0840 | 47.789 | 50.678 |
| 4 | 9940000 | 95.80% | 779.9046 | 40.9944 | 49.867 | 53.257 |
| 5 | 9950000 | 95.79% | 764.3502 | 41.8205 | 50.143 | 53.720 |
| 6 | 9960000 | 95.80% | 736.6436 | 43.3995 | 52.499 | 57.373 |
| 7 | 9970000 | 95.80% | 759.6604 | 42.0830 | 51.855 | 58.633 |
| 8 | 9980000 | 95.80% | 825.1101 | 38.7441 | 46.062 | 49.195 |
| 9 | 9990000 | 95.80% | 829.0792 | 38.5583 | 45.830 | 49.617 |
| 10 | 10000000 | 95.80% | 752.8906 | 42.4577 | 51.315 | 53.544 |

Raw [benchmark JSON](tikv-output.json), [benchmark log](tikv-run.log) and
[process resources](resources.txt) preserve the original measurements.
Retry/error observations remain in the raw log and results JSON.

## Provenance and limitations

This is a **results-only publication**, not a source-fix release or a complete
byte-for-byte reproduction bundle. The binary and dataset/index payloads are
not included. The same locally fixed executable was used for the published
FILEIO run; local source fixes are not part of this commit.
The recorded base commit `a6c00c6b281e2930558f480d57f67461848f4716` was a local
report commit removed from the publication branch, not a promised remotely
available source revision.

[Binary provenance](binary-build-provenance.json), [input hashes](input-checksums.json)
and [truth verification](truth-verification.json) identify the measured material.
Local validation/build paths in provenance refer to unshipped local artifacts.
[Artifact hashes](artifacts.sha256) cover the published files.

FILEIO and TiKV ran sequentially, not concurrently, with matching workload and
client tuning and the same executable. Each independently built head index has
no fixed random seed. The host is shared and caches were not flushed; TiKV also
has replicated server/cache resources. Do not treat this as an isolated backend
causality study or a production performance guarantee. Completion/count acceptance
does not impose a new recall or latency release threshold.
