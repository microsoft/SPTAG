# SIFT10M TiKV: 16 independent connections and 8 completion queues

[Evaluation index](../README.md) | [Reproduction guide](Reproduce.md) |
[All 22 rounds](query-rounds.csv) | [Insertions](insertions.csv) |
[Raw output](tikv-output.json) | [Raw log](tikv-run.log)

**Completed on 2026-11-05. All count/reload checks passed. Recall decreased
relative to the preceding independently built index; precision equivalence is
not established by this full run.**

## What ran

- Release, TIKV=ON, FOLLY=ON; runtime TIKVIO and distributed VersionMap.
- UInt8 SIFT, 128 dimensions, squared L2, TopK=5.
- First 9.9M vectors, then ten 10K insert batches, ending at 10M.
- Native 10K queries; 22 measured passes, with harness warmups retained.
- One posting layer; build 80, insert 16, maintenance 48, search 32 threads.
- MaxCheck=8192; PostingPageLimit/SearchPostingPageLimit/BufferLength=4/4/4.
- DistributedVersionMap=true; SearchCheckVersionMapOnlyLayer0=true.
  Layer 0 still performs version checks. Normal query-triggered maintenance
  was not disabled for this full experiment.
- Client: 16 independent gRPC channels per client/store and eight completion
  queues per client. These are compiled defaults, not INI settings.
- Fresh isolated 3 PD + 3 TiKV deployment on one shared host, replication 3.
  Per-store settings: 16GB memory limit, 4GB block cache, gRPC concurrency 8.
  This is a TiKV configuration limit, not a Docker hard memory limit.

[Recorded INI](tikv-recorded.ini) preserves the original absolute paths.
[Portable template](benchmark.ini), [deployment code](deployment.py), and
[server configuration](tikv.toml) support a new run without original local
directories. Do not copy the recorded index path or key prefix.

## Full-run results

| Metric | Value |
| --- | ---: |
| Initial build | 1624.5751 s |
| Insert 100K, sum of batch insertion timers | 361.1583 s |
| Benchmark process wall time | 47:50.29 |
| Start to completed validation/shutdown | approximately 49:36 |
| Recall@5 across 22 rounds | 95.52%-95.55% |
| QPS across 22 rounds | 1795.1722-2018.2438 |
| Final phase, second pass: QPS | 1852.3547 |
| Final phase, second pass: mean/P95/P99 | 17.2516 / 21.682 / 24.543 ms |
| Final phase, second pass: Recall@5 | 95.53% |
| Final posting/version timer means | 6.9814 / 7.1849 ms |

Insertion timers include the batch background-work barrier; they exclude
build, query, load, clone, and save times and are not end-to-end throughput.
Both passes at every phase are preserved, not just the faster one.

### Context against earlier runs

| Final phase, second pass | Original TiKV | Prior 8 connections / 4 CQ | This 16 / 8 run |
| --- | ---: | ---: | ---: |
| QPS | 752.8906 | 1263.0444 | 1852.3547 |
| Mean ms | 42.4577 | 25.3037 | 17.2516 |
| P99 ms | 53.544 | 38.756 | 24.543 |
| Recall@5 | 95.80% | 95.74% | 95.53% |

The [original TiKV report](../2026-11-03-sift10m-tikv-layer1/README.md)
is already published. The 8/4 column is a contextual summary of the preceding
local full run, not another experiment bundled in this publication.
Relative to 8/4, QPS rose 46.66%, mean latency fell 31.82%, and P99 fell 36.67%.
Recall fell **0.21 percentage points**. The heads were independently rebuilt
without a fixed random seed, background scheduling can differ, and the host
is shared; the cause of the recall difference has not been isolated.
Do not claim identical precision or attribute every performance difference
solely to transport settings.

## Acceptance and reproducibility

All ten checkpoints had the expected counts. An independent reload reported
10M samples, enumerated exactly `[0,10000000)`, and verified all 100K inserted
metadata entries as source-ID bijections within their batches.
This does not exhaustively scan physical posting payloads.
All six owned containers stopped with exit 0 and no OOM indication.
See [count evidence](count-validation.json), [reload log](final-count-check.log),
and [shutdown record](cluster-shutdown.json).

The measured production source and required count/FILEIO fixes are now
committed at **95bcb20f48f7864072f9443ec91027875382b722**; no source patch is needed.
The publication branch also retains a newer build-memory change from the
remote branch. Reproduce this measurement from the fixed source commit, not
the merged branch tip. The guide creates separate report and source checkouts.
[Publication provenance](publication-provenance.json) verifies the committed
source hashes against the measured files. [Original build provenance](binary-build-provenance.json)
and [recorded source base](recorded-source-base.txt) are unchanged historical
records from before that source commit; their local-only wording is not the
current checkout requirement.

The [reproduction guide](Reproduce.md) covers dependencies, pinned submodules,
protobuf generation, input extraction and exact phase truth, image digests,
deployment, execution, count checking, and shutdown. Tests cover deployment
command parity and acceptance failure cases. A separate checkout of the
committed source was rebuilt and passed six focused regressions, reusing
already-built pinned third-party dependencies. The new helper binaries were
built and the truth generator was checked against integer brute force.
The portable preparation path was tested on the real inputs with regenerated
source binaries and existing verified truth; it does not depend on old runs.

The new portable wrapper has **not** been used for another full 10M experiment,
nor was a clean-machine dependency installation repeated. There is no promise
of byte-identical binaries, identical randomized indexes, identical recall, or
identical timing. Report all 22 rounds and any recall changes. Completion/count
acceptance is not a new performance or recall release threshold.

Dataset files, truth/index/database payloads and executables are not in Git.
Input/truth hashes and their format are in [input checksums](input-checksums.json)
and [truth verification](truth-verification.json). The truth can be regenerated
with the included source. [Artifact checksums](artifacts.sha256) cover this report.
