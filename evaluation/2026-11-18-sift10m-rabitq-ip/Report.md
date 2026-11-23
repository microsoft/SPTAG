# Normalized RaBitQ4 / OPQ64 experiment report

[Overview and reproduction](README.md)

## Scope and method

The completed comparison comprises RaBitQ4 L2 and head-repaired RaBitQ4 IP
at 1M-5M, plus encoding-repaired OPQ64 IP at 1M. OPQ 2M-5M are **not run**.
Each growth case starts from 1M and inserts exactly 1M at a time; no smaller
diagnostic is substituted for a million-scale result.

| Setting | Value |
| --- | --- |
| Host | Shared Ubuntu 24.04 host, Xeon E5-2698 v4, 20 cores / 40 threads, approximately 251GiB RAM |
| Storage | All selected current runs use `/data`, RAID0 `/dev/md0`; no TiKV server |
| Compute | CPU-only; GCC 13.3, CMake 3.28.3; Release, FOLLY off, TBB/RaBitQ on, GPU/SPDK/RocksDB/TiKV off |
| Input | Source-order first 1M-5M SIFT128 vectors, original 10K queries |
| Normalization | FP64 squared-norm accumulation/division, stored Float32; no rows removed |
| Truth | Exact top 5 per prefix, independently computed squared L2 and `1-dot` on stored Float32, FP64 accumulation, source-ID tie break |
| Index | One FILEIO posting layer plus BKT, head selection ratio 0.2 |
| Threads | Build/search 32, insertion 16, maintenance 24 |
| Search | MaxCheck 8192, internal results 64, latency limit 50ms, SearchK 20 |
| Posting profile | Posting/search/buffer pages 4/4/4; PostingVectorLimit 0 |
| Measurement | Excluded warmups, two primary measured rounds, save, fresh-process reload with two rounds, logical and physical current-ID checks |

Recall5@20 is the fraction of the exact five source-ID neighbors found among
twenty returned results, averaged over all 10K queries. The harness maps index
VIDs back to source IDs; logical counts or a legacy index-VID recall are not
substitutes. There is no raw-vector reranking, concurrent insertion/query
measurement or reconstruction renormalization.

Normalized IP is cosine-equivalent for nonzero original vectors, **not raw
maximum-inner-product search**. Stored Float32 norms/ties mean its truth must
not simply be copied to the squared-L2 case. The stored 5M input has zero
zero-norm and nonfinite rows. Its independently reread nonzero norms range from
0.9999999476993111 to 1.0000000510977294, within the 1e-6 tolerance.
Source/model/truth hashes are pinned in the [reproduction script](tools/reproduce.py).

## RaBitQ4 recall and performance

QPS/latency below use primary round 2. Every milestone has zero missing current
source IDs, and both primary and both fresh-reload rounds agree on recall.

| Vectors | L2 recall | Repaired IP recall | L2 QPS | IP QPS |
| --- | ---: | ---: | ---: | ---: |
| 1M | 97.98% | 97.83% | 1,592.7140 | 1,898.0227 |
| 2M | 98.21% | 98.09% | 828.3331 | 929.4251 |
| 3M | 97.86% | 97.87% | 707.0302 | 672.1770 |
| 4M | 97.50% | 97.60% | 571.5898 | 650.8518 |
| 5M | 97.40% | 97.31% | 591.0261 | 671.4590 |

The 1M-5M decline is **0.52 percentage points for IP versus 0.58 for L2**.
At 5M, IP is 0.09 points below L2.

| Vectors | Operation | L2 seconds | IP seconds | Included L2 drain seconds | Included IP drain seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| 1M | Initial build | 739.9288 | 724.8297 | N/A | N/A |
| 2M | Insert 1M | 258.7296 | 270.3766 | 0.4815 | 0.240866 |
| 3M | Insert 1M | 388.9067 | 441.3208 | 28.3725 | 66.896145 |
| 4M | Insert 1M | 673.4642 | 705.4742 | 268.3725 | 299.273674 |
| 5M | Insert 1M | 740.9586 | 768.8240 | 334.056310 | 355.540433 |

Insertion seconds already include the final maintenance drain: **do not add
the drain again**. Preparation, checkpoint copying/loading, clone/save,
queries/reload and external audits are not included in that insertion field.
The reproduction keeps their separate timings in local run outputs.

| Case | Vectors | Mean ms | p50 ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| RaBitQ4 L2 | 1M | 20.0639 | 19.386 | 30.881 | 37.678 |
| RaBitQ4 L2 | 2M | 38.5968 | 37.174 | 61.250 | 75.322 |
| RaBitQ4 L2 | 3M | 45.2094 | 43.365 | 69.653 | 82.694 |
| RaBitQ4 L2 | 4M | 55.9359 | 55.641 | 78.257 | 94.549 |
| RaBitQ4 L2 | 5M | 54.0882 | 52.633 | 80.298 | 95.871 |
| RaBitQ4 IP | 1M | 16.8336 | 16.117 | 26.139 | 31.166 |
| RaBitQ4 IP | 2M | 34.3874 | 32.977 | 52.340 | 61.652 |
| RaBitQ4 IP | 3M | 47.5547 | 46.527 | 70.242 | 87.318 |
| RaBitQ4 IP | 4M | 49.1274 | 47.836 | 71.749 | 93.778 |
| RaBitQ4 IP | 5M | 47.6054 | 45.829 | 71.107 | 81.552 |

Fresh-process reload round 2 retained the same recall as the primary rounds:

| Vectors | L2 reload QPS | IP reload QPS |
| --- | ---: | ---: |
| 1M | 1,597.2180 | 1,521.4725 |
| 2M | 831.3484 | 764.0402 |
| 3M | 610.3639 | 616.1218 |
| 4M | 702.8159 | 642.7462 |
| 5M | 603.3426 | 594.1398 |

## Repaired OPQ64 IP: actual 1M result

The rebuilt OPQ index completes build, query, save, fresh reload and full
current-ID coverage. All four measured passes report 95.79% recall.
Build time is **317.3490 seconds**; no insertion/drain timing exists at 1M.

| Process | Round | Recall5@20 | QPS | Mean ms | p50 ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary | 1 | 95.79% | 1,429.6069 | 22.3629 | 22.066 | 32.810 | 40.384 |
| Primary | 2 | 95.79% | 1,448.0240 | 22.0742 | 21.735 | 31.912 | 39.320 |
| Fresh reload | 1 | 95.79% | 1,503.2128 | 21.2609 | 20.784 | 31.471 | 39.931 |
| Fresh reload | 2 | 95.79% | 1,490.0929 | 21.4281 | 21.012 | 31.599 | 38.375 |

There are 201,598 live heads, no retired heads and 7,274,369 current records.
All 1M source IDs are reachable, with no duplicate current records. The maximum
posting has 453 records, below the hard limit 474. Both processes returned zero
and their reviews contain no warning/error messages.

The canonical checkpoint occupies 1,151,631,360 allocated bytes, versus
11,915,415,552 for the head-only/old-encoding baseline. This first-1M reduction
is **not a linear 5M storage forecast**. No later OPQ milestone was measured.

## Useful before/after findings

### IP head maintenance

The original normalized IP run's recall and missing current-ID counts were:

| Vectors | Original IP recall | Original missing IDs | Repaired IP recall | Repaired missing IDs |
| --- | ---: | ---: | ---: | ---: |
| 1M | 97.95% | 0 | 97.83% | 0 |
| 2M | 98.03% | 448 | 98.09% | 0 |
| 3M | 97.07% | 2,667 | 97.87% | 0 |
| 4M | 95.92% | 7,593 | 97.60% | 0 |
| 5M | 95.50% | 11,386 | 97.31% | 0 |

That older run used `/home`, so its timings are not a
matched-storage comparison. Missing IDs are not the sole demonstrated cause
of the recall gap.

IP self-distance need not be zero, and different encoded vectors can have
negative IP distance. The head repair therefore uses VID identity rather than
`distance < Epsilon`, keeps representatives consistent with their partitions,
uses reconstructed centroid geometry, and preserves source postings until
destination capacity/publication checks succeed. It does not claim cross-key
crash atomicity or change the official RaBitQ scoring estimator.

Original OPQ's actual initial-1M attempt
was stopped after 3,600s without observable initial-refill progress; it did not
produce a completed 1M query/reload/coverage result. The same head repair
unblocked the build, but left the separate encoding-quality problem below.

### OPQ codeword assignment

| Native OPQ64 IP 1M | Recall5@20 | Primary round-2 QPS | Build seconds | Missing IDs |
| --- | ---: | ---: | ---: | ---: |
| Head repair, old encoding | 64.87% | 1,450.9050 | 730.1244 | 0 |
| Head repair, corrected encoding | 95.79% | 1,448.0240 | 317.3490 | 0 |

The old-encoding native result
uses the same frozen FAISS L2-trained `OPQ64,PQ64x8` model. Selecting stored
codewords by maximum dot product favored large-norm words, even when an
exact reconstruction was available. The repair minimizes L2 reconstruction
error **only for encoding**; ADC/SDC still use the configured IP scoring.
It does not retrain the model, clamp negative scores or normalize reconstructions.
Existing indexes remain readable but must be re-encoded/rebuilt for this benefit.

The independent all-1M/all-10K diagnosis
visits live current replicas, verifies consistent codes and scores exhaustively
in FP64, using source-ID boundary-tie breaking:

| Representation / score | Exhaustive Recall5@20 |
| --- | ---: |
| Old stored codes / IP | 67.728% |
| Old codes / normalized reconstruction | 62.482% |
| L2-nearest codes / unchanged IP | 97.640% |

Mean squared reconstruction error falls from 1.496828922 to 0.004258332;
mean reconstructed norm changes from 2.030064434 to 0.995908396.
The corrected assignment changes 63,586,079 of 64,000,000 subcodes.
These reference recalls are **not native Float32 ANN measurements or a formal
upper bound on native ANN recall against raw-vector truth**.

## Coverage, warnings and limitations

Coverage checks logical source-ID membership against current-version records
reachable through live heads. Version 255 is valid. A retired head is not itself
data loss when the current source data remains elsewhere. Zero missing source
IDs does not imply that every historical replica is retained or every ANN
query will find every ID.

The repaired RaBitQ IP run retains 5,943 asynchronous-read timeout warnings
across primary/reload processes, and four primary `Abort!!!` warnings.
All processes nevertheless exit zero, reload successfully and pass coverage.
The L2 run also has timeouts/head-lock retries; neither current RaBitQ case
has error-level messages. Detailed warnings remain in the machine-local run logs.
This is not a warning-free or dedicated-host experiment.

Independent randomized builds, cache state, shared-host load and the 50ms
search limit confound causal performance comparisons. There are no repeated-run
confidence intervals. The measured recall/coverage goals are observations on
this workload, not universal correctness or concurrency guarantees.
OPQ remains 2.04 recall points below RaBitQ IP at 1M.

Both frozen models were trained on eligible normalized initial-1M rows.
OPQ used FAISS 1.10.0 with internal 65,536-row sampling caps; its native rotation
stores FAISS's `A.T`. Neither the codebook nor the rotation was retrained for the
encoding repair or later prefixes.

| Quantizer | Code bytes | Posting record bytes | Base / hard capacity, 4+4 pages |
| --- | ---: | ---: | ---: |
| Global RaBitQ4, model v6 | 88 | 93 | 176 / 352 |
| OPQ64, 8 bits per subquantizer | 64 | 69 | 237 / 474 |

## Reproducibility and provenance

The three core revisions reproduced by the setup script are `aac5e14`,
`9480ddd` and `145255a`. They match the recorded 204-file baseline,
two-header head repair, and three-header head-plus-encoding repair respectively.
All cases use the same benchmark harness SHA-256:
`8b905225758b68bf071c249685b92fabb07c5f090186415029c0e20cc2a167ce`.

| Measured executable | SHA-256 |
| --- | --- |
| L2 baseline | `5132806619d58bd67a1b6bd42c9e779c341d3e11d768e9530e769d9b8ba20a79` |
| RaBitQ IP head repair | `17d2ca1ba5df3c4bd397efd1cdaaa9bc097900b3a725976fbaf7a0d7f7782741` |
| OPQ IP encoding repair | `b76fdbc49d0a258bdd35b3d80607f3222345aaacd3a4e68cc088e61b9420817c` |

| Frozen model | SHA-256 |
| --- | --- |
| RaBitQ4 | `feaebba9a7d116a72b8535f29f5077d8b786bfb0dc897c449fd7d28d01f338d1` |
| OPQ64 | `8080a514d5e5988cd6cf445003df4c25b575b1b876690359b807dd09e668712d` |

Model binaries are external input artifacts, not committed files. Supply the
original frozen model through `reproduce.py run --model`; the runner verifies
its hash against the values above.

All result tables above are taken from the measured per-prefix native outputs
and coverage audits. Only this Markdown summary is committed; raw JSON/CSV,
repeated configurations, build manifests and temporary tests are excluded.
Fresh runs create their own local logs/manifests, and `reproduce.py report`
exports recall, performance, latency, reload and coverage tables as Markdown.

The script pins the original UInt8 inputs, every normalized Float32 prefix,
both metrics' exact truths, the frozen models and the three core revisions.
The recipe needs no historical checkpoint or archived build manifest.
A rebuilt executable is not claimed byte-identical to the measured binaries.
These experiment results do not imply that the full native test suite passed.

RaBitQ IP completed on 2026-11-21, with its final manifest written at 17:28:33Z.
The repaired OPQ 1M manifest was written at 2026-11-22T02:34:07.777285Z.
These are manifest-write times, not instrumented process-exit timestamps.
Large indexes/raw datasets and redundant failed-attempt/source archives are
excluded from Git. Existing machine-local checkpoints are left untouched;
the frozen models and reproduction entry point reproduce the native comparison
without those archives.
