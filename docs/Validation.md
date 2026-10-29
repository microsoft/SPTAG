# Recorded local validation

[Documentation index](README.md) | [Support matrix](SupportMatrix.md)

This record describes local checks on 2026-10-19. The initial documentation
worktree was based on `328e5337`; the SDK follow-up was committed as
`f042b770`, based on `49ddc8bd`. The engine-oriented follow-up below is based
on `f042b770`. These checks are not a versioned SDK release. Re-run them after
later code or dependency changes.

## Environment

| Component | Actual local environment |
| --- | --- |
| Host / execution environment | Windows host, Ubuntu 24.04 under WSL |
| CPU path used by CLI | AVX2 |
| GCC | 13.3.0 |
| CMake / generator | 3.28.3 / Ninja |
| Boost | 1.83 |
| oneTBB | 2021.11.0 |
| Python / NumPy used by SDK tests | 3.12.3 / 1.26.4 |
| Node.js | 22.23.3 in WSL; checker also tested with Windows Node 24.14.0 |
| CMake feature flags | `TBB=ON`; `FOLLY`, `SPDK`, `ROCKSDB`, `TIKV`, `GPU`, `RABITQ` off |

## Passed

- CMake configure and build of `indexbuilder`, `indexsearcher`, and `SPTAGTest`.
- Real CLI example: 1,024 generated Float vectors, three exact self-queries,
  persisted BKT index, two independent search processes; all IDs/distances and
  output shapes checked.
- Eleven selected Boost.Test cases, 238 assertions: BKT, KDT, common helpers,
  INI reader, and the non-Folly map regression suite.
- Documentation checker self-tests: seven cases on both Windows and Linux.
- Local Markdown links, heading fragments, images, and code-fence checks on
  both Windows and Linux (including Linux path case sensitivity).
- YAML parsing of GitHub Actions and the preserved Azure configuration;
  JSON parsing of shared VS Code configurations.
- Bash syntax and safe refusal of retired `init_env` and non-opted-in legacy
  experiment execution; missing CLI binaries fail explicitly.
- `git diff --check`.

The C++ test selection was:

```bash
Release/SPTAGTest \
  --run_test=CommonHelperTest:IniReaderTest:ConcurrentHashMapTest:AlgoTest/BKTTest:AlgoTest/KDTTest \
  --log_level=message --report_level=short
```

The remaining tests were not all run; selected-suite success is not a clean
full-suite claim.

## SDK follow-up

The developer-facing workflow adds real public-API checks, not just CLI tests:

- Root-project configuration and build of `_SPTAG`, `indexbuilder`,
  `indexsearcher`, and `SPTAGTest` passed.
- The standalone consumer configured from `Examples/cpp` and built all five
  CI targets: `sptag_sdk_example`, `_SPTAG`, `indexbuilder`, `indexsearcher`,
  and `SPTAGTest`. The consumer uses `SPTAG::SPTAG` without
  manually duplicating the library's include paths, C++ standard, or feature macros.
- The C++ example built/saved a BKT index and queried it, then a separate
  process loaded the retained index and ID sidecar. Three exact self-matches
  passed in both processes. Existing output directories and missing-index
  loads failed explicitly.
- Seven Python SDK tests passed: BKT/KDT result contracts, persistence/repeated
  searches, metadata bytes, repeated result lifetimes, wrong buffer size,
  own-data subprocess build/load, and invalid example input.
  The invalid-input test includes 13 rejection cases and checks that no output
  directory is created.
- The documented four-vector own-data example returned `doc-A`, row 0,
  distance zero from independent build/load processes. The 128-vector
  demonstration also completed from independent processes.
- The Python metadata regression was first reproduced as uninitialized
  result-list slots. Searches without metadata now return explicit `None`
  entries; the result-lifetime and shape tests exercise the repaired typemap.
- Python syntax compilation, 44 Markdown files on Windows/Linux, seven
  documentation-checker tests, workflow YAML parsing, and patch whitespace
  checks passed.
- After the consumer build, the original CLI smoke test passed again and
  the selected C++ suites again passed 11 cases / 238 assertions. The other
  147 C++ cases were not selected.
- The Python code block in the root README was executed directly and returned
  rows `[0, 1, 2]` with squared L2 distances `[0.0, 512.0, 2048.0]`.

The Python suite command was:

```bash
PYTHONPATH="$PWD/Release" OMP_NUM_THREADS=2 \
  python3 -m unittest discover -s Test/python -v
```

The [SDK quick start](QuickStart.md) contains the consumer build and
independent-process example commands. This follow-up's expanded CI workflow
has not yet been executed remotely. The preceding commit's
[successful baseline run](https://github.com/microsoft/SPTAG/actions/runs/39192508235)
does not validate the SDK checkpoint or the engine-oriented follow-up.

## Engine-oriented follow-up

The documentation now describes disk indexing, dynamic maintenance, compression,
and scalable construction as engine capabilities, with readiness tracked
separately. The following additional local checks passed:

- The documented Python SPANN example built a 128-vector, eight-dimensional
  Float/L2 index using the STATIC backend, saved it, and reloaded it.
  A separate Python process loaded the retained artifacts and returned the
  same three exact self-matches: rows 0, 31, and 63, with squared L2 distance zero.
- The new regression checks nonempty `SPTAGFullList.bin*` posting files and the
  saved loader's `IndexAlgoType=SPANN`, `Storage=STATIC`, and `Layers=1`.
  Both build and independent-load processes must return three correctly
  shaped query results with the expected business IDs.
- The complete public Python suite passed **8 tests**. Its invalid-input test
  applies the same 13 rejection cases to BKT and SPANN: **26 subcases**,
  all failing before an output directory is created. Python syntax compilation
  also passed.
- The original CLI save/load/repeat-search example passed again (3/3 matches).
  The selected C++ suites passed again (11 cases / 238 assertions);
  the other 147 cases were not selected.
- Documentation-checker self-tests passed on Windows and WSL Linux (7 cases
  each), as did link/fragment/fence checks across 44 Markdown files.
  The expanded CI YAML parsed successfully, and `git diff --check` passed.
- With the actual `RABITQ=OFF` binary, the exact documented automatic-tuning
  INI was rejected with exit code 1 and the explicit `RABITQ=ON` requirement.
  No quantizer, encoded corpus, or index was created.

The SPANN commands, with `PYTHONPATH` and `OMP_NUM_THREADS` set as above, were:

```bash
python3 Examples/python/quickstart.py --algorithm SPANN --index quickstart-output/spann
python3 Examples/python/quickstart.py --index quickstart-output/spann \
  --load --queries quickstart-output/spann/queries.npy
```

The teaching driver reads the full input into NumPy in both build and load
modes; it is not an out-of-core build or a large-scale recall/performance result.
The STATIC posting path is
not the dynamic FILEIO path.
The native RaBitQ recipe was checked against source, but the enabled
train/encode/build/query pipeline was not run. TiKV, dynamic maintenance,
multi-layer iterators, distributed builds, native Windows, and the expanded
remote CI workflow were not certified by these checks. No core search
algorithm or known dynamic-visibility blocker was changed.

## Failed or unavailable

- The existing `DedupCandidateTest` fails deleted/live visibility sequences.
  Inspection confirms the existing search loops mark candidates seen before
  checking deletion. No search algorithm fix is included here. CI exposes this
  separately with a warning and diagnostic artifact.
- The initial no-Folly map build failed on its container API and locking
  implementation. The accompanying adapter repair and regression tests enabled
  the validated baseline. The original Folly-enabled implementation is unchanged.
- A Linux `TBB=OFF` attempt also exposed incomplete queue APIs. That configuration
  is not the documented baseline and was not repaired by this change.
- Docker is not installed in the validation environment, so the Dockerfile was
  reviewed and its equivalent native Ubuntu build commands exercised, but the
  container image itself was not built.
- Native Windows, remote GitHub/Azure execution, TiKV cluster integration,
  RaBitQ accuracy/model migration, SPDK device access, MPI multi-host execution,
  and CUDA were not certified.

## Reproduce

Follow [Linux installation](LinuxInstallation.md) for a fresh configure/build,
then run the [quick start](QuickStart.md) and
[contribution checks](../CONTRIBUTING.md). Use a separate configuration and
appropriate hardware/services for optional integrations. Never substitute a
successful baseline or a skipped benchmark for integration evidence.
