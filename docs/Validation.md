# Recorded local validation

[Documentation index](README.md) | [Support matrix](SupportMatrix.md)

This record describes validation of the documentation worktree based on
`328e5337` on 2026-10-19, not a remote CI run or a released version.
Re-run the commands after later code or dependency changes.

## Environment

| Component | Actual local environment |
| --- | --- |
| Host / execution environment | Windows host, Ubuntu 24.04 under WSL |
| CPU path used by CLI | AVX2 |
| GCC | 13.3.0 |
| CMake / generator | 3.28.3 / Ninja |
| Boost | 1.83 |
| oneTBB | 2021.11.0 |
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
