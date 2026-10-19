# Contributing

## Before changing code

1. Read the [documentation index](docs/README.md) and
   [support matrix](docs/SupportMatrix.md).
2. Use a focused branch and describe the behavior being changed.
3. Keep generated files, benchmark outputs, credentials, and machine-specific
   configuration out of commits.
4. Keep dependency bumps separate. Do not edit submodule worktrees to apply
   patches already maintained in the parent repository.

Use [GitHub issues](https://github.com/microsoft/SPTAG/issues) for bugs and
feature discussion. Follow [SECURITY.md](SECURITY.md) for private reporting.
Follow the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/);
see its [FAQ](https://opensource.microsoft.com/codeofconduct/faq/) for contacts.

## Build and test

Follow the [Linux](docs/LinuxInstallation.md) or
[Windows](docs/WindowsInstallation.md) instructions. From the repository root:

```bash
node --test Tools/check-docs.test.mjs
node Tools/check-docs.mjs
node Tools/quickstart.mjs Release
PYTHONPATH="$PWD/Release" python3 -m unittest discover -s Test/python -v
```

The Python suite exercises the real built `_SPTAG` extension, not a mock.
Changes to the public SDK also need the
[C++ consumer and independent-process Python examples](docs/QuickStart.md).
Their output is retained, so use fresh directories when repeating a build.
The Linux CI configures that standalone consumer and treats SDK checks as
blocking. Native Windows SDK usage requires its own validation.

Run self-contained Boost.Test suites from the output directory:

```bash
cd Release
./SPTAGTest --run_test=CommonHelperTest:IniReaderTest:ConcurrentHashMapTest:AlgoTest/BKTTest:AlgoTest/KDTTest --log_level=test_suite
```

Use `--list_content` to inspect test names. Select tests relevant to your change;
do not run the entire suite blindly: it includes performance, dataset-dependent,
and storage integration experiments. The benchmark test returns without
running if `BENCHMARK_CONFIG` is unset; that is **not** benchmark validation.
An exit code of zero is not sufficient if the requested experiment was skipped.

For optional backend changes, record the CMake flags and run a corresponding
integration test with the required service/hardware. For RaBitQ changes, enable
`RABITQ`, run its tests, and check accuracy plus model/code compatibility.

`DedupCandidateTest` exposes an existing search visibility/deduplication
regression: a deleted occurrence is marked seen before a later live occurrence.
It is a separate non-blocking diagnostic in CI, with its failure and log
published explicitly. Do not count it as passing or broaden this documentation
change into a search algorithm change without separate review.

## Style and errors

- Follow nearby C++ patterns and the repository's [.clang-format](.clang-format).
- Prefer precise changes and explicit error propagation.
- Document changes to public parameters, storage formats, or compatibility.
- Keep Markdown links relative to the document. Include shell language tags.
- Do not call a configuration supported or a performance number reproducible
  without recording the validation evidence.

## Local debugging

The shared VS Code tasks configure/build a baseline Debug tree in `build`.
They use CMake rather than a host-specific `make` invocation. The Linux GDB
and Windows debugger profiles run the self-contained helper suite without
SPDK devices, external services, Java paths, or forced ASAN preload libraries.

Configure GNU ASAN separately when needed:

```bash
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DUSE_ASAN=ON \
  -DFOLLY=OFF -DTBB=ON -DSPDK=OFF -DROCKSDB=OFF -DTIKV=OFF -DGPU=OFF -DRABITQ=OFF
cmake --build build-asan --target SPTAGTest --parallel 2
```

An ASAN-linked binary loads its matching sanitizer runtime; do not commit an
`LD_PRELOAD` path tied to one GCC installation.

## Pull request checklist

- Explain the problem, scope, and compatibility impact.
- Include exact validation commands and their results, including skipped or
  unavailable checks.
- Update directly affected guides and examples.
- For performance claims, follow [benchmark methodology](BENCHMARK_GUIDE.md).
- Keep local data and outputs untracked; `git diff --check` should pass.

The [GitHub workflow](.github/workflows/ci.yml) covers documentation and a
baseline Linux configuration. The [legacy Azure pipeline](azure-pipelines.yml)
retains its organization-specific security/compliance tasks; they have not been
migrated or certified by this change. The new CI does not replace those checks
or establish native Windows or optional integration coverage.
