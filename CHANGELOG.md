# Changelog

## Unreleased

### Documentation and maintenance

- Introduce a unified documentation index, platform/feature status, explicit
  build dependencies, and a small self-checking CLI example.
- Align the Linux baseline with Ubuntu 24.04, GCC 13, C++20, TBB enabled, and
  Folly plus optional storage/quantization integrations disabled.
- Separate user guides from developer contracts and historical experiments.
- Identify SPIRE (arXiv:2512.17264) as repository research and connect its paper,
  citation, and distributed construction workflow without implying full paper
  reproduction coverage.
- Correct command names, configuration keys/sections, output paths, and
  broken local links; preserve historical performance claims with provenance
  limitations rather than treating them as current guarantees.
- Add contribution guidance, project-specific PR/issue templates, documentation
  checks, and baseline GitHub Actions validation.
- Preserve the legacy Azure security/compliance tasks and repair their YAML
  indentation without claiming that remote pipeline execution was validated.
- Remove the tracked Visual Studio database and personal debugger/device
  settings. CODEOWNERS is intentionally unchanged.

### Compatibility and behavior

- Repair the non-Folly concurrent map's container compatibility and locking.
  Read iterators now own snapshots; conditional updates hold an exclusive
  lock. This fallback favors correctness over Folly-equivalent performance.
- Retire `build_spresh.sh init_env`; disk formatting and rebinding are never
  ordinary build steps. Legacy experiment modes require explicit opt-in and
  output paths, refuse existing output data, and no longer use personal PCI
  defaults.
- Report the existing deduplication/visibility test failure separately; it is
  not fixed or counted as passing by this change.

See [recorded validation](docs/Validation.md) for checks actually run and
platform/integration limits. This entry is not a release announcement.
