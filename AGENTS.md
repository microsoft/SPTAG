# SPTAG Agent Navigation

This file is a navigation map for coding agents working in this repository.

## Scope
- Repository: `SPTAG`
- Primary language: C++ (core), SWIG wrappers (Python/Java/C#), Python packaging
- Build systems: CMake + `setup.py`

## High-Level Architecture
- Core ANN library: `AnnService/inc/Core/**`, `AnnService/src/Core/**`
- Service binaries: `AnnService/src/{Server,Client,Aggregator,IndexBuilder,IndexSearcher,...}`
- Wrappers and bridge layer: `Wrappers/inc/**`, `Wrappers/src/**`
- Python package output: `sptag/`

## First Files To Read
- Project overview: `README.md`
- Top-level build config: `CMakeLists.txt`
- Core build targets: `AnnService/CMakeLists.txt`
- Wrapper build and SWIG invocation: `Wrappers/CMakeLists.txt`
- Python packaging: `setup.py`

## Core Code Map
- Abstract index interface: `AnnService/inc/Core/VectorIndex.h`
- Core index orchestration (build/load/save): `AnnService/src/Core/VectorIndex.cpp`
- Search query/result types: `AnnService/inc/Core/SearchQuery.h`, `AnnService/inc/Core/SearchResult.h`
- Algorithm implementations:
  - BKT: `AnnService/inc/Core/BKT/**`, `AnnService/src/Core/BKT/**`
  - KDT: `AnnService/inc/Core/KDT/**`, `AnnService/src/Core/KDT/**`
  - SPANN: `AnnService/inc/Core/SPANN/**`, `AnnService/src/Core/SPANN/**`

## IO and Cache Ownership
- Core Disk IO abstraction and load/save path live in core, not wrappers.
- SPANN file IO and cache implementation:
  - `AnnService/inc/Core/SPANN/ExtraFileController.h`
  - `AnnService/src/Core/SPANN/SPANNIndex.cpp`
- SPANN cache tuning params:
  - `AnnService/inc/Core/SPANN/ParameterDefinitionList.h`
  - Important keys: `CacheSizeGB`, `CacheShards`

## Wrapper Layer Map
- Main wrapper API surface:
  - Header: `Wrappers/inc/CoreInterface.h`
  - Impl: `Wrappers/src/CoreInterface.cpp`
- Python SWIG interface:
  - `Wrappers/inc/PythonCore.i`
  - Generated file (do not hand-edit): `Wrappers/inc/CoreInterface_pwrap.cpp`
- Python module stubs:
  - `Wrappers/inc/SPTAG.py`, `Wrappers/inc/SPTAGClient.py`

## Build/Save Responsibilities (Quick)
- Wrapper entry for build/save:
  - `Wrappers/src/CoreInterface.cpp` (`AnnIndex::Build*`, `AnnIndex::Save`)
- Actual core build/save implementation:
  - `AnnService/src/Core/VectorIndex.cpp` (`VectorIndex::BuildIndex`, `VectorIndex::SaveIndex`, `VectorIndex::LoadIndex`)

## Service Entrypoints
- Server: `AnnService/src/Server/**`
- Client: `AnnService/src/Client/**`
- Aggregator: `AnnService/src/Aggregator/**`
- Offline builder/search tools: `AnnService/src/IndexBuilder/**`, `AnnService/src/IndexSearcher/**`
- SSD/SPFresh tools: `AnnService/src/SSDServing/**`, `AnnService/src/SPFresh/**`

## Typical Task Routing
- Add/modify search algorithm behavior:
  - Start in `AnnService/inc/Core/VectorIndex.h` and target algorithm folder (`BKT/KDT/SPANN`).
- Modify metadata filtering behavior:
  - Check `VectorIndex` interface and SPANN/BKT/KDT implementations.
  - Wrapper exposure in `Wrappers/inc/CoreInterface.h` and `Wrappers/inc/PythonCore.i`.
- Add Python API:
  - Update `Wrappers/inc/CoreInterface.h/.cpp` and `Wrappers/inc/PythonCore.i`.
  - Rebuild SWIG wrapper and `_SPTAG` module.
- Investigate IO/cache:
  - Use `AnnService/inc/Core/SPANN/ExtraFileController.h` and SPANN options.

## Build Commands (Linux)
- CMake build:
```bash
mkdir -p build && cd build
cmake -DSPDK=OFF -DROCKSDB=OFF ..
make -j
```
- Python wrapper build in place:
```bash
python setup.py build_ext --inplace
```

## Agent Guardrails
- Prefer editing source files, not generated SWIG outputs.
- Avoid changing third-party dependencies under `ThirdParty/` unless task explicitly requires it.
- If modifying wrapper APIs, verify both C++ compile and Python import path.
- Keep cache policy in core SPANN IO controller (`ExtraFileController`) instead of duplicating policy in wrappers.

## Current Repo Notes
- Multi-tenant wrapper logic is in `Wrappers/src/CoreInterface.cpp` (`TenantIndexManager`).
- Keep tenant routing in wrapper, but rely on core IO/cache infrastructure for cache policy.
- Dataset destructor safety patch location:
  - `AnnService/inc/Core/Common/Dataset.h`
