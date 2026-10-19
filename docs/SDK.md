# C++ and Python SDK contracts

[Quick start](QuickStart.md) | [Parameters](Parameters.md) | [Support matrix](SupportMatrix.md)

The developer baseline is **in-memory BKT/KDT, Float vectors, static build,
search, save, and load** on the documented Linux configuration. It is an
embedded library, not an embedding generator or a hosted database.

## Choose an index

Start with BKT to follow the executable examples. KDT uses the same SDK
build/search/persistence API: use `"KDT"` in Python or `IndexAlgoType::KDT`
in C++. Compare recall, latency, build time, and memory on representative
data before choosing; neither is universally best.
If the dataset does not fit memory, evaluate SPANN through its separate
[configuration guide](GettingStart.md) rather than assuming the in-memory
example provides disk-backed storage.

## Python API

Build the `_SPTAG` CMake target and import the generated `SPTAG` module from
the build output as shown in [installation](LinuxInstallation.md).
Use the Python interpreter matching CMake's discovered development files.
`import SPTAG` in this source workflow is not an instruction to install an
unrelated or older package from PyPI.

| Operation | Public API | What to check |
| --- | --- | --- |
| Create | `SPTAG.AnnIndex("BKT", "Float", dimension)` | Valid algorithm, positive dimension; construct before setting parameters |
| Configure | `SetBuildParam(name, value, "Index")` | Names and values are strings; this legacy setter returns no status |
| Build | `Build(vectors, count, False)` | Boolean return must be true |
| Search | `Search(query, k)` | Four lists: row IDs, distances, metadata, relaxed-monotonicity flags |
| Save | `Save(directory)` | Boolean return must be true; use a new destination |
| Load | `SPTAG.AnnIndex.Load(directory)` | Check `ReadyToServe()` before searching |
| Search metadata | `SearchWithMetaData(query, k)` | Use an index built with metadata; metadata entries are bytes |

For ordinary `Search`, metadata entries are `None`; they are not business IDs.
The fourth list is an advanced search flag, not a confidence or probability.
The baseline example sets L2 explicitly; the in-memory index default is Cosine.

```python
if not index.Save("my-index"):
    raise RuntimeError("Index save failed")
loaded = SPTAG.AnnIndex.Load("my-index")
if not loaded.ReadyToServe():
    raise RuntimeError("Index load failed")
row_ids, distances, metadata, relaxed = loaded.Search(query, 3)
```

The native wrapper consumes the Python buffer protocol, not NumPy's semantic
dtype information. Before calling it directly, validate:

- Vectors are a finite, C-contiguous `np.float32` matrix of shape `(count, dimension)`.
- Each `Search` query is a finite, contiguous `np.float32` vector of shape `(dimension,)`.
- Count and dimension agree with the buffer; `1 <= k <= count` for the static example.

`np.ascontiguousarray` fixes strides, **not** a wrong dtype or dimension.
The executable example validates these conditions before crossing into native
code. Do not assume every invalid buffer causes a Python exception; several
legacy methods only return false or log errors. The binding is not a general
validation layer for untrusted input.

## C++ API and CMake integration

Use the [standalone consumer](../Examples/cpp/CMakeLists.txt) as the integration
template. With SPTAG added as a source subdirectory:

```cmake
add_subdirectory(path/to/SPTAG sptag-build EXCLUDE_FROM_ALL)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE SPTAG::SPTAG)
```

Include `inc/Core/VectorIndex.h` and use:

| Operation | Public API |
| --- | --- |
| Create | `VectorIndex::CreateInstance(IndexAlgoType::BKT, VectorValueType::Float)` |
| Configure | `index->SetParameter("DistCalcMethod", "L2")` |
| Build | `index->BuildIndex(data, count, dimension)` |
| Search | `QueryResult result(query, k, false); index->SearchIndex(result)` |
| Save | `index->SaveIndex(directory)` |
| Load | `VectorIndex::LoadIndex(directory, index)` |

Check a non-null factory result and compare each operation's return value to
`SPTAG::ErrorCode::Success`. Read neighbors through `result.GetResult(rank)`:
`VID` is the internal row ID; `Dist` is the distance.

For this overload, `BuildIndex` defaults to copying input storage
(`p_shareOwnership=false`). The input must contain `count * dimension` values.
Keep query storage alive through the search call and while using a query result
that refers to it. The example uses one query result per query. Do not turn on
shared ownership without reviewing the concrete index's lifetime requirements.

This source-based CMake target is not an installed package/export or a stable
binary ABI. Keep compiler, feature definitions, and linked dependencies matched;
in particular, `LARGEVID` changes ID types.

## IDs, distances, and approximation

- IDs in the baseline are zero-based vector rows, not external document IDs.
  Save a row-ordered business-ID sidecar with the index, as the examples do.
- Back up, move, and version the ID mapping with the entire index directory.
  The Python example also retains vectors/queries for demonstration and validation;
  those `.npy` files are not required by the underlying `AnnIndex.Load` API.
- The sidecar approach is demonstrated for static indexes. Deletion, refinement,
  or rebuilding can require remapping; do not apply a stale sidecar to a changed index.
- For Float/L2, distance is **squared Euclidean distance**; smaller is closer.
  It is not a similarity score or a probability.
- Results are approximate. Exact self-match checks establish persistence and API
  wiring, not recall on arbitrary queries. Tune `MaxCheck` and evaluate against
  ground truth for your workload.

## Errors and concurrency

Fail the application operation when build/save/load fails; do not search an
unready index or report successful initialization based only on process exit.
C++ status codes, Python booleans, and diagnostic logs are not interchangeable.
The Python parameter setters do not report status, so use documented keys and
verify behavior through tests rather than treating a setter call as acceptance.

The SDK examples are sequential. They do not establish simultaneous mutation,
search/save consistency, Python thread parallelism, or a latency SLA. Serialize
build/update/delete/save operations in an application until its concurrency
requirements have been validated. Do not share mutable query/result buffers.

Dynamic search has a known deleted/live visibility correctness regression.
It is **outside the static developer baseline and blocks treating the affected
update/delete paths as stable SDK capabilities**. See
[known validation limits](SupportMatrix.md#known-validation-limits).

## Versions and persistence

This is a development snapshot, not a published SDK release with a package or
ABI compatibility policy. Record the source commit, feature flags, compiler,
Python/NumPy versions, and embedding model/dimension with your artifacts.
Use a matching build when loading them; retain source vectors and business IDs
so indexes can be rebuilt when changing incompatible versions or formats.
No cross-version or cross-`LARGEVID` interchangeability is promised.

Before a versioned SDK release, the remaining delivery work includes tested
Python packages, installable CMake exports, a supported-platform/version matrix,
upgrade compatibility tests, and release gates for every advertised capability.
See [actual validation](Validation.md); source examples alone are not completion
of those release requirements.
