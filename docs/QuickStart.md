# SDK quick start

[Documentation index](README.md) | [API contracts](SDK.md) | [Support matrix](SupportMatrix.md)

Use SPTAG inside your own application. Start with in-memory BKT to learn the
interface, then try single-layer SPANN disk postings below. The examples find
Top-K neighbors, map vector rows to business IDs, save the index, and reload
it in a separate process. Both paths keep the output for inspection.
They do not need an embedding service, downloaded dataset, or Node.js.

First follow [Linux installation](LinuxInstallation.md). This guide targets
the documented Ubuntu 24.04 x86-64/AVX2 baseline, Python 3.12, and NumPy 1.26.
The checkout must include the SDK examples on `users/qiazh/doc`; the default
branch does not yet provide this development workflow.

## Python

Build the binding with the **same Python development installation** that will
run it, then add its output directory to the module path:

```bash
cmake --build build --target _SPTAG --parallel 2
export PYTHONPATH="$PWD/Release${PYTHONPATH:+:$PYTHONPATH}"
python3 -c "import SPTAG, numpy; print(SPTAG.__file__, numpy.__version__)"
python3 Examples/python/quickstart.py --index quickstart-output/python
```

The [executable example](../Examples/python/quickstart.py) creates 128
eight-dimensional Float32 vectors. It checks the first neighbor and zero
distance for rows 0, 31, and 63 after saving and reloading. Its final line is:

```text
Python SDK example passed: 3 queries; artifacts retained at <absolute-output-directory>
```

The output contains:

| Artifact | Purpose |
| --- | --- |
| `index/` | The SPTAG index, including its loader configuration and data files |
| `ids.json` | Business IDs in original vector row order |
| `vectors.npy` | Input vectors retained by this example |
| `queries.npy` | Queries retained for another process |

Run another process without rebuilding:

```bash
python3 Examples/python/quickstart.py --index quickstart-output/python \
  --load --queries quickstart-output/python/queries.npy --k 3
```

Each query prints a JSON object with `query` and `neighbors`. Each neighbor
contains `id` (business ID), `row_id` (SPTAG row), and `distance` (squared L2).
For the first demonstration query, rank zero is `document-0`, row 0, distance 0.
Library diagnostic logs also appear; the example is not a JSON-only service.

### Use your own vectors and IDs

Prepare:

- `vectors.npy`: finite `float32` array with shape `(count, dimension)`.
- `queries.npy`: finite `float32` array with shape `(query_count, dimension)`.
- `ids.json`: JSON list of unique, nonempty strings, one per vector row.

For example, these four vectors might represent embeddings of four documents:

```python
import json
from pathlib import Path
import numpy as np

vectors = np.array([[0, 0], [1, 0], [0, 2], [3, 3]], dtype=np.float32)
np.save("vectors.npy", vectors)
np.save("queries.npy", vectors[:1])
Path("ids.json").write_text(
    json.dumps(["doc-A", "doc-B", "doc-C", "doc-D"]), encoding="utf-8"
)
```

Then:

```bash
python3 Examples/python/quickstart.py --index quickstart-output/my-data \
  --vectors vectors.npy --ids ids.json --queries queries.npy --k 3
python3 Examples/python/quickstart.py --index quickstart-output/my-data \
  --load --queries queries.npy --k 3
```

Use a fresh artifact directory when building. The first query's nearest
neighbor should be `doc-A` at distance zero. The example rejects invalid shapes,
dtypes, nonfinite values, mismatched dimensions, invalid IDs, and invalid K.
It does not silently replace a missing or invalid input with demonstration data.
The two-dimensional example is for learning, not recall or performance evaluation.

## SPANN disk-backed search

Use the same built Python binding and driver; no TiKV, SPDK, or additional
dataset is needed:

```bash
python3 Examples/python/quickstart.py --algorithm SPANN --index quickstart-output/spann
python3 Examples/python/quickstart.py --index quickstart-output/spann \
  --load --queries quickstart-output/spann/queries.npy --k 3
```

The first process builds a SPANN index with a BKT head index and writes STATIC
postings to `index/SPTAGFullList.bin_0`. It saves and reloads the index before
searching. The second process loads the existing artifacts without building.
Both return rows 0, 31, and 63 as the three demonstration queries' first
neighbors, with distance zero. The final success line has the same shape as
the BKT example; the saved loader configuration identifies the SPANN algorithm.

The [driver's configuration](../Examples/python/quickstart.py) separates:

| Section | Demonstration settings |
| --- | --- |
| `Base` | BKT head index, L2, isolated index directory |
| `SelectHead` | Execute head selection, ratio 0.25 |
| `BuildHead` | Execute head construction, one worker |
| `BuildSSDIndex` | `Storage=STATIC`, one layer, two replicas, four-page posting limits |
| Search settings in `BuildSSDIndex` | 32 head candidates, one search thread |

`STATIC` is the read-only posting-file path, **not** the dynamic `FILEIO`
backend. This example does not enable insertion, deletion, TiKV, or quantization.
The test checks nonempty on-disk posting files as well as the returned results;
an in-memory fallback is not accepted as success.

The teaching driver reads the full input into NumPy for construction and again
on load for shape/ID checks. This is a driver convenience, not a requirement
that a serving application retain the raw input corpus alongside a loaded
SPANN index. It demonstrates disk search/persistence, not an out-of-core build
or a large-scale memory/recall guarantee. Candidate and head settings are chosen
for this 128-vector corpus, not as production defaults.
For your own corpus, use the driver's `--vectors`, `--ids`, and `--queries`
inputs and evaluate appropriate settings. For file-based construction, continue
to the [SPANN configuration reference](GettingStart.md).

## C++

The [consumer project](../Examples/cpp/CMakeLists.txt) is a standalone CMake
application that adds SPTAG as a source subdirectory and links
`SPTAG::SPTAG`. That target supplies the include directories, C++20 requirement,
OpenMP linkage, and build definitions needed by the public headers.

```bash
cmake -S Examples/cpp -B build-sdk-consumer -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFOLLY=OFF -DTBB=ON -DSPDK=OFF -DROCKSDB=OFF -DTIKV=OFF -DGPU=OFF -DRABITQ=OFF
cmake --build build-sdk-consumer --target sptag_sdk_example --parallel 2
build-sdk-consumer/sptag_sdk_example quickstart-output/cpp
build-sdk-consumer/sptag_sdk_example quickstart-output/cpp --load
```

Both runs must finish with `C++ SDK example passed: 3/3 self-matches; ...`.
They retain `index/` and the row-to-business-ID mapping in `ids.txt`.
The [source](../Examples/cpp/main.cpp) checks each operation's `ErrorCode` and
does not rely on assertions disabled in Release builds.

In your application, replace the example's vector initialization with your own
contiguous `std::vector<float>` containing `count * dimension` values in row
order. Replace the generated `document-N` strings with your business IDs and
replace the demonstration queries/self-match checks with your own query logic.
The raw-pointer API does not check the size of a C++ allocation for you.

Use the same feature flags for the library and consumer. The source-based
consumer is tested; an installed `find_package(SPTAG)` package, binary ABI
compatibility, and native Windows SDK integration are not established by it.
Separate Release build trees currently share the source-root `Release`
directory; do not build incompatible configurations there concurrently.

## Next steps

- Read [API, data, errors, concurrency, and compatibility contracts](SDK.md).
- Adjust [parameters](Parameters.md), measuring recall and latency on your data.
- Use the [CLI example](CLIQuickStart.md) for file-based indexing.
- Follow [dynamic index maintenance](Architecture.md#dynamic-index-maintenance)
  for changing data and its current correctness limits.
- Explore [quantization and native bit-width tuning](RaBitQ_Global_Quantizer.md).
- Distinguish [distributed posting storage](TiKV-Integration.md) from
  [multi-host index construction](Architecture.md#construction-and-multi-layer-indexes).
- Check the [support matrix](SupportMatrix.md) before promoting any new profile.
