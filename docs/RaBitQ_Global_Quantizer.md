# Global RaBitQ quantizer

Global RaBitQ is an `IQuantizer` implementation backed by the official
full-code quantizer, compact-code packer, and asymmetric distance estimator.
Base vectors use `quantize_full_single` and `packing_rabitqplus_code`; ADC
queries are evaluated with `full_est_dist` and the official SIMD inner-product
kernel selected for the configured bit width. Query-only distance factors are
computed once when the ADC query buffer is prepared, not once per candidate.

The SDC compatibility path reconstructs its query code with the official
`reconstruct_vec` API before invoking the same official asymmetric estimator.
Online search with `EnableADC=true` does not reconstruct base vectors.

Graph construction/refinement and posting replica selection compare **two stored
codes**, not a query buffer and a code. They use
`VectorIndex::ComputeDistanceBetweenStoredVectors` and the quantizer's explicit
`L2DistanceSDC` entry point, independently of `EnableADC`. Do not pass a stored
code as the first operand of the ADC query-distance API or toggle a shared
quantizer's mode inside a parallel build. Quantized indexes previously built
with ADC enabled must have their head graph and postings rebuilt to correct
these comparisons; the model and encoded base vectors can be reused.

Train the model and encode Float base vectors with `Release/quantizer`:

```bash
Release/quantizer \
  -d 128 -v Float -f XVEC \
  -i sift/sift_base.fvecs \
  -o sift/sift_base.rabitq3.u8bin \
  -oq sift/official_rabitq3_global.bin \
  -qt RaBitQQuantizer -qd 3 -ts 1000000
```

The encoded vectors are stored in a `UInt8` container, but their code payload is
packed at the configured bit width. Five Float factors follow each packed code,
so 128-dimensional 3-bit SIFT vectors use `Dim=68`
(`128 * 3 / 8 + 5 * sizeof(float)`). Configure the encoded base-vector file and
`QuantizerFilePath` in the regular SPANN build configuration. Queries remain
raw Float vectors and are prepared through the loaded global `IQuantizer`.

The official compact kernels require AVX2/FMA or AVX512 and pad dimensions to a
multiple of 64. The current adapter supports the official L2 estimator; cosine
distance is intentionally unsupported.

## Optional local residual quantization

The default remains a single global centroid and the upstream fast quantizer.
For higher accuracy at a fixed bit width, supply Float XVEC centroids trained
from base vectors only:

```bash
Release/quantizer \
  -d 128 -v Float -f XVEC \
  -i sift/sift_base.fvecs -o sift/sift_base.rabitq7-local.u8bin \
  -oq sift/rabitq7-local.bin -qt RaBitQQuantizer -qd 7 \
  -rc sift/centroids.fvecs -ts 1000000
```

Use a **new** model and output path. `-rc` is rejected when the model already
exists; omit it when reusing a saved local model. Centroids must have the input
dimension, contain finite values, and number between 1 and 65,536. They use the
model's input normalization and persisted rotation. The C++ equivalent is
`SetLocalCentroids` on a trained rotated `RaBitQQuantizer`; bit-width clones
retain those centers.

Each base vector is assigned to its nearest center and its residual is encoded
with upstream per-vector optimal scaling, rather than the shared expected scale.
The model stores the centers; each code appends a `uint32` center ID after the
existing packed payload and five Float factors. For 128-dimensional 7-bit
vectors this is **136 bytes** (112 + 20 + 4), versus 132 for global quantization.
Set the index's `Dim` to that encoded width. Center storage is additional to the
7-bit payload; the bit count is not a claim about total index size.

ADC query preparation computes the query norm relative to every center once,
alongside the rotated query and existing factors. Its buffer uses
`4 * (paddedDimension + 2 + centerCount)` bytes. Scoring selects the matching
center's norm without fetching or reranking raw base vectors. SDC and vector
reconstruction likewise use each code's stored center.

Local models use **version 4**: the version-3 payload is followed by a `uint32`
center count and row-major rotated Float centers. Version-2/3 models keep their
existing encoding, scoring, serialization, and fast quantization behavior.
Never replace a model beneath an existing encoded corpus/index: re-encode and
build a separate index when enabling or changing local centers.

Local quantization reduces distance-estimation error, but does not guarantee
a fixed recall. Search candidate coverage remains an independent requirement;
validate the complete query set and report any changed search parameters.

`Script_AE/iniFile/build_SPANN_sift1m_rabitq3_global.ini` is the canonical
SIFT1M example. It uses STATIC postings containing the global RaBitQ codes.

STATIC search workspaces must use one asynchronous read request per posting,
pointing to the start of that posting's buffer. Dynamic storage instead uses
one request per page. Using the dynamic layout for STATIC reads can overlap
posting data and write past the buffer, causing corrupt IDs, repeated hash-table
expansion, and heap corruption. Workspace initialization selects the layout from
the storage type and refreshes it when a reused workspace changes layouts.
This does not change the index format or require rebuilding an existing index.
