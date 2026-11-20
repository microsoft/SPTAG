# RaBitQ implementation and compatibility

[User guide](../RaBitQ_Global_Quantizer.md) | [Documentation index](../README.md)

This developer record preserves implementation constraints and migration notes.
Historical descriptions of fixes are not a claim that all optional toolchains
or persisted indexes have been revalidated on this branch.

Global RaBitQ is an `IQuantizer` implementation backed by the official
full-code quantizer, compact-code packer, and asymmetric distance estimator.
Base vectors use `quantize_full_single` and `packing_rabitqplus_code`; ADC
queries are evaluated with `full_est_dist` and the official SIMD inner-product
kernel selected for the configured bit width. Query-only distance factors are
computed once when the ADC query buffer is prepared, not once per candidate.

The SDC compatibility path reconstructs its query code with the official
`reconstruct_vec` API before invoking the same official asymmetric estimator.
Online search with `EnableADC=true` does not reconstruct base vectors.

Use `SetQuantizerADC(true)` on the SPANN index to enable ADC at runtime and
persist `EnableADC=true` for save/load and cloning. Calling
`GetQuantizer()->SetEnableADC(true)` directly changes only the live quantizer,
not the index configuration. Keep this setting fixed when measuring incremental
recall: with ADC disabled, raw queries are quantized before distance estimation.
Previously, the SPANN setter also changed only the live mode, so an ADC initial
measurement could silently become SDC after the benchmark reloaded or cloned
the index. Attaching a quantizer now also applies the configured mode, including
when model loading follows configuration loading. For existing indexes, set
the intended mode explicitly and save it;
the quantizer model and stored codes do not need conversion.

Graph construction/refinement and posting replica selection compare **two stored
codes**, not a query buffer and a code. They use
`VectorIndex::ComputeDistanceBetweenStoredVectors` and the quantizer's explicit
`L2DistanceSDC` or `InnerProductDistanceSDC` entry point, independently of `EnableADC`. Do not pass a stored
code as the first operand of the ADC query-distance API or toggle a shared
quantizer's mode inside a parallel build. Quantized indexes previously built
with ADC enabled must have their head graph and postings rebuilt to correct
these comparisons; the model and encoded base vectors can be reused.

RaBitQ's stored-code estimator is directional, unlike PQ's symmetric codebook
L2 distance. Split reassign checks must compare the current, old, and new heads
with the data vector as the first operand throughout. The reassign predicate
accepts the current head vector and computes this distance itself, rather than
accepting a caller-provided scalar that may have the opposite direction.
This does not change the estimator, centroid, model, or version-update rules.

Local reassign version updates separately enforce `expectedOld` with atomic
compare-and-swap. An outdated task must not advance a newer version, and a
failed update must not change the posting's version or append its selected
replicas. Both batched split reassign and individual reassign check the update
result. Unconditional increments (`expectedOld=0xff`) and 7-bit wraparound
remain supported; the existing TiKV idempotent-update behavior is unchanged.

For training and index configuration commands, use the
[user guide](../RaBitQ_Global_Quantizer.md#train-and-encode).

The encoded vectors are stored in a `UInt8` container, but their code payload is
packed at the configured bit width. Six Float factors follow each new packed code:
the five existing L2/reconstruction factors and the official IP additive factor.
The global stored-code size is `paddedDimension * bits / 8 + 6 * sizeof(float)`.
Queries remain raw Float vectors and are prepared through the loaded `IQuantizer`.

The official compact kernels require AVX2/FMA or AVX512 and pad dimensions to a
multiple of 64. The adapter supports L2 and InnerProduct; cosine distance is
intentionally unsupported. Measurement protocols and results belong to the
[normalized RaBitQ4 IP evaluation](../../evaluation/2026-11-18-sift10m-rabitq-ip/normalized-ip.md).

## InnerProduct distance

`DistCalcMethod=InnerProduct` means **1 - dot(query, base)**, with smaller
scores preferred. Negative distances are valid. Use a model trained with
`normalize=false`; normalizing base vectors changes maximum-inner-product
ranking into a different problem. Only the corrected six-factor format is
supported: version 6 for global models and version 7 for local models.
Earlier model versions are rejected for both L2 and IP.

For the rotated residual `r = x - c`, let `u` be the unpacked integer code
minus its midpoint `(2^bits - 1)/2`. The stored L2 factor is
`f_rescale = -2 * ||r||^2 / dot(r,u)`. The official IP estimator is:

```text
ip_add = 1 - dot(r,c) + ||r||^2 * dot(c,u) / dot(r,u)
distance = ip_add - dot(q,c) + 0.5 * f_rescale * dot(q,u)
```

The encoder derives `ip_add = 1 - dot(r,c) + (f_add_L2 - ||r||^2)/2`
from the original residual and the existing L2 factor, without repeating
quantization. The original packed bits, L2 factors, and reconstruction are
unchanged. The extra scalar is precomputed during encoding; no raw-vector
reranking is needed at query time.

The previous IP path used `ip_add=1`, omitting the centroid correction.
For estimator residual error `e`, that path had error `-dot(q,e)` rather than
the official `-dot(q-c,e)`. Historical IP measurements using that path do not
establish the accuracy of the corrected estimator. The correction does not
make IP a metric or guarantee that posting splits converge.

The official packed-code inner-product kernel evaluates `dot(q,u)`; base
vectors are not reconstructed or fetched for ADC. Zero residuals have
`f_rescale=0`, so their score is exactly `1-dot(q,c)`. Query-to-centroid dot
products are computed once during query preparation. SDC reconstructs only
its first operand and uses the same estimator, independently of the ADC flag.
The SDC estimator remains directional.

`IQuantizer` declares pure virtual InnerProduct entry points, so each concrete
quantizer must implement them. PQ explicitly reuses its legacy cosine-named
methods, which evaluate `m_base-dot` without normalization for IP-configured
tables; OPQ inherits this implementation. RaBitQ supplies its own estimator
without enabling its unsupported Cosine API. Both `VectorIndex` stored-code
comparisons and BKTree clustering dispatch the requested metric. Auto-tuning
uses the same metric for ground truth and candidate scoring, and rejects
Cosine rather than silently evaluating candidates with L2.

This review baseline retains the original BKTree single-cluster behavior and
SPANN split/merge/reassignment from `059c99ab`. The only BKTree change is IP
SDC dispatch; no special IP leaves or forced balanced split are included.
The evaluation explicitly normalizes both base and query vectors before
training and encoding, while keeping quantizer-side `normalize=false`.
Its initial1M result is not evidence of dynamic maintenance safety.

The corrected code adds four bytes per vector: global 128-dimensional 4-bit
codes grow from 84 to 88 bytes. Existing codes do not contain the exact residual
information needed to recover the correction. Old models cannot be loaded by
this implementation: train a new model, re-encode the original vectors, and
rebuild a separate index. Never resume an old index using a corrected model.
For supported models, `CloneWithBits` retains rotation and centroids while
changing bit width. A graph/posting index must also be rebuilt when changing
its distance metric.

## ADC query buffer layout

L2 and IP share one transient layout. With padded dimension `D` and local
centroid count `N`, the following Float blocks appear in order:

| Block | Float count |
| --- | --- |
| Rotated, padded query | D |
| Global squared L2 distance and query-sum factor | 2 |
| Squared L2 distances to local centroids | N |
| Inner product with the global centroid | 1 |
| Inner products with local centroids | N |

Allocate with `QuantizeSize()`, not a hard-coded length: ADC needs
`sizeof(float) * (D + 3 + 2*N)` bytes; global models have `N=0`.
Both sets of factors are currently prepared even for L2-only queries.
Compared with the old L2 layout, IP adds `1+N` floats and centroid dot products
per query preparation, not per candidate. This does not change stored-code size.
SDC takes stored codes instead of this query buffer.

## Rotation and model compatibility

New models apply the same random orthogonal rotation to training vectors,
centroids, encoded vectors, and ADC queries. Reconstruction applies the inverse
transform, and bit-width clones retain the original rotation.

Global models use version 6 and local models use version 7, both with a stored
rotation and six-factor codes. There is no old-format compatibility path.
Versions 2/3/4/5 and unknown versions fail explicitly at model loading, with
instructions to retrain, re-encode, and rebuild. Do not replace only the model
file under an existing index.

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
packed payload and six Float factors. For 128-dimensional 7-bit
vectors this is **140 bytes** (112 + 24 + 4), versus 136 for global quantization.
Set the index's `Dim` to that encoded width. Center storage is additional to the
7-bit payload; the bit count is not a claim about total index size.

ADC scoring selects the code's center-specific factors from the
[query buffer](#adc-query-buffer-layout), without fetching or reranking raw
base vectors. SDC and reconstruction likewise use each code's stored center.

New local models use **version 7**: the rotated-global payload is followed by a
`uint32` center count and row-major rotated Float centers. Old local version 4
is not accepted; its codes lack the IP correction factor.
Never replace a model beneath an existing encoded corpus/index: re-encode and
build a separate index when enabling or changing local centers.

Local quantization reduces distance-estimation error, but does not guarantee
a fixed recall. Search candidate coverage remains an independent requirement;
validate the complete query set and report any changed search parameters.

STATIC search workspaces must use one asynchronous read request per posting,
pointing to the start of that posting's buffer. Dynamic storage instead uses
one request per page. Using the dynamic layout for STATIC reads can overlap
posting data and write past the buffer, causing corrupt IDs, repeated hash-table
expansion, and heap corruption. Workspace initialization selects the layout from
the storage type and refreshes it when a reused workspace changes layouts.
This does not change the index format or require rebuilding an existing index.
