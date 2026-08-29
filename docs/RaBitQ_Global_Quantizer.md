# Global RaBitQ quantizer

Global RaBitQ is an `IQuantizer` implementation backed by the official
full-code quantizer, compact-code packer, and asymmetric distance estimator.
Base vectors use `quantize_full_single` and `packing_rabitqplus_code`; ADC
queries are evaluated with `full_est_dist` and the official SIMD inner-product
kernel selected for the configured bit width. Query-only distance factors are
computed once when the ADC query buffer is prepared, not once per candidate.
The adapter also applies the official `FhtKacRotator` to base vectors, query
vectors, and the learned centroid before quantization or distance estimation,
and persists the sampled rotator state in the model.

The SDC compatibility path reconstructs its query code with the official
`reconstruct_vec` API before invoking the same official asymmetric estimator.
Online search with `EnableADC=true` does not reconstruct base vectors.

Train the model and encode Float base vectors with `Release/quantizer`:

```bash
Release/quantizer \
  -d 128 -v Float -f XVEC \
  -i sift/sift_base.fvecs \
  -o sift/sift_base.rabitq3.u8bin \
  -oq sift/official_rabitq3_global.bin \
  -qt RaBitQQuantizer -qd 3 -ts 1000000 \
  -m L2 -rqm exact
```

The encoded vectors are stored in a `UInt8` container, but their code payload is
packed at the configured bit width. Five Float factors follow each packed code,
so 128-dimensional 3-bit SIFT vectors use `Dim=68`
(`128 * 3 / 8 + 5 * sizeof(float)`). Configure the encoded base-vector file and
`QuantizerFilePath` in the regular SPANN build configuration. Queries remain
raw Float vectors and are prepared through the loaded global `IQuantizer`.

The official compact kernels require AVX2/FMA or AVX512 and pad dimensions to a
multiple of 64. For dimensions that are not already multiples of 64, the stored
code size is based on the padded dimension. The quantizer CLI accepts
`-rqm exact|fast`: `exact` is the default and preserves the official
configuration (`RabitqConfig{t_const=-1}`), while `fast` uses
`faster_config(...)`. Metric selection is `-m L2|Cosine|InnerProduct`; cosine
normalizes vectors and routes estimation through the official `METRIC_IP`
path, while `InnerProduct` uses that path without normalization. The existing
`IQuantizer` selector routes both inner-product metrics through
`CosineDistance`.

Models now store the metric, quantization mode, and rotator state in version 3
of the RaBitQ format. Older version-2 models are rejected because they do not
persist the official rotator state; existing codes cannot be migrated to a
rotated v3 model, so re-encoding and index rebuild are required.
Reconstruction remains available for debugging and fallback flows in the
original vector space. Because the upstream rotator API does not expose an
inverse transform, the adapter materializes the rotator as a black-box linear
map by rotating each original-space basis vector once and then applies the
transpose of that learned transform during reconstruction.
When the quantizer CLI reuses an existing RaBitQ model, omitting `-qd` trusts
the persisted bit width, and encoding follows the model’s persisted
normalization behavior without an extra outer normalization pass.

`RaBitQQuantizer` also exposes the split-code API used by STATIC SPANN posting
quantization. It reports the official split layout sizes via
`GetSplitCodeLayout()`, quantizes a base vector against a caller-provided local
centroid with `QuantizeSplitVector(...)`, prepares a reusable per-query context
with `PrepareSplitQueryContext(...)`, and estimates 1-bit or full-B-bit local
distances with `EstimateSplitDistance(...)`. These entry points reuse the
official split-code layout helpers, `quantize_split_single`, `SplitSingleQuery`,
`split_single_estdist`, and `split_single_fulldist`. For cosine, raw data and
query vectors are normalized before rotation, while caller-provided local
centroids are only padded and rotated. Fast query preparation follows the
official 4-bit `SplitSingleQuery` preprocessing config.

The batch split adapter reports the official `fastscan::kBatchSize` and exact
`BatchDataMap<float>`/`ExDataMap<float>` byte components through
`GetSplitBatchLayout()`. `QuantizeSplitBatch(...)` accepts up to one official
batch of contiguous Float vectors, duplicates the final valid vector into
unused lanes, and returns the valid count. `PrepareSplitBatchQueryContext(...)`
reuses the loaded rotator and caller-provided local centroid;
`EstimateSplitBatchDistances(...)` returns distance, lower/upper/error bound,
and intermediate-inner-product arrays for valid lanes; and
`BoostSplitBatchDistance(...)` applies all extended bits to one indexed lane.
These paths call the pinned official `quantize_split_batch`,
`SplitBatchQuery`, `split_batch_estdist`, and `split_distance_boosting` APIs.

## STATIC SPANN posting RaBitQ

STATIC SPANN can also quantize posting payloads with a v3 RaBitQ model while
keeping the base vectors and head vectors as raw `Float`. Set
`PostingQuantizer=RaBitQ` and `PostingQuantizerFile=<official_v3_model>` in
`[BuildSSDIndex]`, and leave `QuantizerFilePath` empty. The current STATIC path
stores each record as `[VID | split binary blob | split extended blob]`, where
the split layout comes directly from `GetSplitCodeLayout()` and each posting
uses its own head vector as the local centroid. Build persists an index-local
copy of that exact posting model and binds the posting files to its serialized
fingerprint, so reload uses the copied model and rejects same-shape
replacements without re-encoding.

This path rejects non-Float vectors, global quantizers, dimension/metric
mismatches, delta encoding, posting rearrangement, and `Rerank>0`. Search uses
adaptive official lower-bound/full-bit boosting: it evaluates the official
1-bit lower bound first, skips candidates only when their lower bound exceeds
the maximum upper bound retained by the current top-K, and only then runs the full-bit estimate. The binary and
extended bytes are still stored together in the current file layout, so the
incremental benefit is compute-side pruning rather than separate ex-bit I/O.

`PostingQuantizer=RaBitQBatch` enables the version-3 batch posting format
without changing SPANN head selection or posting routing. Each posting computes
and persists an arithmetic-mean quantization centroid independent of its
navigational head. The main posting file contains IDs and the official
`BatchDataMap<float>` payload; extended `ExDataMap<float>` records are stored in
an index-local `.rabitq.ext` sidecar and fetched only for batches with surviving
candidates. This makes the initial posting read independent of the extended
bits and uses the official `SplitBatchQuery`, `split_batch_estdist`, and
`split_distance_boosting` paths.

Set `PostingRaBitQRerank=N` to persist a `.rabitq.raw` sidecar and rerank the
best `N` estimated candidates with exact raw-vector distances. `WithVec` is
supported only when this raw sidecar is enabled. Batch postings support `L2`,
normalized `Cosine`, and unnormalized `InnerProduct`; sidecar reads are included
in reported disk I/O statistics. Batch files and sidecars are versioned,
fingerprinted, checked for truncation/overlap, and copied with the index.

`Script_AE/iniFile/build_SPANN_sift1m_rabitq3_global.ini` remains the canonical
global-quantizer SIFT1M example. For raw-Float STATIC postings with local
RaBitQ codes, use `Script_AE/iniFile/build_SPANN_sift1m_raw_static_rabitq3.ini`.
For official batch scanning and split extended-code I/O, use
`Script_AE/iniFile/build_SPANN_sift1m_raw_static_rabitq3_batch.ini`.
