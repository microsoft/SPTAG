# Official RaBitQ postings

`PostingQuantizer=RaBitQOfficial` uses the Apache-2.0
`VectorDB-NTU/RaBitQ-Library` split-code estimator. It is the recommended
RaBitQ integration for SPANN.

The BKT head graph remains raw Float vectors. During `BuildSSDIndex`, SPTAG
trains or loads the official FHT/Kac rotation and centroid, then encodes each
posting record as the official `[bin | ex]` split representation. Queries stay
raw Float vectors; the search path prepares one rotated query context and uses
the official asymmetric estimator for every posting. This path does not
quantize the query, reconstruct data vectors, or use reranking.

For 128-dimensional vectors and `PostingQuantBits=3`, the data code is 68
bytes:

```text
bin: 16-byte binary code + 12-byte factors
ex:  32-byte extended code + 8-byte factors
```

The model is trained from the first
`PostingQuantizerTrainingSamples` raw vectors when
`PostingQuantizerFile` does not exist; subsequent builds and loads use that
persisted model.

Both `Storage=FILEIO` and `Storage=STATIC` are supported. Static postings use
an 80-byte aligned physical record (`VID`, padding, 68-byte code, tail padding)
so the official binary-code words stay aligned. Static official postings do not
support delta encoding, posting-list rearrangement, compression, or reranking.
Official RaBitQ requires an AVX2/FMA CPU and Float dimensions from 64 through
4095. Online insertion, reassignment, refinement, WAL recovery, and
static-to-FileIO conversion are intentionally rejected for this mode.

Use `Script_AE/iniFile/build_SPANN_sift1m_official_rabitq3_posting.ini` for a
fresh FileIO SIFT1M build, or
`Script_AE/iniFile/build_SPANN_sift1m_official_rabitq3_posting_static.ini` for
the static path. Both use raw Float heads, 3-bit official postings, raw-query
ADC estimation, and `Rerank=0`.

The static build INI is deliberately build-only. Run one of
`search_SPANN_sift1m_official_rabitq3_static_n*.ini` in a new process for
reproducible recall/QPS measurements.

The SIFT1M FileIO and static configs pin `BKTLambdaFactor=0.01` for both
SelectHead and BuildHead. This matches the validated vanilla raw-head
baseline; leaving BuildHead at `-1` lets its independent auto-selection choose
a different lambda and materially changes the recall curve.
