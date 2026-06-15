#pragma once
// Real (extended) RaBitQ flat single-centroid store + per-vector estimator.
// Isolated from rabitqlib: this header exposes a pure API; the implementation
// (RaBitQ2.cpp) is compiled in its own -mavx2 -mfma static library so the rest
// of SPTAG keeps its default ISA flags.
#include <string>

namespace SPTAG
{
    namespace SPANN
    {
        class RaBitQ2
        {
        public:
            RaBitQ2();
            ~RaBitQ2();

            // Load a rabitq2.bin sidecar (magic 'RBQ2'). Returns false if missing/invalid.
            bool Load(const std::string& path);
            // Same as Load but does NOT keep the resident per-vector code store
            // (rotator + centroid + config only). Use when the per-vector codes live
            // elsewhere (e.g. in the posting records) and are passed to EstimateCode.
            bool LoadMeta(const std::string& path);
            bool Loaded() const { return m_loaded; }

            int GetN() const { return m_N; }
            int GetDim() const { return m_dim; }
            int GetExBits() const { return m_exBits; }

            // Per-query context (one per thread). Allocate once, reuse across candidates.
            void* AllocQuery() const;
            void FreeQuery(void* ctx) const;

            // Prepare the query (cosine-normalize, rotate, compute RaBitQ factors).
            // rawQuery: m_dim floats, un-normalized.
            void PrepareQuery(void* ctx, const float* rawQuery) const;

            // Estimated L2 distance in normalized rotated space between the prepared
            // query and vector vid. Monotone with cosine distance -> use directly to
            // screen, then exact-rerank survivors from the full-precision store.
            float Estimate(void* ctx, int vid) const;

            // Same estimate, but reads the per-vector code from a caller-supplied
            // buffer instead of the resident store. Lets the b1 code live IN the
            // posting record (zero resident codes). binCode must point to binBytes
            // bytes; exCode to exBytes (may be null when exBits==0).
            float EstimateCode(void* ctx, const void* binCode, const void* exCode = nullptr) const;

            // Bytes of the per-vector binary / ex codes (for in-posting layout).
            int GetBinBytes() const { return m_binBytes; }
            int GetExBytes() const { return m_exBytes; }

        private:
            bool m_loaded = false;
            int m_N = 0, m_dim = 0, m_padded = 0, m_exBits = 0;
            int m_binBytes = 0, m_exBytes = 0;
            void* m_impl = nullptr;  // Impl* (holds rotator, centroid, packed stores)
        };
    }
}
