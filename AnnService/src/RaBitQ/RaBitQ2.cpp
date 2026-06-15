// Real (extended) RaBitQ store + estimator. Compiled with -mavx2 -mfma and the
// vendored rabitqlib headers (SPTAG/ThirdParty/RaBitQ). See RaBitQ2.h.
#include "inc/Core/SPANN/RaBitQ2.h"

#include <vector>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdint>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

namespace SPTAG
{
    namespace SPANN
    {
        using namespace rabitqlib;

        struct RaBitQ2Impl
        {
            Rotator<float>* rot = nullptr;
            std::vector<float> centroid_rot;       // padded
            std::vector<char> binStore;            // N * binBytes
            std::vector<char> exStore;             // N * exBytes (empty if exBits==0)
            size_t padded = 0;
            size_t exBits = 0;
            size_t binBytes = 0;
            size_t exBytes = 0;
            ex_ipfunc ip_func = nullptr;
            quant::RabitqConfig qcfg;              // query-side config

            ~RaBitQ2Impl() { delete rot; }
        };

        struct RaBitQ2QueryCtx
        {
            std::vector<float> rq;                 // rotated (stable storage for the query)
            SplitSingleQuery<float>* q = nullptr;
            ~RaBitQ2QueryCtx() { delete q; }
        };

        RaBitQ2::RaBitQ2() {}
        RaBitQ2::~RaBitQ2()
        {
            delete reinterpret_cast<RaBitQ2Impl*>(m_impl);
            m_impl = nullptr;
        }

        bool RaBitQ2::Load(const std::string& path)
        {
            std::ifstream in(path, std::ios::binary);
            if (!in) return false;
            int32_t magic = 0, N = 0, dim = 0, pdim = 0, ex = 0, rtype = 0, rbytes = 0;
            in.read((char*)&magic, 4);
            if (!in || magic != 0x52425132) return false;  // 'RBQ2'
            in.read((char*)&N, 4);
            in.read((char*)&dim, 4);
            in.read((char*)&pdim, 4);
            in.read((char*)&ex, 4);
            in.read((char*)&rtype, 4);
            in.read((char*)&rbytes, 4);
            if (!in || N <= 0 || dim <= 0 || pdim <= 0) return false;

            auto* impl = new RaBitQ2Impl();
            impl->padded = (size_t)pdim;
            impl->exBits = (size_t)ex;

            std::vector<char> rotBytes(rbytes);
            in.read(rotBytes.data(), rbytes);
            impl->rot = choose_rotator<float>(dim, RotatorType::FhtKacRotator, (size_t)pdim);
            if (impl->rot->size() != (size_t)pdim) { delete impl; return false; }
            impl->rot->load(rotBytes.data());

            impl->centroid_rot.resize(pdim);
            in.read((char*)impl->centroid_rot.data(), (size_t)pdim * sizeof(float));

            impl->binBytes = BinDataMap<float>::data_bytes(impl->padded);
            impl->exBytes = ExDataMap<float>::data_bytes(impl->padded, impl->exBits);

            impl->binStore.resize((size_t)N * impl->binBytes);
            if (impl->exBytes) impl->exStore.resize((size_t)N * impl->exBytes);

            for (int i = 0; i < N; i++)
            {
                in.read(&impl->binStore[(size_t)i * impl->binBytes], impl->binBytes);
                if (impl->exBytes)
                    in.read(&impl->exStore[(size_t)i * impl->exBytes], impl->exBytes);
            }
            if (!in) { delete impl; return false; }

            impl->ip_func = select_excode_ipfunc(impl->exBits);
            impl->qcfg = quant::faster_config(impl->padded, SplitSingleQuery<float>::kNumBits);

            m_impl = impl;
            m_N = N; m_dim = dim; m_padded = pdim; m_exBits = ex;
            m_binBytes = (int)impl->binBytes;
            m_exBytes = (int)impl->exBytes;
            m_loaded = true;
            return true;
        }

        bool RaBitQ2::LoadMeta(const std::string& path)
        {
            std::ifstream in(path, std::ios::binary);
            if (!in) return false;
            int32_t magic = 0, N = 0, dim = 0, pdim = 0, ex = 0, rtype = 0, rbytes = 0;
            in.read((char*)&magic, 4);
            if (!in || magic != 0x52425132) return false;  // 'RBQ2'
            in.read((char*)&N, 4);
            in.read((char*)&dim, 4);
            in.read((char*)&pdim, 4);
            in.read((char*)&ex, 4);
            in.read((char*)&rtype, 4);
            in.read((char*)&rbytes, 4);
            if (!in || N <= 0 || dim <= 0 || pdim <= 0) return false;

            auto* impl = new RaBitQ2Impl();
            impl->padded = (size_t)pdim;
            impl->exBits = (size_t)ex;

            std::vector<char> rotBytes(rbytes);
            in.read(rotBytes.data(), rbytes);
            impl->rot = choose_rotator<float>(dim, RotatorType::FhtKacRotator, (size_t)pdim);
            if (impl->rot->size() != (size_t)pdim) { delete impl; return false; }
            impl->rot->load(rotBytes.data());

            impl->centroid_rot.resize(pdim);
            in.read((char*)impl->centroid_rot.data(), (size_t)pdim * sizeof(float));
            if (!in) { delete impl; return false; }

            // NOTE: per-vector bin/ex codes are intentionally NOT read into RAM here;
            // they live in the posting records and are supplied to EstimateCode().
            impl->binBytes = BinDataMap<float>::data_bytes(impl->padded);
            impl->exBytes = ExDataMap<float>::data_bytes(impl->padded, impl->exBits);
            impl->ip_func = select_excode_ipfunc(impl->exBits);
            impl->qcfg = quant::faster_config(impl->padded, SplitSingleQuery<float>::kNumBits);

            m_impl = impl;
            m_N = N; m_dim = dim; m_padded = pdim; m_exBits = ex;
            m_binBytes = (int)impl->binBytes;
            m_exBytes = (int)impl->exBytes;
            m_loaded = true;
            return true;
        }

        void* RaBitQ2::AllocQuery() const
        {
            auto* ctx = new RaBitQ2QueryCtx();
            ctx->rq.resize(m_padded);
            return ctx;
        }

        void RaBitQ2::FreeQuery(void* ctx) const
        {
            delete reinterpret_cast<RaBitQ2QueryCtx*>(ctx);
        }

        void RaBitQ2::PrepareQuery(void* ctxv, const float* rawQuery) const
        {
            auto* impl = reinterpret_cast<RaBitQ2Impl*>(m_impl);
            auto* ctx = reinterpret_cast<RaBitQ2QueryCtx*>(ctxv);

            // Match the encoder's normalization choice. Default: cosine-normalize (DB
            // vectors were normalized before encoding). SPTAG_RABITQ_NONORM=1 -> true L2
            // (no normalization on either side), for L2-trained sidecars.
            static const bool s_noNorm = []() {
                const char* e = std::getenv("SPTAG_RABITQ_NONORM"); return e && e[0] == '1';
            }();
            std::vector<float> qn(m_dim);
            double n = 0;
            for (int i = 0; i < m_dim; i++) { qn[i] = rawQuery[i]; n += (double)qn[i] * qn[i]; }
            if (!s_noNorm) {
                n = std::sqrt(n);
                if (n > 0) { float inv = (float)(1.0 / n); for (int i = 0; i < m_dim; i++) qn[i] *= inv; }
            }

            impl->rot->rotate(qn.data(), ctx->rq.data());

            delete ctx->q;
            ctx->q = new SplitSingleQuery<float>(
                ctx->rq.data(), impl->padded, impl->exBits, impl->qcfg, METRIC_L2);

            float cnorm = std::sqrt(euclidean_sqr(ctx->rq.data(), impl->centroid_rot.data(), impl->padded));
            ctx->q->set_g_add(cnorm);
        }

        float RaBitQ2::Estimate(void* ctxv, int vid) const
        {
            auto* impl = reinterpret_cast<RaBitQ2Impl*>(m_impl);
            auto* ctx = reinterpret_cast<RaBitQ2QueryCtx*>(ctxv);
            const char* bin = &impl->binStore[(size_t)vid * impl->binBytes];
            float est = 0, low = 0, ipx0 = 0;
            if (impl->exBits > 0)
            {
                const char* ex = &impl->exStore[(size_t)vid * impl->exBytes];
                split_single_fulldist(bin, ex, impl->ip_func, *ctx->q, impl->padded,
                    impl->exBits, est, low, ipx0, ctx->q->g_add(), ctx->q->g_error());
            }
            else
            {
                split_single_estdist(bin, *ctx->q, impl->padded, ipx0, est, low,
                    ctx->q->g_add(), ctx->q->g_error());
            }
            return est;
        }

        float RaBitQ2::EstimateCode(void* ctxv, const void* binCode, const void* exCode) const
        {
            auto* impl = reinterpret_cast<RaBitQ2Impl*>(m_impl);
            auto* ctx = reinterpret_cast<RaBitQ2QueryCtx*>(ctxv);
            const char* bin = reinterpret_cast<const char*>(binCode);
            float est = 0, low = 0, ipx0 = 0;
            if (impl->exBits > 0)
            {
                const char* ex = reinterpret_cast<const char*>(exCode);
                split_single_fulldist(bin, ex, impl->ip_func, *ctx->q, impl->padded,
                    impl->exBits, est, low, ipx0, ctx->q->g_add(), ctx->q->g_error());
            }
            else
            {
                split_single_estdist(bin, *ctx->q, impl->padded, ipx0, est, low,
                    ctx->q->g_add(), ctx->q->g_error());
            }
            return est;
        }
    }
}
