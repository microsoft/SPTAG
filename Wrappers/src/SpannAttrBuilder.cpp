// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Native C++ build entry for attribute-aware SPANN indexes.
//
// This is the C++ analog of the Python demo build_yfcc_facetA.py /
// build_5col.py: it drives the EXACT same attribute pipeline
// (TenantIndexManager::BuildFromDataWithTags -> SetVectorTags posting
// embedding + PerTagBKT head selection + ACL pivot routing + numeric quant
// signatures) but without the Python/SWIG layer, and it mmaps the vector and
// tag files so a billion-scale build streams from disk (zero extra copies, no
// 400GB float blow-up).
//
// Attribute/routing configuration is taken from the SAME SPTAG_* environment
// variables the Python demo sets from build_tag_config.json
// (SPTAG_ACL_COLS, SPTAG_HIER_LEVEL_WIDTHS, SPTAG_NUMERIC_COLS,
//  SPTAG_PER_VECTOR_TAGS_FILE, SPTAG_PERTAG_HEAD_RATIO, SPTAG_SELECT_TYPE_OVERRIDE,
//  SPTAG_DIST_METHOD, SPTAG_STORAGE_BACKEND, ...). This tool only needs the
// file layout (paths/offsets/dims) on the command line; it invents no new
// config format.
//
// Usage:
//   spannbuilder \
//     --vectors <file> --vec-offset <bytes> --n <count> --dim <D> \
//     --value-type Int8|UInt8|Float \
//     --tags <file> --tags-offset <bytes> --num-tags-per-vec <K> \
//     --index-dir <out> [--tenant 0] [--storage-backend FILEIO|ROCKSDBIO] \
//     [--build-signatures] [--with-meta-index] [--normalized]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "inc/CoreInterface.h"
#include "inc/Core/CommonDataStructure.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common/IQuantizer.h"
#include "inc/Core/SPANN/PipePQ.h"
#include "inc/Helper/SimpleIniReader.h"

using namespace SPTAG;

namespace {

struct MappedFile {
    void* base = nullptr;
    size_t length = 0;
    int fd = -1;
    bool Map(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { fprintf(stderr, "[spannbuilder] open failed: %s\n", path.c_str()); return false; }
        struct stat st;
        if (::fstat(fd, &st) != 0) { fprintf(stderr, "[spannbuilder] fstat failed: %s\n", path.c_str()); return false; }
        length = (size_t)st.st_size;
        base = ::mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) { fprintf(stderr, "[spannbuilder] mmap failed: %s\n", path.c_str()); base = nullptr; return false; }
        // The build accesses vectors by SHUFFLED index (BKT head selection,
        // graph build, posting assignment) -- i.e. effectively RANDOM, not
        // sequential. MADV_SEQUENTIAL was actively harmful here: it triggers
        // aggressive readahead (each random 100-byte fault dragged in ~45 KB)
        // AND frees pages behind the read pointer, so on a 100 GB base that
        // does not fully fit in page cache every random access re-faulted from
        // disk -> ~9x read amplification / thrashing (818 GB read for a 100 GB
        // file in the first SelectHead pass). MADV_RANDOM disables readahead
        // and page-dropping, so the working set accumulates in cache instead.
        ::madvise(base, length, MADV_RANDOM);
        return true;
    }
    ~MappedFile() {
        if (base) ::munmap(base, length);
        if (fd >= 0) ::close(fd);
    }
};

size_t ValueTypeSize(const std::string& vt) {
    if (vt == "Float" || vt == "float") return 4;
    if (vt == "Int16" || vt == "Int8" || vt == "UInt8" || vt == "int16" || vt == "int8" || vt == "uint8") {
        return (vt == "Int16" || vt == "int16") ? 2 : 1;
    }
    return 0;
}

const char* ArgVal(int argc, char** argv, const char* key, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return def;
}
bool ArgFlag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], key) == 0) return true;
    return false;
}

// --- Native .ini config resolution (mirrors classic IndexBuilder semantics) ---
// Precedence: an explicit CLI flag overrides the ini value, which overrides the
// built-in default. This keeps the config file the single source of truth while
// still allowing a one-off CLI override (exactly like the classic builder's
// `Section.Param=Value` overrides).
std::string Resolve(int argc, char** argv, const char* cliKey,
                    const Helper::IniReader* ini, const char* section, const char* key,
                    const char* def) {
    if (const char* c = ArgVal(argc, argv, cliKey, nullptr)) return std::string(c);
    if (ini && ini->DoesParameterExist(section, key))
        return ini->GetParameter<std::string>(section, key, std::string(def ? def : ""));
    return std::string(def ? def : "");
}
bool ResolveFlag(int argc, char** argv, const char* cliKey,
                 const Helper::IniReader* ini, const char* section, const char* key) {
    if (ArgFlag(argc, argv, cliKey)) return true;
    if (ini && ini->DoesParameterExist(section, key)) {
        std::string v = ini->GetParameter<std::string>(section, key, std::string("false"));
        return v == "1" || v == "true" || v == "True" || v == "yes" || v == "on";
    }
    return false;
}
// Bridge a config value to the SPTAG_* environment knob its consumer reads via
// getenv (routing / ratio / dist / unfilter-enhancement extensions have no
// native SPANN [BuildSSDIndex] param, so the ini section is delivered through
// the existing getenv sites). Config is authoritative (overwrite=1).
void IniEnv(const Helper::IniReader* ini, const char* section, const char* key, const char* env) {
    if (!ini || !ini->DoesParameterExist(section, key)) return;
    std::string v = ini->GetParameter<std::string>(section, key, std::string());
    if (v.empty()) return;
    ::setenv(env, v.c_str(), 1);
    fprintf(stderr, "[spannbuilder][cfg] %s = %s   (from [%s] %s)\n",
            env, v.c_str(), section, key);
}

} // namespace

int main(int argc, char** argv) {
    // --- Post-build slim transform mode (C++ analog of build_inpost_rbq2_contig.py) ---
    // Rewrites an existing index's full postings to slim [meta | RaBitQ2-code] stored
    // contiguously, driven by the same SPTAG_INPOST_RBQ* env the Python demo sets, then
    // exits. Requires the RBQ2 code sidecar (from rabitq2_encode_stream) inside the
    // index's tenant_0/ dir. Query-time uses SPTAG_INPOST_RBQ + SPTAG_INPOST_BASE.
    if (ArgFlag(argc, argv, "--inpost-rbq-transform")) {
        const char* indexDir = ArgVal(argc, argv, "--index-dir", nullptr);
        const char* rbqFile = ArgVal(argc, argv, "--rbq-file", nullptr);
        const int dim = (int)std::strtol(ArgVal(argc, argv, "--dim", "0"), nullptr, 10);
        const std::string valueType = ArgVal(argc, argv, "--value-type", "Int8");
        const int tenant = (int)std::strtol(ArgVal(argc, argv, "--tenant", "0"), nullptr, 10);
        if (!indexDir || !rbqFile || dim <= 0) {
            fprintf(stderr, "usage: spannbuilder --inpost-rbq-transform --index-dir <d> "
                            "--rbq-file <rabitq2.bin> --dim <D> [--value-type Int8] [--tenant 0]\n");
            return 2;
        }
        const size_t valSize = ValueTypeSize(valueType);
        if (valSize == 0) { fprintf(stderr, "[spannbuilder] bad value-type\n"); return 2; }
        // The transform runs lazily inside the ExtraDynamicSearcher ctor on first query.
        setenv("SPTAG_INPOST_RBQ", "1", 1);
        setenv("SPTAG_INPOST_RBQ_FILE", rbqFile, 1);
        setenv("SPTAG_INPOST_RBQ_BUILD", "1", 1);
        setenv("SPTAG_INPOST_RBQ_CONTIG", "1", 1);
        fprintf(stderr, "[spannbuilder] inpost-rbq-transform: load %s (rbq=%s) ...\n", indexDir, rbqFile);
        TenantIndexManager mgr(dim, "SPANN", valueType.c_str());
        if (!mgr.LoadAll(indexDir)) { fprintf(stderr, "[spannbuilder] LoadAll FAILED\n"); return 1; }
        // Dummy query forces lazy per-tenant searcher construction -> TransformInPostingsRbqContig.
        std::vector<std::uint8_t> q((size_t)dim * valSize, 0);
        ByteArray query(q.data(), q.size(), false);
        ByteArray noTags(nullptr, 0, false);
        mgr.SearchWithACL(query, tenant, 10, noTags, 0);
        fprintf(stderr, "[spannbuilder] inpost-rbq-transform complete.\n");
        return 0;
    }

    // --- OPQ code generation mode (C++ analog of AnnService/src/Quantizer/main.cpp) ---
    // Encodes a raw vector file into the in-posting OPQ code sidecar
    // (opq_codes_m<M>.bin) that the SPANN build consumes (config
    // [BuildSSDIndex] PostingQuantizerFile). Mimics the original Quantizer main
    // (IQuantizer::LoadIQuantizer + per-vector QuantizeVector) but follows the
    // in-posting convention that ExtraDynamicSearcher uses when it self-encodes
    // (ExtraDynamicSearcher.h ~5165): widen the vector to float WITHOUT
    // normalization and call QuantizeVector(vf, code, /*ADC=*/false), writing the
    // raw, header-less N*M uint8 codes vid-indexed. NOTE: the generic
    // Release/quantizer tool normalizes and emits an (n,d) header, so it is NOT a
    // drop-in for this sidecar -- use this mode instead.
    if (ArgFlag(argc, argv, "--gen-opq-codes")) {
        const char* vectors  = ArgVal(argc, argv, "--vectors", nullptr);
        const char* quantF   = ArgVal(argc, argv, "--quantizer", nullptr);
        const char* outF     = ArgVal(argc, argv, "--out", nullptr);
        const int   dim      = (int)std::strtol(ArgVal(argc, argv, "--dim", "0"), nullptr, 10);
        const long  vecOff   = std::strtol(ArgVal(argc, argv, "--vec-offset", "8"), nullptr, 10);
        const long  nArg     = std::strtol(ArgVal(argc, argv, "--n", "-1"), nullptr, 10);
        const std::string valueType = ArgVal(argc, argv, "--value-type", "Int8");
        if (!vectors || !quantF || !outF || dim <= 0) {
            fprintf(stderr, "usage: spannbuilder --gen-opq-codes --vectors <base.i8bin> "
                            "--quantizer <opq_quantizer.bin> --out <opq_codes_m<M>.bin> "
                            "--dim <D> [--vec-offset 8] [--n <count>] [--value-type Int8]\n");
            return 2;
        }
        const size_t valSize = ValueTypeSize(valueType);
        if (valSize == 0) { fprintf(stderr, "[spannbuilder] bad value-type\n"); return 2; }

        // Load the OPQ quantizer exactly like Quantizer/main.cpp.
        auto fp = SPTAG::f_createIO();
        if (fp == nullptr || !fp->Initialize(quantF, std::ios::binary | std::ios::in)) {
            fprintf(stderr, "[spannbuilder] cannot open quantizer: %s\n", quantF); return 1;
        }
        auto quantizer = SPTAG::COMMON::IQuantizer::LoadIQuantizer(fp);
        if (!quantizer) { fprintf(stderr, "[spannbuilder] failed to load quantizer\n"); return 1; }
        quantizer->SetEnableADC(false);
        const int M = quantizer->GetNumSubvectors();
        fprintf(stderr, "[spannbuilder][gen-opq-codes] M=%d dim=%d valueType=%s vec-offset=%ld\n",
                M, dim, valueType.c_str(), vecOff);

        MappedFile mf;
        if (!mf.Map(vectors)) return 1;
        const size_t recBytes = (size_t)dim * valSize;
        const size_t avail = (mf.length > (size_t)vecOff) ? (mf.length - (size_t)vecOff) : 0;
        long N = (long)(avail / recBytes);
        if (nArg >= 0 && nArg < N) N = nArg;
        if (N <= 0) { fprintf(stderr, "[spannbuilder] no vectors (avail=%zu rec=%zu)\n", avail, recBytes); return 1; }
        const char* basePtr = static_cast<const char*>(mf.base) + vecOff;

        FILE* out = std::fopen(outF, "wb");
        if (!out) { fprintf(stderr, "[spannbuilder] cannot open out: %s\n", outF); return 1; }
        fprintf(stderr, "[spannbuilder][gen-opq-codes] encoding %ld vectors -> %s (%ld bytes)\n",
                N, outF, (long)N * M);

        const size_t CHUNK = 1u << 16;  // 65536 vectors per write batch
        std::vector<std::uint8_t> codes(CHUNK * (size_t)M);
        std::vector<float> vf((size_t)dim);
        for (long s = 0; s < N; s += (long)CHUNK) {
            const long e = std::min<long>(s + (long)CHUNK, N);
            for (long i = s; i < e; ++i) {
                const char* rec = basePtr + (size_t)i * recBytes;
                // Widen to float WITHOUT normalization (matches the build's self-encode path).
                if (valueType == "Float" || valueType == "float") {
                    const float* v = reinterpret_cast<const float*>(rec);
                    for (int d = 0; d < dim; ++d) vf[d] = v[d];
                } else if (valSize == 2) {
                    const std::int16_t* v = reinterpret_cast<const std::int16_t*>(rec);
                    for (int d = 0; d < dim; ++d) vf[d] = (float)v[d];
                } else if (valueType == "UInt8" || valueType == "uint8") {
                    const std::uint8_t* v = reinterpret_cast<const std::uint8_t*>(rec);
                    for (int d = 0; d < dim; ++d) vf[d] = (float)v[d];
                } else {
                    const std::int8_t* v = reinterpret_cast<const std::int8_t*>(rec);
                    for (int d = 0; d < dim; ++d) vf[d] = (float)v[d];
                }
                quantizer->QuantizeVector(vf.data(), &codes[(size_t)(i - s) * M], /*ADC=*/false);
            }
            std::fwrite(codes.data(), 1, (size_t)(e - s) * M, out);
            fprintf(stderr, "\r[spannbuilder][gen-opq-codes] %ld/%ld", e, N);
        }
        std::fclose(out);
        fprintf(stderr, "\n[spannbuilder][gen-opq-codes] done.\n");
        return 0;
    }

    // --- PipeANN fixed-chunk PQ code generation mode ---
    // Uses the PipeANN pq_pivots.bin format and writes a raw, header-less N*M
    // uint8 sidecar consumed by PostingQuantizer=PipePQ. Existing PipeANN
    // compressed.bin files ([uint32 N][uint32 M] + codes) are also accepted
    // directly by the SPANN build/search transform path, so this mode is mainly
    // for same-algorithm/same-code-length experiments such as PQ25.
    if (ArgFlag(argc, argv, "--gen-pipepq-codes")) {
        const char* vectors = ArgVal(argc, argv, "--vectors", nullptr);
        const char* pivots = ArgVal(argc, argv, "--pivots", nullptr);
        const char* outF = ArgVal(argc, argv, "--out", nullptr);
        const int dim = (int)std::strtol(ArgVal(argc, argv, "--dim", "0"), nullptr, 10);
        const int M = (int)std::strtol(ArgVal(argc, argv, "--posting-quant-m", "0"), nullptr, 10);
        const long vecOff = std::strtol(ArgVal(argc, argv, "--vec-offset", "8"), nullptr, 10);
        const long nArg = std::strtol(ArgVal(argc, argv, "--n", "-1"), nullptr, 10);
        const std::string valueType = ArgVal(argc, argv, "--value-type", "Int8");
        if (!vectors || !pivots || !outF || dim <= 0 || M <= 0) {
            fprintf(stderr, "usage: spannbuilder --gen-pipepq-codes --vectors <base.i8bin> "
                            "--pivots <pipeann_pq_pivots.bin> --out <pipepq_codes_m<M>.bin> "
                            "--dim <D> --posting-quant-m <M> [--vec-offset 8] [--n <count>] "
                            "[--value-type Int8]\n");
            return 2;
        }
        const size_t valSize = ValueTypeSize(valueType);
        if (valSize == 0) { fprintf(stderr, "[spannbuilder] bad value-type\n"); return 2; }

        SPTAG::SPANN::PipePQTable table;
        if (!table.Load(pivots, M) || table.Dim() != dim) {
            fprintf(stderr, "[spannbuilder][gen-pipepq-codes] failed to load pivots=%s dim=%d M=%d\n",
                    pivots, dim, M);
            return 1;
        }

        MappedFile mf;
        if (!mf.Map(vectors)) return 1;
        const size_t recBytes = (size_t)dim * valSize;
        const size_t avail = (mf.length > (size_t)vecOff) ? (mf.length - (size_t)vecOff) : 0;
        long N = (long)(avail / recBytes);
        if (nArg >= 0 && nArg < N) N = nArg;
        if (N <= 0) { fprintf(stderr, "[spannbuilder] no vectors (avail=%zu rec=%zu)\n", avail, recBytes); return 1; }
        const char* basePtr = static_cast<const char*>(mf.base) + vecOff;

        FILE* out = std::fopen(outF, "wb");
        if (!out) { fprintf(stderr, "[spannbuilder] cannot open out: %s\n", outF); return 1; }
        fprintf(stderr, "[spannbuilder][gen-pipepq-codes] encoding %ld vectors -> %s (%ld bytes), M=%d\n",
                N, outF, (long)N * M, M);

        const size_t CHUNK = 1u << 16;
        std::vector<std::uint8_t> codes(CHUNK * (size_t)M);
        std::vector<float> vf((size_t)dim);
        for (long s = 0; s < N; s += (long)CHUNK) {
            const long e = std::min<long>(s + (long)CHUNK, N);
            for (long i = s; i < e; ++i) {
                const char* rec = basePtr + (size_t)i * recBytes;
                if (valueType == "Float" || valueType == "float") {
                    const float* v = reinterpret_cast<const float*>(rec);
                    for (int d = 0; d < dim; ++d) vf[d] = v[d];
                } else if (valSize == 2) {
                    const std::int16_t* v = reinterpret_cast<const std::int16_t*>(rec);
                    for (int d = 0; d < dim; ++d) vf[d] = (float)v[d];
                } else if (valueType == "UInt8" || valueType == "uint8") {
                    const std::uint8_t* v = reinterpret_cast<const std::uint8_t*>(rec);
                    for (int d = 0; d < dim; ++d) vf[d] = (float)v[d];
                } else {
                    const std::int8_t* v = reinterpret_cast<const std::int8_t*>(rec);
                    for (int d = 0; d < dim; ++d) vf[d] = (float)v[d];
                }
                table.Encode(vf.data(), &codes[(size_t)(i - s) * M]);
            }
            std::fwrite(codes.data(), 1, (size_t)(e - s) * M, out);
            fprintf(stderr, "\r[spannbuilder][gen-pipepq-codes] %ld/%ld", e, N);
        }
        std::fclose(out);
        fprintf(stderr, "\n[spannbuilder][gen-pipepq-codes] done.\n");
        return 0;
    }

    // --- Tag-merge mode: build the builder's 5-col tag sidecar from the dataset's
    //     .npy attribute arrays (C++, no Python). Reads tags.npy [N,acl] uint32 +
    //     num_attr.npy [N] int32 and writes:
    //       --out-tags5  : raw uint32 [N, acl+1] = [acl cols | numeric], row-major
    //       --out-group  : the routing-key column (default col 0), one int per line
    //     (.npy v1.0: magic 6B + ver 2B + hlen(u16) + header; data at 10+hlen.) ---
    if (ArgFlag(argc, argv, "--merge-tags5")) {
        const char* tagsNpy = ArgVal(argc, argv, "--tags-npy", nullptr);
        const char* numNpy  = ArgVal(argc, argv, "--num-npy", nullptr);
        const char* outT    = ArgVal(argc, argv, "--out-tags5", nullptr);
        const char* outG    = ArgVal(argc, argv, "--out-group", nullptr);
        const int aclCols   = (int)std::strtol(ArgVal(argc, argv, "--acl-cols", "4"), nullptr, 10);
        const int groupCol  = (int)std::strtol(ArgVal(argc, argv, "--group-col", "0"), nullptr, 10);
        const long nArg     = std::strtol(ArgVal(argc, argv, "--n", "-1"), nullptr, 10);
        if (!tagsNpy || !numNpy || !outT || !outG || aclCols <= 0) {
            fprintf(stderr, "usage: spannbuilder --merge-tags5 --tags-npy <tags.npy> "
                            "--num-npy <num_attr.npy> --out-tags5 <tags5.u32> "
                            "--out-group <group.txt> [--acl-cols 4] [--group-col 0] [--n <count>]\n");
            return 2;
        }
        auto npyData = [](const MappedFile& mf, size_t* outOff) -> bool {
            if (mf.length < 10) return false;
            const unsigned char* p = static_cast<const unsigned char*>(mf.base);
            if (std::memcmp(p, "\x93NUMPY", 6) != 0) return false;
            const size_t hlen = (size_t)p[8] | ((size_t)p[9] << 8);
            *outOff = 10 + hlen;
            return true;
        };
        MappedFile mt, mn;
        if (!mt.Map(tagsNpy) || !mn.Map(numNpy)) return 1;
        size_t tOff = 0, nOff = 0;
        if (!npyData(mt, &tOff) || !npyData(mn, &nOff)) { fprintf(stderr, "[spannbuilder] bad .npy header\n"); return 1; }
        const auto* tags = reinterpret_cast<const std::uint32_t*>(static_cast<const char*>(mt.base) + tOff);
        const auto* nums = reinterpret_cast<const std::int32_t*>(static_cast<const char*>(mn.base) + nOff);
        long N = (long)((mn.length - nOff) / sizeof(std::int32_t));
        const long Nt = (long)((mt.length - tOff) / (sizeof(std::uint32_t) * (size_t)aclCols));
        if (Nt < N) N = Nt;
        if (nArg >= 0 && nArg < N) N = nArg;
        if (groupCol < 0 || groupCol >= aclCols) { fprintf(stderr, "[spannbuilder] bad group-col\n"); return 1; }
        fprintf(stderr, "[spannbuilder][merge-tags5] N=%ld aclCols=%d -> 5col=%d group-col=%d\n",
                N, aclCols, aclCols + 1, groupCol);

        FILE* fT = std::fopen(outT, "wb");
        FILE* fG = std::fopen(outG, "wb");
        if (!fT || !fG) { fprintf(stderr, "[spannbuilder] cannot open outputs\n"); return 1; }
        const int W = aclCols + 1;
        const size_t CHUNK = 1u << 20;
        std::vector<std::uint32_t> rowbuf(CHUNK * (size_t)W);
        std::string gbuf; gbuf.reserve(CHUNK * 8);
        char numtmp[16];
        for (long s = 0; s < N; s += (long)CHUNK) {
            const long e = std::min<long>(s + (long)CHUNK, N);
            gbuf.clear();
            for (long i = s; i < e; ++i) {
                std::uint32_t* dst = &rowbuf[(size_t)(i - s) * W];
                const std::uint32_t* src = &tags[(size_t)i * aclCols];
                for (int c = 0; c < aclCols; ++c) dst[c] = src[c];
                dst[aclCols] = (std::uint32_t)nums[i];
                int len = std::snprintf(numtmp, sizeof(numtmp), "%u\n", src[groupCol]);
                gbuf.append(numtmp, len);
            }
            std::fwrite(rowbuf.data(), sizeof(std::uint32_t), (size_t)(e - s) * W, fT);
            std::fwrite(gbuf.data(), 1, gbuf.size(), fG);
            fprintf(stderr, "\r[spannbuilder][merge-tags5] %ld/%ld", e, N);
        }
        std::fclose(fT); std::fclose(fG);
        fprintf(stderr, "\n[spannbuilder][merge-tags5] done.\n");
        return 0;
    }

    // --- Native SPANN .ini config (single source of truth, classic-builder style) ---
    // -c/--config <file.ini> loads all build parameters from a sectioned ini via
    // the same Helper::IniReader the classic IndexBuilder uses. Standard SPANN
    // [BuildSSDIndex] params are applied through the native SetSSDBuildParam path;
    // the multi-tenant / unfilter-enhancement extensions live in a [MultiTenant]
    // section and the routing/ratio/dist knobs in [Base]/[SelectHead]/[BuildHead],
    // bridged to their getenv consumers. Explicit CLI flags still override the ini.
    Helper::IniReader iniStore;
    const Helper::IniReader* ini = nullptr;
    {
        const char* cfg = ArgVal(argc, argv, "--config", nullptr);
        if (!cfg) cfg = ArgVal(argc, argv, "-c", nullptr);
        if (cfg) {
            if (iniStore.LoadIniFile(cfg) != ErrorCode::Success) {
                fprintf(stderr, "[spannbuilder] cannot open config file: %s\n", cfg);
                return 2;
            }
            ini = &iniStore;
            fprintf(stderr, "[spannbuilder] config = %s\n", cfg);

            // (a) Routing / metric / head-selection knobs consumed via getenv.
            IniEnv(ini, "Base",        "DistCalcMethod",          "SPTAG_DIST_METHOD");
            IniEnv(ini, "SelectHead",  "Ratio",                   "SPTAG_PERTAG_HEAD_RATIO");
            IniEnv(ini, "SelectHead",  "SelectType",              "SPTAG_SELECT_TYPE_OVERRIDE");
            IniEnv(ini, "SelectHead",  "BKTLambdaFactor",         "SPTAG_BKT_LAMBDA_FACTOR");
            IniEnv(ini, "SelectHead",  "NumberOfThreads",         "SPTAG_SELECT_HEAD_THREADS");
            IniEnv(ini, "SelectHead",  "ParallelBKTBuild",        "SPTAG_PARALLEL_BKT");
            IniEnv(ini, "BuildHead",   "NumberOfThreads",         "SPTAG_BUILD_HEAD_THREADS");
            // (b) Multi-tenant ACL routing + numeric attribute layout.
            IniEnv(ini, "MultiTenant", "ACLCols",                 "SPTAG_ACL_COLS");
            IniEnv(ini, "MultiTenant", "HierLevelWidths",         "SPTAG_HIER_LEVEL_WIDTHS");
            IniEnv(ini, "MultiTenant", "NumericCols",             "SPTAG_NUMERIC_COLS");
            IniEnv(ini, "MultiTenant", "PerVectorTagsFile",       "SPTAG_PER_VECTOR_TAGS_FILE");
            IniEnv(ini, "MultiTenant", "PivotForceNodeCount",     "SPTAG_PIVOT_FORCE_NODE_COUNT");
            // (c) Unfilter-enhancement layers (U_extra heads + unfilter tail).
            IniEnv(ini, "MultiTenant", "DualPoolAugment",         "SPTAG_DUAL_POOL_AUGMENT");
            IniEnv(ini, "MultiTenant", "DualPoolExtraRatio",      "SPTAG_DUAL_POOL_EXTRA_RATIO");
            // Unfilter-tail K/buffer are native SSD params (TailReplicaCount /
            // UnfilterTailBufferLength) set via [BuildSSDIndex]; no env bridge.
        }
    }

    std::string sVecPath    = Resolve(argc, argv, "--vectors",          ini, "Base", "VectorPath",      nullptr);
    std::string sTagPath    = Resolve(argc, argv, "--tags",             ini, "Tags", "TagFile",         nullptr);
    std::string sIndexDir   = Resolve(argc, argv, "--index-dir",        ini, "Base", "IndexDirectory",  nullptr);
    const char* vecPath  = sVecPath.empty()  ? nullptr : sVecPath.c_str();
    const char* tagPath  = sTagPath.empty()  ? nullptr : sTagPath.c_str();
    const char* indexDir = sIndexDir.empty() ? nullptr : sIndexDir.c_str();
    if (!vecPath || !tagPath || !indexDir) {
        fprintf(stderr,
            "usage: spannbuilder -c <config.ini>   (native SPANN ini, single source of truth)\n"
            "   or: spannbuilder --vectors <f> --vec-offset <b> --n <N> --dim <D> "
            "--value-type Int8|UInt8|Float --tags <f> --tags-offset <b> "
            "--num-tags-per-vec <K> --index-dir <out> [--tenant 0] "
            "[--storage-backend FILEIO|ROCKSDBIO] [--build-signatures] "
            "[--with-meta-index] [--normalized] "
            "[--posting-quantizer None|RaBitQ|OPQ|PipePQ] [--posting-quant-m <B>] "
            "[--posting-quant-bits <b>] [--posting-quant-file <f>] "
            "[--full-vector-file <f>] [--rerank-l <L>] [--quantize-head] [--quant-adc-only] "
            "[--ssd-start-file-gb <GB>] [--ssd-max-file-gb <GB>] [--ssd-growth-file-gb <GB>]\n");
        return 2;
    }

    const size_t vecOffset = (size_t)std::strtoull(Resolve(argc, argv, "--vec-offset",        ini, "Base", "VectorOffset",   "0").c_str(), nullptr, 10);
    const size_t tagOffset = (size_t)std::strtoull(Resolve(argc, argv, "--tags-offset",       ini, "Tags", "TagOffset",      "0").c_str(), nullptr, 10);
    const long long nArg   = std::strtoll(        Resolve(argc, argv, "--n",                  ini, "Base", "VectorCount",    "0").c_str(), nullptr, 10);
    const int dim          = (int)std::strtol(    Resolve(argc, argv, "--dim",                ini, "Base", "Dim",            "0").c_str(), nullptr, 10);
    const int numTagsPerVec= (int)std::strtol(    Resolve(argc, argv, "--num-tags-per-vec",   ini, "Tags", "NumTagsPerVec",  "0").c_str(), nullptr, 10);
    const int tenant       = (int)std::strtol(    Resolve(argc, argv, "--tenant",             ini, "Tags", "Tenant",         "0").c_str(), nullptr, 10);
    const std::string valueType     = Resolve(argc, argv, "--value-type",      ini, "Base", "VectorType", "Int8");
    const std::string storageBackend= Resolve(argc, argv, "--storage-backend", ini, "BuildSSDIndex", "Storage", "FILEIO");
    const bool buildSignatures = ResolveFlag(argc, argv, "--build-signatures", ini, "Build", "BuildSignatures");
    const bool withMetaIndex   = ResolveFlag(argc, argv, "--with-meta-index",  ini, "Build", "WithMetaIndex");
    const bool normalized      = ResolveFlag(argc, argv, "--normalized",       ini, "Base",  "Normalized");

    const size_t valSize = ValueTypeSize(valueType);
    if (valSize == 0 || dim <= 0 || numTagsPerVec <= 0) {
        fprintf(stderr, "[spannbuilder] invalid value-type/dim/num-tags-per-vec\n");
        return 2;
    }

    MappedFile vecMap, tagMap;
    if (!vecMap.Map(vecPath) || !tagMap.Map(tagPath)) return 1;

    // Infer N from the vector file if not given (file_size - offset) / (dim*valSize).
    long long n = nArg;
    if (n <= 0) {
        n = (long long)((vecMap.length - vecOffset) / ((size_t)dim * valSize));
    }
    const size_t vecBytes = (size_t)n * dim * valSize;
    const size_t tagBytes = (size_t)n * numTagsPerVec * sizeof(uint32_t);
    if (vecOffset + vecBytes > vecMap.length) {
        fprintf(stderr, "[spannbuilder] vector file too small: need %zu have %zu\n",
                vecOffset + vecBytes, vecMap.length);
        return 1;
    }
    if (tagOffset + tagBytes > tagMap.length) {
        fprintf(stderr, "[spannbuilder] tag file too small: need %zu have %zu\n",
                tagOffset + tagBytes, tagMap.length);
        return 1;
    }

    fprintf(stderr,
        "[spannbuilder] N=%lld dim=%d valueType=%s tagsPerVec=%d tenant=%d backend=%s\n"
        "               vectors=%s (+%zu, %.2f GB)  tags=%s (+%zu)\n"
        "               buildSignatures=%d index-dir=%s\n",
        n, dim, valueType.c_str(), numTagsPerVec, tenant, storageBackend.c_str(),
        vecPath, vecOffset, vecBytes / 1e9, tagPath, tagOffset,
        (int)buildSignatures, indexDir);

    // Zero-copy borrowed views over the mmapped regions (ownership=false).
    std::uint8_t* vecPtr = reinterpret_cast<std::uint8_t*>(vecMap.base) + vecOffset;
    std::uint8_t* tagPtr = reinterpret_cast<std::uint8_t*>(tagMap.base) + tagOffset;
    ByteArray vectors(vecPtr, vecBytes, false);
    ByteArray tags(tagPtr, tagBytes, false);

    // Single-tenant metadata: one integer tenant id per line ("<tenant>\n").
    std::string tline = std::to_string(tenant) + "\n";
    std::string meta;
    meta.reserve((size_t)n * tline.size());
    for (long long i = 0; i < n; ++i) meta.append(tline);
    ByteArray metadata(reinterpret_cast<std::uint8_t*>(const_cast<char*>(meta.data())),
                       meta.size(), false);

    // The zero-copy borrow fast path in BuildFromData honors this env knob.
    setenv("SPTAG_BUILD_SHARE_OWNERSHIP", "1", 0 /* don't override caller */);

    TenantIndexManager mgr(dim, "SPANN", valueType.c_str());
    if (storageBackend != "FILEIO") mgr.SetStorageBackend(storageBackend.c_str());

    // Native [BuildSSDIndex] section: apply every param through the SPANN parameter
    // system (SetSSDBuildParam -> staged m_extraSSDBuildParams -> SetBuildParam at
    // build, CoreInterface.cpp). This is the same mechanism the classic IndexBuilder
    // uses, so ReplicaCount / PostingPageLimit / StartFileSizeGB / MaxFileSizeGB /
    // GrowthFileSizeGB / PostingQuantizer / PostingQuantM / PostingQuantBits /
    // PostingQuantizerFile / FullVectorFile / RerankL / QuantizeHead / QuantADCOnly
    // all flow from the ini. ("Storage" is handled above via SetStorageBackend.)
    if (ini) {
        for (const auto& kv : ini->GetParameters("BuildSSDIndex")) {
            if (kv.first == "storage") continue;
            mgr.SetSSDBuildParam(kv.first.c_str(), kv.second.c_str());
            fprintf(stderr, "[spannbuilder][cfg] [BuildSSDIndex] %s = %s\n",
                    kv.first.c_str(), kv.second.c_str());
        }
    }

    // In-posting quantization config (unified, env-free). Explicit CLI flags override
    // the ini (pushed after the [BuildSSDIndex] loop -> applied later -> win).
    {
        const char* pq = ArgVal(argc, argv, "--posting-quantizer", nullptr);   // None|RaBitQ|OPQ|PipePQ
        if (pq) mgr.SetSSDBuildParam("PostingQuantizer", pq);
        const char* pqm = ArgVal(argc, argv, "--posting-quant-m", nullptr);     // OPQ code bytes
        if (pqm) mgr.SetSSDBuildParam("PostingQuantM", pqm);
        const char* pqb = ArgVal(argc, argv, "--posting-quant-bits", nullptr);  // RaBitQ bits/dim
        if (pqb) mgr.SetSSDBuildParam("PostingQuantBits", pqb);
        const char* pqf = ArgVal(argc, argv, "--posting-quant-file", nullptr);  // code sidecar
        if (pqf) mgr.SetSSDBuildParam("PostingQuantizerFile", pqf);
        const char* fvf = ArgVal(argc, argv, "--full-vector-file", nullptr);    // cold-rerank base
        if (fvf) mgr.SetSSDBuildParam("FullVectorFile", fvf);
        const char* rl = ArgVal(argc, argv, "--rerank-l", nullptr);             // rerank depth
        if (rl) mgr.SetSSDBuildParam("RerankL", rl);
        if (ArgFlag(argc, argv, "--quantize-head")) mgr.SetSSDBuildParam("QuantizeHead", "true");
        if (ArgFlag(argc, argv, "--quant-adc-only")) mgr.SetSSDBuildParam("QuantADCOnly", "true");
        // Explicit SSD block-pool sizing (GB). When set, these pin the pre-alloc /
        // growth ceiling exactly and bypass the auto estimate (which is sized for
        // billion-scale slim postings but kept conservative). Keeping the disk
        // budget in the build script makes it visible and reproducible.
        const char* sfs = ArgVal(argc, argv, "--ssd-start-file-gb", nullptr);
        if (sfs) mgr.SetSSDBuildParam("StartFileSizeGB", sfs);
        const char* mfs = ArgVal(argc, argv, "--ssd-max-file-gb", nullptr);
        if (mfs) mgr.SetSSDBuildParam("MaxFileSizeGB", mfs);
        const char* gfs = ArgVal(argc, argv, "--ssd-growth-file-gb", nullptr);
        if (gfs) mgr.SetSSDBuildParam("GrowthFileSizeGB", gfs);
    }

    fprintf(stderr, "[spannbuilder] BuildFromDataWithTags ...\n");
    const bool routingOnly = ArgFlag(argc, argv, "--routing-only") ||
                             (std::getenv("SPTAG_ROUTING_ONLY") != nullptr);
    if (routingOnly) {
        // Repair mode: the index store already exists on disk; only (re)generate
        // the query-time tag->bundle-node routing sidecar (tag_node_index.bin).
        // Skips the full SPANN rebuild and BuildSignatures' posting scan.
        setenv("SPTAG_ROUTING_ONLY", "1", 1);
        fprintf(stderr, "[spannbuilder] ROUTING-ONLY: LoadAll(%s) ...\n", indexDir);
        if (!mgr.LoadAll(indexDir)) {
            fprintf(stderr, "[spannbuilder] ROUTING-ONLY LoadAll FAILED\n");
            return 1;
        }
        fprintf(stderr, "[spannbuilder] ROUTING-ONLY: BuildSignatures ...\n");
        if (!mgr.BuildSignatures(tenant, tags, (SizeType)n, numTagsPerVec)) {
            fprintf(stderr, "[spannbuilder] ROUTING-ONLY BuildSignatures FAILED\n");
            return 1;
        }
        fprintf(stderr, "[spannbuilder] ROUTING-ONLY done.\n");
        return 0;
    }
    bool ok = mgr.BuildFromDataWithTags(vectors, metadata, (SizeType)n,
                                        tags, numTagsPerVec, withMetaIndex, normalized);
    if (!ok) {
        fprintf(stderr, "[spannbuilder] BuildFromDataWithTags FAILED\n");
        return 1;
    }

    if (buildSignatures) {
        fprintf(stderr, "[spannbuilder] BuildSignatures (numeric quant) ...\n");
        if (!mgr.BuildSignatures(tenant, tags, (SizeType)n, numTagsPerVec)) {
            fprintf(stderr, "[spannbuilder] BuildSignatures FAILED\n");
            return 1;
        }
    }

    fprintf(stderr, "[spannbuilder] SaveAll -> %s\n", indexDir);
    if (!mgr.SaveAll(indexDir)) {
        fprintf(stderr, "[spannbuilder] SaveAll FAILED\n");
        return 1;
    }
    fprintf(stderr, "[spannbuilder] done.\n");
    return 0;
}
