// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/CoreInterface.h"
#include "inc/Helper/AtomicFile.h"
#include "inc/Test.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#include <sys/stat.h>
#define STAT_STRUCT struct _stat
#define STAT_CALL _stat
#else
#include <sys/stat.h>
#include <unistd.h>
#define STAT_STRUCT struct stat
#define STAT_CALL stat
#endif

namespace {

std::string MakeTempDir()
{
#ifdef _WIN32
    char basePath[MAX_PATH];
    DWORD baseLen = GetTempPathA(MAX_PATH, basePath);
    BOOST_REQUIRE(baseLen > 0 && baseLen < MAX_PATH);

    std::string dir = std::string(basePath) + "sptag_acl_concurrent_" +
                      std::to_string(_getpid()) + "_" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    int mkret = _mkdir(dir.c_str());
    BOOST_REQUIRE(mkret == 0);
    return dir;
#else
    char dirTemplate[] = "/tmp/sptag_acl_concurrent_XXXXXX";
    char* dir = mkdtemp(dirTemplate);
    BOOST_REQUIRE(dir != nullptr);
    return std::string(dir);
#endif
}

bool PathExists(const std::string& path)
{
    STAT_STRUCT st;
    return STAT_CALL(path.c_str(), &st) == 0;
}

void RemoveTree(const std::string& path)
{
    if (path.empty()) {
        return;
    }

#ifdef _WIN32
    std::string cmd = "rmdir /s /q \"" + path + "\"";
#else
    std::string cmd = "rm -rf \"" + path + "\"";
#endif
    std::ignore = std::system(cmd.c_str());
}

struct ScopedTempDir {
    explicit ScopedTempDir(std::string p_path)
        : path(std::move(p_path))
    {
    }

    ~ScopedTempDir()
    {
        RemoveTree(path);
    }

    std::string path;
};

std::vector<int> ExtractValidIds(const std::shared_ptr<QueryResult>& result)
{
    std::vector<int> ids;
    if (result == nullptr) {
        return ids;
    }

    for (int i = 0; i < result->GetResultNum(); ++i)
    {
        auto* entry = result->GetResult(i);
        if (entry != nullptr && entry->VID >= 0)
        {
            ids.push_back(static_cast<int>(entry->VID));
        }
    }
    return ids;
}

void FillNormalizedVectors(std::vector<float>& vectors, int numVectors, int dimension)
{
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int vectorId = 0; vectorId < numVectors; ++vectorId)
    {
        float normSq = 0.0f;
        for (int dim = 0; dim < dimension; ++dim)
        {
            float value = dist(rng);
            vectors[static_cast<size_t>(vectorId) * static_cast<size_t>(dimension) + static_cast<size_t>(dim)] = value;
            normSq += value * value;
        }

        float invNorm = 1.0f / std::sqrt(std::max(normSq, 1e-6f));
        for (int dim = 0; dim < dimension; ++dim)
        {
            vectors[static_cast<size_t>(vectorId) * static_cast<size_t>(dimension) + static_cast<size_t>(dim)] *= invNorm;
        }
    }
}

} // namespace

BOOST_AUTO_TEST_SUITE(ACLConcurrentTest)

BOOST_AUTO_TEST_CASE(SearchWithACLSameTenantThreadLocalState)
{
    constexpr int kDim = 16;
    constexpr int kNumVectors = 256;
    constexpr int kNumTagsPerVec = 4;
    constexpr int kResultNum = 10;
    constexpr int kIterationsPerThread = 20;

    std::vector<float> vectors(static_cast<size_t>(kNumVectors) * static_cast<size_t>(kDim));
    FillNormalizedVectors(vectors, kNumVectors, kDim);

    std::vector<uint32_t> tags(static_cast<size_t>(kNumVectors) * static_cast<size_t>(kNumTagsPerVec));
    for (int i = 0; i < kNumVectors; ++i)
    {
        tags[static_cast<size_t>(i) * kNumTagsPerVec + 0] = static_cast<uint32_t>(i % 2);
        tags[static_cast<size_t>(i) * kNumTagsPerVec + 1] = static_cast<uint32_t>((i / 2) % 4);
        tags[static_cast<size_t>(i) * kNumTagsPerVec + 2] = static_cast<uint32_t>((i / 4) % 8);
        tags[static_cast<size_t>(i) * kNumTagsPerVec + 3] = static_cast<uint32_t>(i % 32);
    }

    std::string metadata;
    metadata.reserve(static_cast<size_t>(kNumVectors) * 8);
    for (int i = 0; i < kNumVectors; ++i)
    {
        metadata += "tenant0\n";
    }

    TenantIndexManager builder(kDim, "SPANN", "Float");
    BOOST_REQUIRE(builder.BuildFromDataWithTags(
        ByteArray(reinterpret_cast<std::uint8_t*>(vectors.data()), vectors.size() * sizeof(float), false),
        ByteArray(reinterpret_cast<std::uint8_t*>(metadata.data()), metadata.size(), false),
        kNumVectors,
        ByteArray(reinterpret_cast<std::uint8_t*>(tags.data()), tags.size() * sizeof(uint32_t), false),
        kNumTagsPerVec,
        false,
        true));

    ScopedTempDir saveDir(MakeTempDir());
    BOOST_REQUIRE(builder.SaveAll(saveDir.path.c_str()));

    TenantIndexManager loaded(kDim, "SPANN", "Float");
    BOOST_REQUIRE(loaded.LoadAll(saveDir.path.c_str()));

    const int tenantId = loaded.GetInternalTenantId("tenant0");
    BOOST_REQUIRE_EQUAL(tenantId, 0);

    BOOST_CHECK(PathExists(saveDir.path + "/tenant_" + std::to_string(tenantId) + "/HeadIndex/head_bundle_manifest.bin"));

    const float* queryA = vectors.data();
    const float* queryB = vectors.data() + static_cast<size_t>(127) * static_cast<size_t>(kDim);
    std::vector<uint32_t> queryTagsA = {tags[0], tags[2]};
    std::vector<uint32_t> queryTagsB = {
        tags[static_cast<size_t>(127) * static_cast<size_t>(kNumTagsPerVec) + 1],
        tags[static_cast<size_t>(127) * static_cast<size_t>(kNumTagsPerVec) + 3]
    };

    auto baselineA = loaded.SearchWithACL(
        ByteArray(reinterpret_cast<std::uint8_t*>(const_cast<float*>(queryA)), kDim * sizeof(float), false),
        tenantId,
        kResultNum,
        ByteArray(reinterpret_cast<std::uint8_t*>(queryTagsA.data()), queryTagsA.size() * sizeof(uint32_t), false),
        static_cast<int>(queryTagsA.size()));
    auto baselineB = loaded.SearchWithACL(
        ByteArray(reinterpret_cast<std::uint8_t*>(const_cast<float*>(queryB)), kDim * sizeof(float), false),
        tenantId,
        kResultNum,
        ByteArray(reinterpret_cast<std::uint8_t*>(queryTagsB.data()), queryTagsB.size() * sizeof(uint32_t), false),
        static_cast<int>(queryTagsB.size()));

    std::vector<int> expectedA = ExtractValidIds(baselineA);
    std::vector<int> expectedB = ExtractValidIds(baselineB);
    BOOST_REQUIRE(!expectedA.empty());
    BOOST_REQUIRE(!expectedB.empty());
    BOOST_REQUIRE_NE(expectedA.front(), expectedB.front());

    std::mutex startMutex;
    std::condition_variable startCv;
    int readyThreads = 0;
    bool startSearch = false;

    std::mutex errorMutex;
    std::vector<std::string> failures;

    auto worker = [&](const char* name,
                      const float* query,
                      const std::vector<uint32_t>& queryTags,
                      int expectedTop1) {
        {
            std::unique_lock<std::mutex> lock(startMutex);
            ++readyThreads;
            startCv.notify_all();
            startCv.wait(lock, [&] { return startSearch; });
        }

        for (int iter = 0; iter < kIterationsPerThread; ++iter)
        {
            auto result = loaded.SearchWithACL(
                ByteArray(reinterpret_cast<std::uint8_t*>(const_cast<float*>(query)), kDim * sizeof(float), false),
                tenantId,
                kResultNum,
                ByteArray(reinterpret_cast<std::uint8_t*>(const_cast<uint32_t*>(queryTags.data())), queryTags.size() * sizeof(uint32_t), false),
                static_cast<int>(queryTags.size()));

            std::vector<int> ids = ExtractValidIds(result);
            if (ids.empty())
            {
                std::lock_guard<std::mutex> guard(errorMutex);
                failures.emplace_back(std::string(name) + ": empty result at iteration " + std::to_string(iter));
                return;
            }

            if (ids.front() != expectedTop1)
            {
                std::lock_guard<std::mutex> guard(errorMutex);
                failures.emplace_back(std::string(name) + ": top1=" + std::to_string(ids.front()) +
                                      " expected=" + std::to_string(expectedTop1) +
                                      " at iteration " + std::to_string(iter));
                return;
            }
        }
    };

    std::thread threadA(worker, "A", queryA, std::cref(queryTagsA), expectedA.front());
    std::thread threadB(worker, "B", queryB, std::cref(queryTagsB), expectedB.front());

    {
        std::unique_lock<std::mutex> lock(startMutex);
        startCv.wait(lock, [&] { return readyThreads == 2; });
        startSearch = true;
        startCv.notify_all();
    }

    threadA.join();
    threadB.join();

    if (!failures.empty())
    {
        BOOST_FAIL(failures.front());
    }

    BOOST_CHECK(failures.empty());
}

BOOST_AUTO_TEST_CASE(HybridTagRoutingStatsPersistRepairAndReload)
{
    constexpr int kDim = 8;
    constexpr int kNumVectors = 256;
    constexpr int kNumTagsPerVec = 5;
    constexpr int kResultNum = 5;

    std::vector<float> vectors(
        static_cast<size_t>(kNumVectors) *
        static_cast<size_t>(kDim));
    FillNormalizedVectors(vectors, kNumVectors, kDim);

    std::vector<uint32_t> tags(
        static_cast<size_t>(kNumVectors) *
        static_cast<size_t>(kNumTagsPerVec));
    for (int i = 0; i < kNumVectors; ++i)
    {
        tags[static_cast<size_t>(i) * kNumTagsPerVec + 0] =
            1000U + static_cast<uint32_t>(i / 128);
        tags[static_cast<size_t>(i) * kNumTagsPerVec + 1] =
            2000U + static_cast<uint32_t>(i / 32);
        tags[static_cast<size_t>(i) * kNumTagsPerVec + 2] =
            3000U + static_cast<uint32_t>(i / 8);
        tags[static_cast<size_t>(i) * kNumTagsPerVec + 3] =
            i < 2
                ? 4000U
                : 3000U +
                      static_cast<uint32_t>(i / 2);
        tags[static_cast<size_t>(i) * kNumTagsPerVec + 4] =
            0x80000000U +
            static_cast<uint32_t>(i);
    }

    std::string metadata;
    for (int i = 0; i < kNumVectors; ++i)
    {
        metadata += "tenant0\n";
    }

    TenantIndexManager builder(kDim, "SPANN", "Float");
    builder.SetStorageBackend("STATIC");
    builder.SetBuildParam("DistCalcMethod", "L2", "Base");
    builder.SetBuildParam("IndexAlgoType", "BKT", "Base");
    builder.SetBuildParam("SelectHeadType", "Random", "SelectHead");
    builder.SetBuildParam("Ratio", "0.25", "SelectHead");
    builder.SetBuildParam("NumberOfThreads", "1", "SelectHead");
    builder.SetBuildParam("NumberOfThreads", "1", "BuildHead");
    builder.SetBuildParam("NeighborhoodSize", "16", "BuildHead");
    builder.SetSSDBuildParam("InternalResultNum", "16");
    builder.SetSSDBuildParam("SearchInternalResultNum", "8");
    builder.SetSSDBuildParam("NumberOfThreads", "2");
    builder.SetSSDBuildParam("PostingPageLimit", "2");
    builder.SetSSDBuildParam("SearchPostingPageLimit", "2");
    builder.SetSSDBuildParam("ReplicaCount", "3");
    builder.SetSSDBuildParam("TailReplicaCount", "2");
    builder.SetSSDBuildParam("EnableUnfilterTail", "true");
    builder.SetSSDBuildParam("UnfilterTailBufferLength", "-1");
    builder.SetSSDBuildParam("CrossEdges", "1");
    builder.SetSSDBuildParam("CrossExtraEdges", "4");
    builder.SetSSDBuildParam("ExcludeHead", "true");
    builder.SetSSDBuildParam("StaticACLTagCols", "4");
    builder.SetSSDBuildParam("EnableHybridDistance", "true");
    builder.SetSSDBuildParam("HybridVectorWeight", "1");
    builder.SetSSDBuildParam("HybridCategoricalCols", "0,1,2,3");
    builder.SetSSDBuildParam(
        "HybridCategoricalWeights", "8,16,32,64");
    builder.SetSSDBuildParam("HybridNumericCols", "4");
    builder.SetSSDBuildParam("HybridNumericWeights", "0.01");
    builder.SetSSDBuildParam("HybridGraphDegree", "8");
    builder.SetSSDBuildParam("HybridCandidateCount", "32");

    ByteArray vectorBytes(
        reinterpret_cast<std::uint8_t*>(vectors.data()),
        vectors.size() * sizeof(float), false);
    ByteArray metadataBytes(
        reinterpret_cast<std::uint8_t*>(metadata.data()),
        metadata.size(), false);
    ByteArray tagBytes(
        reinterpret_cast<std::uint8_t*>(tags.data()),
        tags.size() * sizeof(uint32_t), false);
    BOOST_REQUIRE(builder.BuildFromDataWithTags(
        vectorBytes, metadataBytes, kNumVectors,
        tagBytes, kNumTagsPerVec, false, true));
    BOOST_REQUIRE(builder.BuildSignatures(
        0, tagBytes, kNumVectors, kNumTagsPerVec));

    ScopedTempDir saveDir(MakeTempDir());
    BOOST_REQUIRE(builder.SaveAll(saveDir.path.c_str()));
    const std::string tenantDir =
        saveDir.path + "/tenant_0";
    const std::string routeStats =
        tenantDir + "/tag_routing_stats.bin";
    BOOST_REQUIRE(PathExists(routeStats));
    BOOST_REQUIRE(PathExists(
        tenantDir + "/SPTAGHybridList.bin.stats"));
    BOOST_REQUIRE(PathExists(
        tenantDir +
        "/HeadIndex/head_hybrid_edges.bin"));
    {
        struct RouteRecord {
            std::uint32_t column;
            std::uint32_t tag;
            std::int32_t vectorCount;
            std::int32_t postingCount;
        };
        const ByteArray routeBlob =
            builder.GetColumnAwareTagRoutingStatsBlob(0);
        BOOST_REQUIRE_EQUAL(
            routeBlob.Length() %
                sizeof(RouteRecord),
            0);
        BOOST_CHECK_EQUAL(
            routeBlob.Length() /
                sizeof(RouteRecord),
            170);
        bool foundColumn2 = false;
        bool foundColumn3 = false;
        const auto* records =
            reinterpret_cast<const RouteRecord*>(
                routeBlob.Data());
        for (size_t index = 0;
             index <
             routeBlob.Length() /
                 sizeof(RouteRecord);
             ++index) {
            BOOST_CHECK_LT(records[index].column, 4);
            if (records[index].tag == 3001U &&
                records[index].column == 2) {
                foundColumn2 = true;
                BOOST_CHECK_EQUAL(
                    records[index].vectorCount, 8);
            }
            if (records[index].tag == 3001U &&
                records[index].column == 3) {
                foundColumn3 = true;
                BOOST_CHECK_EQUAL(
                    records[index].vectorCount, 2);
            }
        }
        BOOST_CHECK(foundColumn2);
        BOOST_CHECK(foundColumn3);

        struct LegacyRouteRecord {
            std::uint32_t tag;
            std::int32_t vectorCount;
            std::int32_t postingCount;
        };
        const ByteArray legacyBlob =
            builder.GetTagRoutingStatsBlob(0);
        BOOST_REQUIRE_EQUAL(
            legacyBlob.Length() %
                sizeof(LegacyRouteRecord),
            0);
        const auto* legacyRecords =
            reinterpret_cast<const LegacyRouteRecord*>(
                legacyBlob.Data());
        bool foundLegacy3001 = false;
        for (size_t index = 0;
             index <
             legacyBlob.Length() /
                 sizeof(LegacyRouteRecord);
             ++index) {
            if (legacyRecords[index].tag == 3001U) {
                foundLegacy3001 = true;
                BOOST_CHECK_EQUAL(
                    legacyRecords[index].vectorCount,
                    10);
            }
        }
        BOOST_CHECK(foundLegacy3001);
    }

    const auto filteredSearch =
        [&](TenantIndexManager& manager) {
            const std::uint32_t queryTag = tags[3];
            return manager.SearchWithACL(
                ByteArray(
                    reinterpret_cast<std::uint8_t*>(
                        vectors.data()),
                    kDim * sizeof(float), false),
                0, kResultNum,
                ByteArray(
                    reinterpret_cast<std::uint8_t*>(
                        const_cast<std::uint32_t*>(
                            &queryTag)),
                    sizeof(queryTag), false),
                1);
        };

    TenantIndexManager loaded(kDim, "SPANN", "Float");
    BOOST_REQUIRE(loaded.LoadAll(saveDir.path.c_str()));
    BOOST_CHECK_GT(
        loaded.GetColumnAwareTagRoutingStatsBlob(0).Length(), 0);
    BOOST_REQUIRE(filteredSearch(loaded) != nullptr);

    const std::vector<std::uint32_t> mixedOrDNF = {
        0x444E4633U, 2,
        1, 0, 3, SPTAG::Cache::DNF_EQ, 4000U,
        1, 1, 4, SPTAG::Cache::DNF_GE,
        tags[static_cast<size_t>(200) *
             kNumTagsPerVec + 4]};
    auto mixedResult = loaded.SearchWithACL(
        ByteArray(
            reinterpret_cast<std::uint8_t*>(
                vectors.data() +
                static_cast<size_t>(200) *
                    kDim),
            kDim * sizeof(float), false),
        0, kResultNum,
        ByteArray(
            reinterpret_cast<std::uint8_t*>(
                const_cast<std::uint32_t*>(
                    mixedOrDNF.data())),
            mixedOrDNF.size() *
                sizeof(std::uint32_t),
            false),
        -1);
    BOOST_REQUIRE(mixedResult != nullptr);
    const auto mixedIds =
        ExtractValidIds(mixedResult);
    BOOST_CHECK(
        std::find(
            mixedIds.begin(), mixedIds.end(),
            200) != mixedIds.end());
    for (int vectorId : mixedIds) {
        const bool categoricalMatch =
            tags[static_cast<size_t>(vectorId) *
                     kNumTagsPerVec +
                 3] == 4000U;
        const bool numericMatch =
            tags[static_cast<size_t>(vectorId) *
                     kNumTagsPerVec +
                 4] >=
            tags[static_cast<size_t>(200) *
                     kNumTagsPerVec +
                 4];
        BOOST_CHECK(
            categoricalMatch ||
            numericMatch);
    }

    const std::vector<std::uint32_t> columnDNF = {
        0x444E4633U, 1,
        1, 0, 3, SPTAG::Cache::DNF_EQ, 3001U};
    auto columnResult = loaded.SearchWithACL(
        ByteArray(
            reinterpret_cast<std::uint8_t*>(
                vectors.data() +
                static_cast<size_t>(2) * kDim),
            kDim * sizeof(float), false),
        0, kResultNum,
        ByteArray(
            reinterpret_cast<std::uint8_t*>(
                const_cast<std::uint32_t*>(
                    columnDNF.data())),
            columnDNF.size() *
                sizeof(std::uint32_t),
            false),
        -1);
    BOOST_REQUIRE(columnResult != nullptr);
    const auto columnIds =
        ExtractValidIds(columnResult);
    BOOST_REQUIRE(
        std::find(
            columnIds.begin(), columnIds.end(),
            2) != columnIds.end());
    for (int vectorId : columnIds) {
        BOOST_CHECK_EQUAL(
            tags[static_cast<size_t>(vectorId) *
                     kNumTagsPerVec +
                 3],
            3001U);
    }

    {
        FILE* file = std::fopen(routeStats.c_str(), "r+b");
        BOOST_REQUIRE(file != nullptr);
        BOOST_REQUIRE(std::fseek(file, 16, SEEK_SET) == 0);
        const std::uint64_t staleGeneration =
            0x123456789abcdef0ULL;
        BOOST_REQUIRE(
            std::fwrite(
                &staleGeneration,
                sizeof(staleGeneration), 1, file) == 1);
        BOOST_REQUIRE(std::fclose(file) == 0);
    }
    TenantIndexManager stale(kDim, "SPANN", "Float");
    BOOST_REQUIRE(stale.LoadAll(saveDir.path.c_str()));
    BOOST_CHECK(filteredSearch(stale) == nullptr);
    BOOST_REQUIRE(stale.BuildSignatures(
        0, tagBytes, kNumVectors, kNumTagsPerVec));
    BOOST_REQUIRE(filteredSearch(stale) != nullptr);

    BOOST_REQUIRE(std::remove(routeStats.c_str()) == 0);
    TenantIndexManager missing(kDim, "SPANN", "Float");
    BOOST_REQUIRE(missing.LoadAll(saveDir.path.c_str()));
    BOOST_CHECK(filteredSearch(missing) == nullptr);
    BOOST_REQUIRE(missing.BuildSignatures(
        0, tagBytes, kNumVectors, kNumTagsPerVec));
    BOOST_REQUIRE(filteredSearch(missing) != nullptr);

    {
        FILE* file = std::fopen(
            routeStats.c_str(), "r+b");
        BOOST_REQUIRE(file != nullptr);
        BOOST_REQUIRE(
            std::fseek(file, 12, SEEK_SET) == 0);
        const std::int32_t impossibleRecordCount =
            (std::numeric_limits<
                 std::int32_t>::max)();
        BOOST_REQUIRE(
            std::fwrite(
                &impossibleRecordCount,
                sizeof(impossibleRecordCount), 1,
                file) == 1);
        BOOST_REQUIRE(std::fclose(file) == 0);
    }
    TenantIndexManager corrupt(kDim, "SPANN", "Float");
    BOOST_REQUIRE(
        corrupt.LoadAll(saveDir.path.c_str()));
    BOOST_CHECK(filteredSearch(corrupt) == nullptr);
    BOOST_REQUIRE(corrupt.BuildSignatures(
        0, tagBytes, kNumVectors,
        kNumTagsPerVec));
    BOOST_REQUIRE(filteredSearch(corrupt) != nullptr);

    const std::string headMetadata =
        tenantDir +
        "/HeadIndex/head_node_meta.bin";
    BOOST_REQUIRE(
        std::filesystem::exists(headMetadata));
    {
        FILE* file = std::fopen(
            headMetadata.c_str(), "r+b");
        BOOST_REQUIRE(file != nullptr);
        BOOST_REQUIRE(
            std::fseek(
                file,
                static_cast<long>(
                    sizeof(std::int32_t) * 2),
                SEEK_SET) == 0);
        const std::int32_t impossibleNumericColumns =
            (std::numeric_limits<
                 std::int32_t>::max)();
        BOOST_REQUIRE(
            std::fwrite(
                &impossibleNumericColumns,
                sizeof(impossibleNumericColumns),
                1, file) == 1);
        BOOST_REQUIRE(std::fclose(file) == 0);
    }
    TenantIndexManager forgedMetadata(
        kDim, "SPANN", "Float");
    BOOST_REQUIRE(
        forgedMetadata.LoadAll(
            saveDir.path.c_str()));
    BOOST_REQUIRE(
        forgedMetadata.BuildSignatures(
            0, tagBytes, kNumVectors,
            kNumTagsPerVec));
    BOOST_REQUIRE(
        filteredSearch(forgedMetadata) !=
        nullptr);

    std::filesystem::resize_file(
        headMetadata, 4);
    BOOST_REQUIRE(corrupt.BuildSignatures(
        0, tagBytes, kNumVectors,
        kNumTagsPerVec));
    BOOST_CHECK_GT(
        std::filesystem::file_size(
            headMetadata),
        4);
    TenantIndexManager repaired(
        kDim, "SPANN", "Float");
    BOOST_REQUIRE(
        repaired.LoadAll(saveDir.path.c_str()));
    BOOST_REQUIRE(
        filteredSearch(repaired) != nullptr);
}

BOOST_AUTO_TEST_CASE(HeadMetadataWidthsRemainIndexLocal)
{
    int narrowBits[SPTAG::Cache::HIER_LEVELS] = {
        64, 64, 64, 64, 64};
    int wideBits[SPTAG::Cache::HIER_LEVELS] = {
        256, 128, 128, 256, 64};
    SPTAG::Cache::HierWidthTable narrow;
    SPTAG::Cache::HierWidthTable wide;
    narrow.Set(
        narrowBits,
        SPTAG::Cache::HIER_LEVELS);
    wide.Set(
        wideBits,
        SPTAG::Cache::HIER_LEVELS);

    auto first = SPTAG::VectorIndex::CreateInstance(
        SPTAG::IndexAlgoType::BKT,
        SPTAG::VectorValueType::Float);
    auto second = SPTAG::VectorIndex::CreateInstance(
        SPTAG::IndexAlgoType::BKT,
        SPTAG::VectorValueType::Float);
    BOOST_REQUIRE(first != nullptr);
    BOOST_REQUIRE(second != nullptr);

    first->InitializeHeadNodeMeta(
        1, 0, narrow);
    SPTAG::Cache::HierarchicalPostingMask firstMask;
    firstMask.Clear();
    firstMask.Insert(4, 12345U, narrow);
    first->SetHeadNodePostingHierMask(
        0, firstMask);

    second->InitializeHeadNodeMeta(
        1, 0, wide);
    SPTAG::Cache::HierarchicalPostingMask secondMask;
    secondMask.Clear();
    secondMask.Insert(1, 98765U, wide);
    second->SetHeadNodePostingHierMask(
        0, secondMask);

    const auto savedGlobalWidths =
        SPTAG::Cache::HierWidths();
    SPTAG::Cache::SetHierWidths(
        wideBits,
        SPTAG::Cache::HIER_LEVELS);

    BOOST_CHECK_NE(
        first->GetHeadNodeMetaStride(),
        second->GetHeadNodeMetaStride());
    BOOST_CHECK_EQUAL(
        first->GetHeadNodeHierWidths().bits[4],
        narrow.bits[4]);
    BOOST_CHECK_EQUAL(
        second->GetHeadNodeHierWidths().bits[1],
        wide.bits[1]);

    SPTAG::Cache::HierarchicalPostingMask firstQuery;
    firstQuery.Clear();
    firstQuery.Insert(4, 12345U, narrow);
    BOOST_CHECK(
        first->HeadPostingHierMaskMayIntersect(
            0, firstQuery));

    SPTAG::Cache::HierarchicalPostingMask secondQuery;
    secondQuery.Clear();
    secondQuery.Insert(1, 98765U, wide);
    BOOST_CHECK(
        second->HeadPostingHierMaskMayIntersect(
            0, secondQuery));
    BOOST_CHECK(
        !first->HeadPostingHierMaskMayIntersect(
            0, secondQuery));
    SPTAG::Cache::SetHierWidths(
        savedGlobalWidths.bits,
        SPTAG::Cache::HIER_LEVELS);
}

BOOST_AUTO_TEST_CASE(AtomicReplacementPreservesPublishedFileOnFailure)
{
    ScopedTempDir dir(MakeTempDir());
    const std::string destination =
        dir.path + "/published.bin";
    const std::string temporary =
        destination + ".tmp";
    {
        std::ofstream output(
            destination,
            std::ios::binary);
        BOOST_REQUIRE(output.good());
        output << "old";
    }

    BOOST_CHECK(
        !SPTAG::Helper::AtomicReplaceFile(
            temporary, destination));
    {
        std::ifstream input(
            destination,
            std::ios::binary);
        std::string contents;
        input >> contents;
        BOOST_CHECK_EQUAL(contents, "old");
    }

    {
        std::ofstream output(
            temporary,
            std::ios::binary);
        BOOST_REQUIRE(output.good());
        output << "new";
    }
    BOOST_REQUIRE(
        SPTAG::Helper::AtomicReplaceFile(
            temporary, destination));
    {
        std::ifstream input(
            destination,
            std::ios::binary);
        std::string contents;
        input >> contents;
        BOOST_CHECK_EQUAL(contents, "new");
    }
}

BOOST_AUTO_TEST_SUITE_END()