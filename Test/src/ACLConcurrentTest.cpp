// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/CoreInterface.h"
#include "inc/Test.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

BOOST_AUTO_TEST_SUITE_END()