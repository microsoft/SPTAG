// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"
#include "inc/TestDataGenerator.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Helper/DiskIO.h"
#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/SPANN/SPANNResultIterator.h"

#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <thread>
#include <atomic>
#include <chrono>
#include <filesystem>

using namespace SPTAG;

namespace SPFreshTest {
    SizeType N = 1000;
    DimensionType M = 100;
    int K = 10;

    template<typename T>
    std::shared_ptr<VectorIndex> BuildIndex(
        const std::string& outDirectory,
        std::shared_ptr<VectorSet> vecset,
        std::shared_ptr<MetadataSet> metaset,
        const std::string& distMethod = "L2")
    {
        auto vecIndex = VectorIndex::CreateInstance(IndexAlgoType::SPANN, GetEnumValueType<T>());

        std::string configuration = R"(
        [Base]
            DistCalcMethod=L2
            IndexAlgoType=BKT
            ValueType=Int8
            Dim=)" + std::to_string(M) + R"(
            IndexDirectory=)" + outDirectory + R"(

        [SelectHead]
            isExecute=true
            NumberOfThreads=16
            SelectThreshold=0
            SplitFactor=0
            SplitThreshold=0
            Ratio=0.2

        [BuildHead]
            isExecute=true
            NumberOfThreads=16

        [BuildSSDIndex]
            isExecute=true
            BuildSsdIndex=true
            InternalResultNum=64
            SearchInternalResultNum=64
            NumberOfThreads=16
	    PostingPageLimit=)" + std::to_string(4 * sizeof(T)) + R"(
            SearchPostingPageLimit=)" + std::to_string(4 * sizeof(T)) + R"(
            TmpDir=tmpdir
            Storage=FILEIO
            SpdkBatchSize=64
            ExcludeHead=false
            ResultNum=10
            SearchThreadNum=2
            Update=true
            SteadyState=true
            InsertThreadNum=1
            AppendThreadNum=1
            ReassignThreadNum=0
            DisableReassign=false
            ReassignK=64
            LatencyLimit=50.0
            SearchDuringUpdate=true
            MergeThreshold=10
            Sampling=4
            BufferLength=6
            InPlace=true
	    StartFileSizeGB=1
        )";

        std::shared_ptr<Helper::DiskIO> buffer(new Helper::SimpleBufferIO());
        SPTAG::Helper::IniReader reader;
        if (!buffer->Initialize(configuration.data(), std::ios::in, configuration.size())) return nullptr;
        if (SPTAG::ErrorCode::Success != reader.LoadIni(buffer)) return nullptr;

        std::string sections[] = { "Base", "SelectHead", "BuildHead", "BuildSSDIndex" };
        for (const auto& sec : sections) {
            auto params = reader.GetParameters(sec.c_str());
            for (const auto& [key, val] : params) {
                vecIndex->SetParameter(key.c_str(), val.c_str(), sec.c_str());
            }
        }

        auto buildStatus = vecIndex->BuildIndex(vecset, metaset, true, false, false);
        if (buildStatus != ErrorCode::Success) return nullptr;

        return vecIndex;
}

    template <typename T>
    std::vector<QueryResult> SearchOnly(std::shared_ptr<VectorIndex>& vecIndex,
        std::shared_ptr<VectorSet>& queryset,
        int k)
    {
        std::vector<QueryResult> res(queryset->Count(), QueryResult(nullptr, k, true));

        auto t1 = std::chrono::high_resolution_clock::now();
        for (SizeType i = 0; i < queryset->Count(); i++) {
            res[i].SetTarget(queryset->GetVector(i));
            vecIndex->SearchIndex(res[i]);
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        float avgUs = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / static_cast<float>(queryset->Count());
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Avg search time: %.2fus/query\n", avgUs);

        return res;
    }

    template <typename T>
    float EvaluateRecall(const std::vector<QueryResult>& res,
        std::shared_ptr<VectorIndex>& vecIndex,
        std::shared_ptr<VectorSet>& queryset,
        std::shared_ptr<VectorSet>& truth,
        std::shared_ptr<VectorSet>& baseVec,
        std::shared_ptr<VectorSet>& addVec,
        SizeType baseCount,
        int k)
    {
        if (!truth) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Truth data is null. Cannot compute recall.\n");
            return 0.0f;
        }

        const SizeType recallK = min(k, static_cast<int>(truth->Dimension()));
        float totalRecall = 0.0f;
        float eps = 1e-4f;

        for (SizeType i = 0; i < queryset->Count(); ++i) {
            const SizeType* truthNN = reinterpret_cast<const SizeType*>(truth->GetVector(i));
            for (int j = 0; j < recallK; ++j) {
                SizeType truthVid = truthNN[j];
                float truthDist = (truthVid < baseCount)
                    ? vecIndex->ComputeDistance(queryset->GetVector(i), baseVec->GetVector(truthVid))
                    : vecIndex->ComputeDistance(queryset->GetVector(i), addVec->GetVector(truthVid - baseCount));

                for (int l = 0; l < k; ++l) {
                    const auto result = res[i].GetResult(l);
                    if (truthVid == result->VID ||
                        std::fabs(truthDist - result->Dist) <= eps * (std::fabs(truthDist) + eps)) {
                        totalRecall += 1.0f;
                        break;
                    }
                }
            }
        }

        float avgRecall = totalRecall / (queryset->Count() * recallK);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recall %d@%d = %.4f\n", k, recallK, avgRecall);
        return avgRecall;
    }

    template <typename T>
    float Search(std::shared_ptr<VectorIndex>& vecIndex,
        std::shared_ptr<VectorSet>& queryset,
        std::shared_ptr<VectorSet>& baseVec,
        std::shared_ptr<VectorSet>& addVec,
        int k,
        std::shared_ptr<VectorSet>& truth,
        SizeType baseCount)
    {
        auto results = SearchOnly<T>(vecIndex, queryset, k);
        return EvaluateRecall<T>(results, vecIndex, queryset, truth, baseVec, addVec, baseCount, k);
    }

    template <typename ValueType>
    void InsertVectors(SPANN::Index<ValueType>* p_index,
        int insertThreads,
        int step,
        std::shared_ptr<SPTAG::VectorSet> addset,
        std::shared_ptr<MetadataSet>& metaset)
    {
        SPANN::Options& p_opts = *(p_index->GetOptions());
        p_index->ForceCompaction();
        p_index->GetDBStat();

        std::vector<std::thread> threads;

        std::atomic_size_t vectorsSent(0);
        auto func = [&]()
            {
                size_t index = 0;
                while (true)
                {
                    index = vectorsSent.fetch_add(1);
                    if (index < step)
                    {
                        if ((index & ((1 << 5) - 1)) == 0)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Sent %.2lf%%...\n", index * 100.0 / step);
                        }
                        ByteArray p_meta = metaset->GetMetadata((SizeType)index);
                        std::uint64_t* offsets = new std::uint64_t[2]{ 0,  p_meta.Length() };
                        std::shared_ptr<SPTAG::MetadataSet> meta(new SPTAG::MemMetadataSet(p_meta, ByteArray((std::uint8_t*)offsets, 2 * sizeof(std::uint64_t), true), 1));
                        auto status = p_index->AddIndex(addset->GetVector((SizeType)index), 1, p_opts.m_dim, meta, true);
                        BOOST_REQUIRE(status == ErrorCode::Success);
                    }
                    else
                    {
                        return;
                    }
                }
            };
        for (int j = 0; j < insertThreads; j++) { threads.emplace_back(func); }
        for (auto& thread : threads) { thread.join(); }

        while (!p_index->AllFinished())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
}

bool CompareFilesWithLogging(const std::filesystem::path& file1, const std::filesystem::path& file2) {
    std::ifstream f1(file1, std::ios::binary);
    std::ifstream f2(file2, std::ios::binary);

    if (!f1.is_open() || !f2.is_open()) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
            "Failed to open one of the files:\n  %s\n  %s\n",
            file1.string().c_str(),
            file2.string().c_str());
        return false;
    }

    // Check file sizes first
    f1.seekg(0, std::ios::end);
    f2.seekg(0, std::ios::end);
    if (f1.tellg() != f2.tellg()) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
            "File size differs: %s\n", file1.filename().string().c_str());
        return false;
    }

    f1.seekg(0, std::ios::beg);
    f2.seekg(0, std::ios::beg);

    const int bufferSize = 4096; // Adjust buffer size as needed
    std::vector<char> buffer1(bufferSize);
    std::vector<char> buffer2(bufferSize);

    while (f1.read(buffer1.data(), bufferSize) && f2.read(buffer2.data(), bufferSize)) {
        if (std::memcmp(buffer1.data(), buffer2.data(), f1.gcount()) != 0) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                "File mismatch at: %s\n", file1.filename().string().c_str());
            return false; // Mismatch found
        }
    }

    return true;
}

bool CompareDirectoriesWithLogging(const std::filesystem::path& dir1,
    const std::filesystem::path& dir2,
    const std::unordered_set<std::string>& exceptions = {}) {
    std::map<std::string, std::filesystem::path> files1, files2;

    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir1)) {
        if (entry.is_regular_file()) {
            files1[std::filesystem::relative(entry.path(), dir1).string()] = entry.path();
        }
    }

    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir2)) {
        if (entry.is_regular_file()) {
            files2[std::filesystem::relative(entry.path(), dir2).string()] = entry.path();
        }
    }

    bool matched = true;

    for (const auto& [relPath, filePath1] : files1) {
        if (exceptions.count(relPath)) continue;

        auto it = files2.find(relPath);
        if (it == files2.end()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                "Missing in %s: %s\n", dir2.string().c_str(), relPath.c_str());
            matched = false;
            continue;
        }
        if (!CompareFilesWithLogging(filePath1, it->second)) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                "File end differs: %s\n", filePath1.filename().string().c_str());
            matched = false;
        }
    }

    for (const auto& [relPath, _] : files2) {
        if (exceptions.count(relPath)) continue;
        if (files1.find(relPath) == files1.end()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                "Extra in %s: %s\n", dir2.string().c_str(), relPath.c_str());
            matched = false;
        }
    }

    return matched;
}

void NormalizeVector(float* embedding, int dimension) {
    // get magnitude
    float magnitude = 0.0f;
    {
        float sum = 0.0;
        for (int i = 0; i < dimension; i++) {
            sum += embedding[i] * embedding[i];
        }
        magnitude = std::sqrt(sum);
    }

    // normalized target vector
    for (int i = 0; i < dimension; i++) {
        embedding[i] /= magnitude;
    }
}

template<typename T>
std::shared_ptr<VectorSet> get_embeddings(uint32_t row_id, uint32_t end_id, uint32_t embedding_dim, uint32_t array_index) {
    uint32_t count = end_id - row_id;
    ByteArray vec = ByteArray::Alloc(sizeof(T) * count * embedding_dim);
    for (uint32_t rid = 0; rid < count; rid++) {
        for (int idx = 0; idx < embedding_dim; ++idx) {
            ((T*)vec.Data())[rid * embedding_dim + idx] = (row_id + rid) * 17 + idx * 19 + (array_index + 1) * 23;
        }
        NormalizeVector(((T*)vec.Data()) + rid * embedding_dim, embedding_dim);
    }
    return std::make_shared<BasicVectorSet>(vec, GetEnumValueType<T>(), embedding_dim, count);
}

BOOST_AUTO_TEST_SUITE(SPFreshTest)

BOOST_AUTO_TEST_CASE(TestLoadAndSave)
{
    using namespace SPFreshTest;

    // Prepare test data using TestDataGenerator
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, 10, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    originalIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedIndex;
    BOOST_REQUIRE(VectorIndex::LoadIndex("original_index", loadedIndex) == ErrorCode::Success);
    BOOST_REQUIRE(loadedIndex != nullptr);
    BOOST_REQUIRE(loadedIndex->SaveIndex("loaded_and_saved_index") == ErrorCode::Success);
    loadedIndex = nullptr;
    
    std::unordered_set<std::string> exceptions = {
        "indexloader.ini"
    };

    // Compare files in both directories
    BOOST_REQUIRE_MESSAGE(
        CompareDirectoriesWithLogging("original_index", "loaded_and_saved_index", exceptions),
        "Saved index does not match loaded-then-saved index"
    );

    std::filesystem::remove_all("original_index");
    std::filesystem::remove_all("loaded_and_saved_index");
}

BOOST_AUTO_TEST_CASE(TestReopenIndexRecall)
{
    using namespace SPFreshTest;

    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, 10, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    float recall1 = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N);
    originalIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedOnce;
    BOOST_REQUIRE(VectorIndex::LoadIndex("original_index", loadedOnce) == ErrorCode::Success);
    BOOST_REQUIRE(loadedOnce != nullptr);
    BOOST_REQUIRE(loadedOnce->SaveIndex("reopened_index") == ErrorCode::Success);
    loadedOnce = nullptr;

    std::shared_ptr<VectorIndex> loadedTwice;
    BOOST_REQUIRE(VectorIndex::LoadIndex("reopened_index", loadedTwice) == ErrorCode::Success);
    BOOST_REQUIRE(loadedTwice != nullptr);
    float recall2 = Search<int8_t>(loadedTwice, queryset, vecset, addvecset, K, truth, N);
    loadedTwice = nullptr;

    BOOST_REQUIRE_MESSAGE(std::fabs(recall1 - recall2) < 1e-5,
        "Recall mismatch between original and reopened index");

    std::filesystem::remove_all("original_index");
    std::filesystem::remove_all("reopened_index");
}

BOOST_AUTO_TEST_CASE(TestInsertAndSearch)
{
    using namespace SPFreshTest;

    // Prepare test data using TestDataGenerator
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, 10, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    // Build base index
    auto index = BuildIndex<int8_t>("insert_test_index", vecset, metaset);
    BOOST_REQUIRE(index != nullptr);
    BOOST_REQUIRE(index->SaveIndex("insert_test_index") == ErrorCode::Success);
    index = nullptr;

    std::shared_ptr<VectorIndex> loadedOnce;
    BOOST_REQUIRE(VectorIndex::LoadIndex("insert_test_index", loadedOnce) == ErrorCode::Success);
    BOOST_REQUIRE(loadedOnce != nullptr);

    InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t>*>(loadedOnce.get()), 2, 1000, addvecset, addmetaset);
    SearchOnly<int8_t>(loadedOnce, queryset, K);
    loadedOnce = nullptr;

    std::filesystem::remove_all("insert_test_index");
}

BOOST_AUTO_TEST_CASE(TestClone)
{
    using namespace SPFreshTest;

    // Prepare test data using TestDataGenerator
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, 10, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);

    auto clonedIndex = originalIndex->Clone("cloned_index");
    BOOST_REQUIRE(clonedIndex != nullptr);
    BOOST_REQUIRE(clonedIndex->SaveIndex("cloned_index") == ErrorCode::Success);
    originalIndex.reset();
    clonedIndex = nullptr;

    std::unordered_set<std::string> exceptions = {
        "indexloader.ini"
    };

    // Compare files in both directories
    BOOST_REQUIRE_MESSAGE(
        CompareDirectoriesWithLogging("original_index", "cloned_index", exceptions),
        "Saved index does not match loaded-then-saved index"
    );

    std::filesystem::remove_all("original_index");
    std::filesystem::remove_all("cloned_index");
}

BOOST_AUTO_TEST_CASE(TestCloneRecall)
{
    using namespace SPFreshTest;

    // Prepare test data using TestDataGenerator
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, 10, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    float originalRecall = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N);
    
    auto clonedIndex = originalIndex->Clone("cloned_index");
    BOOST_REQUIRE(clonedIndex != nullptr);
    originalIndex.reset();
    clonedIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedClonedIndex;
    BOOST_REQUIRE(VectorIndex::LoadIndex("cloned_index", loadedClonedIndex) == ErrorCode::Success);
    BOOST_REQUIRE(loadedClonedIndex != nullptr);
    float clonedRecall = Search<int8_t>(loadedClonedIndex, queryset, vecset, addvecset, K, truth, N);
    loadedClonedIndex = nullptr;

    BOOST_REQUIRE_MESSAGE(std::fabs(originalRecall - clonedRecall) < 1e-5,
        "Recall mismatch between original and cloned index: "
        << "original=" << originalRecall << ", cloned=" << clonedRecall);

    std::filesystem::remove_all("original_index");
    std::filesystem::remove_all("cloned_index");
}

BOOST_AUTO_TEST_CASE(IndexPersistenceAndInsertSanity)
{
    using namespace SPFreshTest;

    // Prepare test data
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, 10, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    // Build and save base index
    auto baseIndex = BuildIndex<int8_t>("insert_test_index", vecset, metaset);
    BOOST_REQUIRE(baseIndex != nullptr);
    BOOST_REQUIRE(baseIndex->SaveIndex("insert_test_index") == ErrorCode::Success);
    baseIndex = nullptr;

    // Load the saved index
    std::shared_ptr<VectorIndex> loadedOnce;
    BOOST_REQUIRE(VectorIndex::LoadIndex("insert_test_index", loadedOnce) == ErrorCode::Success);
    BOOST_REQUIRE(loadedOnce != nullptr);

    // Search sanity check
    SearchOnly<int8_t>(loadedOnce, queryset, K);
    
    // Clone the loaded index
    auto clonedIndex = loadedOnce->Clone("insert_cloned_index");
    BOOST_REQUIRE(clonedIndex != nullptr);

    // Save and reload the cloned index
    BOOST_REQUIRE(clonedIndex->SaveIndex("insert_cloned_index") == ErrorCode::Success);
    loadedOnce.reset();
    clonedIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedClone;
    BOOST_REQUIRE(VectorIndex::LoadIndex("insert_cloned_index", loadedClone) == ErrorCode::Success);
    BOOST_REQUIRE(loadedClone != nullptr);

    // Insert new vectors
    InsertVectors<int8_t>(
        static_cast<SPTAG::SPANN::Index<int8_t>*>(loadedClone.get()),
        1,
        static_cast<int>(addvecset->Count()),
        addvecset,
        addmetaset
    );

    // Final save and reload after insert
    BOOST_REQUIRE(loadedClone->SaveIndex("insert_final_index") == ErrorCode::Success);
    loadedClone = nullptr;

    std::shared_ptr<VectorIndex> reloadedFinal;
    BOOST_REQUIRE(VectorIndex::LoadIndex("insert_final_index", reloadedFinal) == ErrorCode::Success);

    // Final search sanity
    SearchOnly<int8_t>(reloadedFinal, queryset, K);
    reloadedFinal = nullptr;

    // Cleanup
    std::filesystem::remove_all("insert_test_index");
    std::filesystem::remove_all("insert_cloned_index");
    std::filesystem::remove_all("insert_final_index");
}

BOOST_AUTO_TEST_CASE(IndexPersistenceAndInsertMultipleThreads)
{
    using namespace SPFreshTest;

    // Prepare test data
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, 10, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    // Build and save base index
    auto baseIndex = BuildIndex<int8_t>("insert_test_index_multi", vecset, metaset);
    BOOST_REQUIRE(baseIndex != nullptr);
    BOOST_REQUIRE(baseIndex->SaveIndex("insert_test_index_multi") == ErrorCode::Success);
    baseIndex = nullptr;

    // Load the saved index
    std::shared_ptr<VectorIndex> loadedOnce;
    BOOST_REQUIRE(VectorIndex::LoadIndex("insert_test_index_multi", loadedOnce) == ErrorCode::Success);
    BOOST_REQUIRE(loadedOnce != nullptr);

    // Search sanity check
    SearchOnly<int8_t>(loadedOnce, queryset, K);
    
    // Clone the loaded index
    auto clonedIndex = loadedOnce->Clone("insert_cloned_index_multi");
    BOOST_REQUIRE(clonedIndex != nullptr);

    // Save and reload the cloned index
    BOOST_REQUIRE(clonedIndex->SaveIndex("insert_cloned_index_multi") == ErrorCode::Success);
    loadedOnce.reset();
    clonedIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedClone;
    BOOST_REQUIRE(VectorIndex::LoadIndex("insert_cloned_index_multi", loadedClone) == ErrorCode::Success);
    BOOST_REQUIRE(loadedClone != nullptr);

    // Insert new vectors
    InsertVectors<int8_t>(
        static_cast<SPTAG::SPANN::Index<int8_t>*>(loadedClone.get()),
        2,
        static_cast<int>(addvecset->Count()),
        addvecset,
        addmetaset
    );

    // Final save and reload after insert
    BOOST_REQUIRE(loadedClone->SaveIndex("insert_final_index_multi") == ErrorCode::Success);
    loadedClone = nullptr;

    std::shared_ptr<VectorIndex> reloadedFinal;
    BOOST_REQUIRE(VectorIndex::LoadIndex("insert_final_index_multi", reloadedFinal) == ErrorCode::Success);
    BOOST_REQUIRE(reloadedFinal != nullptr);
    // Final search sanity
    SearchOnly<int8_t>(reloadedFinal, queryset, K);
    reloadedFinal = nullptr;

    // Cleanup
    std::filesystem::remove_all("insert_test_index_multi");
    std::filesystem::remove_all("insert_cloned_index_multi");
    std::filesystem::remove_all("insert_final_index_multi");
}

BOOST_AUTO_TEST_CASE(IndexSaveDuringQuery)
{
    using namespace SPFreshTest;

    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, 10, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    auto index = BuildIndex<int8_t>("save_during_query_index", vecset, metaset);
    BOOST_REQUIRE(index != nullptr);

    std::atomic<bool> keepQuerying(true);
    std::thread queryThread([&]() {
        while (keepQuerying) {
            for (int q = 0; q < queryset->Count(); ++q) {
                QueryResult result(queryset->GetVector(q), K, true);
                index->SearchIndex(result);
            }
        }
        });

    // Wait a bit before saving
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ErrorCode saveStatus = index->SaveIndex("save_during_query_index");
    BOOST_REQUIRE(saveStatus == ErrorCode::Success);

    keepQuerying = false;
    queryThread.join();

    index = nullptr;

    std::shared_ptr<VectorIndex> reloaded;
    BOOST_REQUIRE(VectorIndex::LoadIndex("save_during_query_index", reloaded) == ErrorCode::Success);
    BOOST_REQUIRE(reloaded != nullptr);

    SearchOnly<int8_t>(reloaded, queryset, K);
    reloaded = nullptr;

    std::filesystem::remove_all("save_during_query_index");
}

BOOST_AUTO_TEST_CASE(IndexMultiThreadedQuerySanity)
{
    using namespace SPFreshTest;

    // Generate test data
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, 10, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    // Build and save index
    auto index = BuildIndex<int8_t>("multi_query_index", vecset, metaset);
    BOOST_REQUIRE(index != nullptr);
    BOOST_REQUIRE(index->SaveIndex("multi_query_index") == ErrorCode::Success);
    index = nullptr;

    // Reload the index
    std::shared_ptr<VectorIndex> loaded;
    BOOST_REQUIRE(VectorIndex::LoadIndex("multi_query_index", loaded) == ErrorCode::Success);
    BOOST_REQUIRE(loaded != nullptr);

    // Insert additional vectors
    InsertVectors<int8_t>(
        static_cast<SPTAG::SPANN::Index<int8_t>*>(loaded.get()),
        2,
        static_cast<int>(addvecset->Count()),
        addvecset,
        addmetaset
    );

    // Perform multithreaded query
    const int threadCount = 4;
    std::vector<std::thread> threads;
    std::atomic<int> nextQuery(0);
    std::atomic<int> completedQueries(0);

    for (int t = 0; t < threadCount; ++t) {
        threads.emplace_back([&, t]() {
            QueryResult result(nullptr, K, true);
            while (true) {
                int i = nextQuery.fetch_add(1);
                if (i >= queryset->Count()) break;

                result.SetTarget(queryset->GetVector(static_cast<SPTAG::SizeType>(i)));
                loaded->SearchIndex(result);

                ++completedQueries;
            }
            });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Multithreaded query completed: %d queries\n", completedQueries.load());
    loaded = nullptr;

    // Cleanup
    std::filesystem::remove_all("multi_query_index");
}

BOOST_AUTO_TEST_CASE(IndexShadowCloneLifecycleKeepLast)
{
    using namespace SPFreshTest;

    constexpr int iterations = 5;
    constexpr int insertBatchSize = 100;

    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, 10, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    const std::string baseIndexName = "base_index";
    BOOST_REQUIRE(BuildIndex<int8_t>(baseIndexName, vecset, metaset)->SaveIndex(baseIndexName) == ErrorCode::Success);

    std::string previousIndexName = baseIndexName;

    for (int iter = 0; iter < iterations; ++iter)
    {
        std::string shadowIndexName = "shadow_index_" + std::to_string(iter);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[%d] Loading index: %s\n", iter, previousIndexName.c_str());

        // Load previous index
        std::shared_ptr<VectorIndex> loaded;
        BOOST_REQUIRE(VectorIndex::LoadIndex(previousIndexName, loaded) == ErrorCode::Success);
        BOOST_REQUIRE(loaded != nullptr);

        // Query check
        for (int i = 0; i < std::min<SizeType>(queryset->Count(), 5); ++i) {
            QueryResult result(queryset->GetVector(i), K, true);
            loaded->SearchIndex(result);
        }

        // Cleanup previous base index after first iteration
        if (iter == 1) {
            std::filesystem::remove_all(baseIndexName);
        }
        
        // Clone to shadow
        BOOST_REQUIRE(loaded->Clone(shadowIndexName) != nullptr);
        loaded.reset();

        std::shared_ptr<VectorIndex> shadowLoaded;
        BOOST_REQUIRE(VectorIndex::LoadIndex(shadowIndexName, shadowLoaded) == ErrorCode::Success);
        BOOST_REQUIRE(shadowLoaded != nullptr);
	    auto* shadowIndex = static_cast<SPTAG::SPANN::Index<int8_t>*>(shadowLoaded.get());

        // Prepare insert batch
        const int insertOffset = (iter * insertBatchSize) % static_cast<int>(addvecset->Count());
        const int insertCount = min(insertBatchSize, static_cast<int>(addvecset->Count()) - insertOffset);

        std::vector<std::uint8_t> metaBytes;
        std::vector<std::uint64_t> offsetTable(insertCount + 1);
        std::uint64_t offset = 0;
        for (int i = 0; i < insertCount; ++i) {
            ByteArray meta = addmetaset->GetMetadata(insertOffset + i);
            offsetTable[i] = offset;
            metaBytes.insert(metaBytes.end(), meta.Data(), meta.Data() + meta.Length());
            offset += meta.Length();
        }
        offsetTable[insertCount] = offset;

        ByteArray metaBuf(new std::uint8_t[metaBytes.size()], metaBytes.size(), true);
        std::memcpy(metaBuf.Data(), metaBytes.data(), metaBytes.size());

        ByteArray offsetBuf(new std::uint8_t[offsetTable.size() * sizeof(std::uint64_t)], offsetTable.size() * sizeof(std::uint64_t), true);
        std::memcpy(offsetBuf.Data(), offsetTable.data(), offsetTable.size() * sizeof(std::uint64_t));

        auto batchMeta = std::make_shared<SPTAG::MemMetadataSet>(metaBuf, offsetBuf, insertCount);
        const void* vectorStart = addvecset->GetVector(insertOffset);

        shadowIndex->AddIndex(vectorStart, insertCount, shadowIndex->GetOptions()->m_dim, batchMeta, true);

        while (!shadowIndex->AllFinished()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        BOOST_REQUIRE(shadowLoaded->SaveIndex(shadowIndexName) == ErrorCode::Success);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[%d] Created new shadow index: %s\n", iter, shadowIndexName.c_str());
        shadowLoaded = nullptr;

        previousIndexName = shadowIndexName;
    }

    // Keep the final shadow index directory for debugging/inspection
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Kept final index: %s\n", previousIndexName.c_str());

    // Cleanup all created indexes after test
    std::filesystem::remove_all(baseIndexName);
    for (int iter = 0; iter < iterations; ++iter) {
        std::string shadow = "shadow_index_" + std::to_string(iter);
        std::filesystem::remove_all(shadow);
    }
}

BOOST_AUTO_TEST_CASE(IterativeSearch)
{
    using namespace SPFreshTest;

    constexpr int insertIterations = 5;
    constexpr int insertBatchSize = 1000;
    constexpr int dimension = 1024;
    std::shared_ptr<SPTAG::VectorSet> vecset = get_embeddings<float>(0, insertBatchSize, dimension, -1);
    std::shared_ptr<SPTAG::MetadataSet> metaset = TestUtils::TestDataGenerator<float>::GenerateMetadataSet(insertBatchSize, 0);

    auto originalIndex = BuildIndex<float>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    originalIndex = nullptr;

    std::string prevPath = "original_index";
    for (int iter = 0; iter < insertIterations; iter++) {
        std::string clone_path = "clone_index_" + std::to_string(iter);
        std::shared_ptr<SPTAG::VectorIndex> prevIndex;
        BOOST_REQUIRE(VectorIndex::LoadIndex(prevPath, prevIndex) == ErrorCode::Success);
        BOOST_REQUIRE(prevIndex != nullptr);
	    
        auto cloneIndex = prevIndex->Clone(clone_path);
        auto* cloneIndexPtr = static_cast<SPTAG::SPANN::Index<float>*>(cloneIndex.get());
        std::shared_ptr<SPTAG::VectorSet> tmpvecs = get_embeddings<float>((iter + 1) * insertBatchSize, (iter + 2) * insertBatchSize, dimension, -1);
        std::shared_ptr<SPTAG::MetadataSet> tmpmetas = TestUtils::TestDataGenerator<float>::GenerateMetadataSet(insertBatchSize, (iter + 1) * insertBatchSize);
        InsertVectors<float>(cloneIndexPtr, 1, insertBatchSize, tmpvecs, tmpmetas);

        BOOST_REQUIRE(cloneIndex->SaveIndex(clone_path) == ErrorCode::Success);
        cloneIndex = nullptr;

        std::shared_ptr<VectorIndex> loadedIndex;
        BOOST_REQUIRE(VectorIndex::LoadIndex(clone_path, loadedIndex) == ErrorCode::Success);
        BOOST_REQUIRE(loadedIndex != nullptr);

        std::shared_ptr<SPTAG::VectorSet> embedding = get_embeddings<float>((1000 * iter) + 500, ((1000 * iter) + 501), dimension, -1);
        std::shared_ptr<ResultIterator> resultIterator = loadedIndex->GetIterator(embedding->GetData(), false);
        int batch = 100;
        int ri = 0;
        float current = INT_MAX, previous = INT_MAX;
        bool relaxMono = false;
        while (!relaxMono) {
            auto results = resultIterator->Next(batch);
            int resultCount = results->GetResultNum();
            if (resultCount <= 0) break;
            BOOST_CHECK(resultCount == batch);
            previous = current;
            current = 0;
            for (int j = 0; j < resultCount; j++) {
                std::cout << "Result[" << ri << "] VID:" << results->GetResult(j)->VID << " Dist:" << results->GetResult(j)->Dist << " RelaxedMono:"
                    << results->GetResult(j)->RelaxedMono << " current:" << current << " previous:" << previous << std::endl;
                relaxMono = results->GetResult(j)->RelaxedMono;
                current += results->GetResult(j)->Dist;
                ri++;
            }
            current /= resultCount;
        }
        resultIterator->Close();
        loadedIndex = nullptr;
    }
    
    for (int iter = 0; iter < insertIterations; iter++) {
        std::filesystem::remove_all("clone_index_" + std::to_string(iter));
    }
    std::filesystem::remove_all("original_index");
}

BOOST_AUTO_TEST_SUITE_END()
