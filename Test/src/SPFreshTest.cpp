// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/SPANNResultIterator.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/DiskIO.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Test.h"
#include "inc/TestDataGenerator.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using namespace SPTAG;

namespace SPFreshTest
{
SizeType N = 1000;
DimensionType M = 100;
int K = 10;
int queries = 10;

template <typename T>
std::shared_ptr<VectorIndex> BuildIndex(const std::string &outDirectory, std::shared_ptr<VectorSet> vecset,
                                        std::shared_ptr<MetadataSet> metaset, const std::string &distMethod = "L2")
{
    auto vecIndex = VectorIndex::CreateInstance(IndexAlgoType::SPANN, GetEnumValueType<T>());

    std::string configuration = R"(
        [Base]
            DistCalcMethod=L2
            IndexAlgoType=BKT
            ValueType=)" + Helper::Convert::ConvertToString(GetEnumValueType<T>()) +
                                R"(
            Dim=)" + std::to_string(M) +
                                R"(
            IndexDirectory=)" + outDirectory +
                                R"(

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
	    PostingPageLimit=)" + std::to_string(4 * sizeof(T)) +
                                R"(
            SearchPostingPageLimit=)" +
                                std::to_string(4 * sizeof(T)) + R"(
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
            OneClusterCutMax=true
        )";

    std::shared_ptr<Helper::DiskIO> buffer(new Helper::SimpleBufferIO());
    Helper::IniReader reader;
    if (!buffer->Initialize(configuration.data(), std::ios::in, configuration.size()))
        return nullptr;
    if (ErrorCode::Success != reader.LoadIni(buffer))
        return nullptr;

    std::string sections[] = {"Base", "SelectHead", "BuildHead", "BuildSSDIndex"};
    for (const auto &sec : sections)
    {
        auto params = reader.GetParameters(sec.c_str());
        for (const auto &[key, val] : params)
        {
            vecIndex->SetParameter(key.c_str(), val.c_str(), sec.c_str());
        }
    }

    auto buildStatus = vecIndex->BuildIndex(vecset, metaset, true, false, false);
    if (buildStatus != ErrorCode::Success)
        return nullptr;

    return vecIndex;
}

template <typename T>
std::vector<QueryResult> SearchOnly(std::shared_ptr<VectorIndex> &vecIndex, std::shared_ptr<VectorSet> &queryset, int k)
{
    std::vector<QueryResult> res(queryset->Count(), QueryResult(nullptr, k, true));

    auto t1 = std::chrono::high_resolution_clock::now();
    for (SizeType i = 0; i < queryset->Count(); i++)
    {
        res[i].SetTarget(queryset->GetVector(i));
        vecIndex->SearchIndex(res[i]);
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    float avgUs =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / static_cast<float>(queryset->Count());
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Avg search time: %.2fus/query\n", avgUs);

    return res;
}

template <typename T>
float EvaluateRecall(const std::vector<QueryResult> &res, std::shared_ptr<VectorIndex> &vecIndex,
                     std::shared_ptr<VectorSet> &queryset, std::shared_ptr<VectorSet> &truth,
                     std::shared_ptr<VectorSet> &baseVec, std::shared_ptr<VectorSet> &addVec, SizeType baseCount, int k,
                     int batch)
{
    if (!truth)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Truth data is null. Cannot compute recall.\n");
        return 0.0f;
    }

    const SizeType recallK = min(k, static_cast<int>(truth->Dimension()));
    float totalRecall = 0.0f;
    float eps = 1e-4f;

    for (SizeType i = 0; i < queryset->Count(); ++i)
    {
        const SizeType *truthNN = reinterpret_cast<const SizeType *>(truth->GetVector(i + batch * queryset->Count()));
        for (int j = 0; j < recallK; ++j)
        {
            SizeType truthVid = truthNN[j];
            float truthDist =
                (truthVid < baseCount)
                    ? vecIndex->ComputeDistance(queryset->GetVector(i), baseVec->GetVector(truthVid))
                    : vecIndex->ComputeDistance(queryset->GetVector(i), addVec->GetVector(truthVid - baseCount));

            for (int l = 0; l < k; ++l)
            {
                const auto result = res[i].GetResult(l);
                if (truthVid == result->VID ||
                    std::fabs(truthDist - result->Dist) <= eps * (std::fabs(truthDist) + eps))
                {
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
float Search(std::shared_ptr<VectorIndex> &vecIndex, std::shared_ptr<VectorSet> &queryset,
             std::shared_ptr<VectorSet> &baseVec, std::shared_ptr<VectorSet> &addVec, int k,
             std::shared_ptr<VectorSet> &truth, SizeType baseCount, int batch = 0)
{
    auto results = SearchOnly<T>(vecIndex, queryset, k);
    return EvaluateRecall<T>(results, vecIndex, queryset, truth, baseVec, addVec, baseCount, k, batch);
}

template <typename ValueType>
void InsertVectors(SPANN::Index<ValueType> *p_index, int insertThreads, int step, std::shared_ptr<VectorSet> addset,
                   std::shared_ptr<MetadataSet> &metaset, int start = 0)
{
    SPANN::Options &p_opts = *(p_index->GetOptions());
    p_index->ForceCompaction();
    p_index->GetDBStat();

    std::vector<std::thread> threads;

    std::atomic_size_t vectorsSent(start);
    auto func = [&]() {
        size_t index = start;
        while (true)
        {
            index = vectorsSent.fetch_add(1);
            if (index < start + step)
            {
                if ((index & ((1 << 5) - 1)) == 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Sent %.2lf%%...\n", index * 100.0 / step);
                }
                ByteArray p_meta = metaset->GetMetadata((SizeType)index);
                std::uint64_t *offsets = new std::uint64_t[2]{0, p_meta.Length()};
                std::shared_ptr<MetadataSet> meta(
                    new MemMetadataSet(p_meta, ByteArray((std::uint8_t *)offsets, 2 * sizeof(std::uint64_t), true), 1));
                auto status = p_index->AddIndex(addset->GetVector((SizeType)index), 1, p_opts.m_dim, meta, true);
                ASSERT_EQ(status, ErrorCode::Success);
            }
            else
            {
                return;
            }
        }
    };
    for (int j = 0; j < insertThreads; j++)
    {
        threads.emplace_back(func);
    }
    for (auto &thread : threads)
    {
        thread.join();
    }

    while (!p_index->AllFinished())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}
} // namespace SPFreshTest

bool CompareFilesWithLogging(const std::filesystem::path &file1, const std::filesystem::path &file2)
{
    std::ifstream f1(file1, std::ios::binary);
    std::ifstream f2(file2, std::ios::binary);

    if (!f1.is_open() || !f2.is_open())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open one of the files:\n  %s\n  %s\n",
                     file1.string().c_str(), file2.string().c_str());
        return false;
    }

    // Check file sizes first
    f1.seekg(0, std::ios::end);
    f2.seekg(0, std::ios::end);
    if (f1.tellg() != f2.tellg())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File size differs: %s\n", file1.filename().string().c_str());
        return false;
    }

    f1.seekg(0, std::ios::beg);
    f2.seekg(0, std::ios::beg);

    const int bufferSize = 4096; // Adjust buffer size as needed
    std::vector<char> buffer1(bufferSize);
    std::vector<char> buffer2(bufferSize);

    while (f1.read(buffer1.data(), bufferSize) && f2.read(buffer2.data(), bufferSize))
    {
        if (std::memcmp(buffer1.data(), buffer2.data(), f1.gcount()) != 0)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File mismatch at: %s\n", file1.filename().string().c_str());
            return false; // Mismatch found
        }
    }

    return true;
}

bool CompareDirectoriesWithLogging(const std::filesystem::path &dir1, const std::filesystem::path &dir2,
                                   const std::unordered_set<std::string> &exceptions = {})
{
    std::map<std::string, std::filesystem::path> files1, files2;

    for (const auto &entry : std::filesystem::recursive_directory_iterator(dir1))
    {
        if (entry.is_regular_file())
        {
            files1[std::filesystem::relative(entry.path(), dir1).string()] = entry.path();
        }
    }

    for (const auto &entry : std::filesystem::recursive_directory_iterator(dir2))
    {
        if (entry.is_regular_file())
        {
            files2[std::filesystem::relative(entry.path(), dir2).string()] = entry.path();
        }
    }

    bool matched = true;

    for (const auto &[relPath, filePath1] : files1)
    {
        if (exceptions.count(relPath))
            continue;

        auto it = files2.find(relPath);
        if (it == files2.end())
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Missing in %s: %s\n", dir2.string().c_str(), relPath.c_str());
            matched = false;
            continue;
        }
        if (!CompareFilesWithLogging(filePath1, it->second))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File end differs: %s\n", filePath1.filename().string().c_str());
            matched = false;
        }
    }

    for (const auto &[relPath, _] : files2)
    {
        if (exceptions.count(relPath))
            continue;
        if (files1.find(relPath) == files1.end())
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Extra in %s: %s\n", dir2.string().c_str(), relPath.c_str());
            matched = false;
        }
    }

    return matched;
}

void NormalizeVector(float *embedding, int dimension)
{
    // get magnitude
    float magnitude = 0.0f;
    {
        float sum = 0.0;
        for (int i = 0; i < dimension; i++)
        {
            sum += embedding[i] * embedding[i];
        }
        magnitude = std::sqrt(sum);
    }

    // normalized target vector
    for (int i = 0; i < dimension; i++)
    {
        embedding[i] /= magnitude;
    }
}

template <typename T>
std::shared_ptr<VectorSet> get_embeddings(uint32_t row_id, uint32_t end_id, uint32_t embedding_dim,
                                          uint32_t array_index)
{
    uint32_t count = end_id - row_id;
    ByteArray vec = ByteArray::Alloc(sizeof(T) * count * embedding_dim);
    for (uint32_t rid = 0; rid < count; rid++)
    {
        for (int idx = 0; idx < embedding_dim; ++idx)
        {
            ((T *)vec.Data())[rid * embedding_dim + idx] = (row_id + rid) * 17 + idx * 19 + (array_index + 1) * 23;
        }
        NormalizeVector(((T *)vec.Data()) + rid * embedding_dim, embedding_dim);
    }
    return std::make_shared<BasicVectorSet>(vec, GetEnumValueType<T>(), embedding_dim, count);
}

namespace SPFreshTestSuite
{

TEST(SPFreshTest, TestLoadAndSave)
{
    using namespace SPFreshTest;

    // Prepare test data using TestDataGenerator
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    ASSERT_NE(originalIndex, nullptr);
    ASSERT_EQ(originalIndex->SaveIndex("original_index"), ErrorCode::Success);
    originalIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedIndex;
    ASSERT_EQ(VectorIndex::LoadIndex("original_index", loadedIndex), ErrorCode::Success);
    ASSERT_NE(loadedIndex, nullptr);
    ASSERT_EQ(loadedIndex->SaveIndex("loaded_and_saved_index"), ErrorCode::Success);
    loadedIndex = nullptr;

    std::unordered_set<std::string> exceptions = {"indexloader.ini"};

    // Compare files in both directories
    ASSERT_TRUE(CompareDirectoriesWithLogging("original_index", "loaded_and_saved_index", exceptions))
        << "Saved index does not match loaded-then-saved index";

    std::filesystem::remove_all("original_index");
    std::filesystem::remove_all("loaded_and_saved_index");
}

TEST(SPFreshTest, TestReopenIndexRecall)
{
    using namespace SPFreshTest;

    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    ASSERT_NE(originalIndex, nullptr);
    ASSERT_EQ(originalIndex->SaveIndex("original_index"), ErrorCode::Success);
    float recall1 = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N);
    originalIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedOnce;
    ASSERT_EQ(VectorIndex::LoadIndex("original_index", loadedOnce), ErrorCode::Success);
    ASSERT_NE(loadedOnce, nullptr);
    ASSERT_EQ(loadedOnce->SaveIndex("reopened_index"), ErrorCode::Success);
    loadedOnce = nullptr;

    std::shared_ptr<VectorIndex> loadedTwice;
    ASSERT_EQ(VectorIndex::LoadIndex("reopened_index", loadedTwice), ErrorCode::Success);
    ASSERT_NE(loadedTwice, nullptr);
    float recall2 = Search<int8_t>(loadedTwice, queryset, vecset, addvecset, K, truth, N);
    loadedTwice = nullptr;

    ASSERT_LT(std::fabs(recall1 - recall2), 1e-5) << "Recall mismatch between original and reopened index";

    std::filesystem::remove_all("original_index");
    std::filesystem::remove_all("reopened_index");
}

TEST(SPFreshTest, TestInsertAndSearch)
{
    using namespace SPFreshTest;

    // Prepare test data using TestDataGenerator
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    // Build base index
    auto index = BuildIndex<int8_t>("insert_test_index", vecset, metaset);
    ASSERT_NE(index, nullptr);
    ASSERT_EQ(index->SaveIndex("insert_test_index"), ErrorCode::Success);
    index = nullptr;

    std::shared_ptr<VectorIndex> loadedOnce;
    ASSERT_EQ(VectorIndex::LoadIndex("insert_test_index", loadedOnce), ErrorCode::Success);
    ASSERT_NE(loadedOnce, nullptr);

    InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(loadedOnce.get()), 2, 1000, addvecset, addmetaset);
    SearchOnly<int8_t>(loadedOnce, queryset, K);
    loadedOnce = nullptr;

    std::filesystem::remove_all("insert_test_index");
}

TEST(SPFreshTest, TestClone)
{
    using namespace SPFreshTest;

    // Prepare test data using TestDataGenerator
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    ASSERT_NE(originalIndex, nullptr);
    ASSERT_EQ(originalIndex->SaveIndex("original_index"), ErrorCode::Success);
    originalIndex = nullptr;

    auto clonedIndex = originalIndex->Clone("cloned_index");
    ASSERT_NE(clonedIndex, nullptr);
    ASSERT_EQ(clonedIndex->SaveIndex("cloned_index"), ErrorCode::Success);
    clonedIndex = nullptr;

    std::unordered_set<std::string> exceptions = {"indexloader.ini"};

    // Compare files in both directories
    ASSERT_TRUE(CompareDirectoriesWithLogging("original_index", "cloned_index", exceptions))
        << "Saved index does not match loaded-then-saved index";

    std::filesystem::remove_all("original_index");
    std::filesystem::remove_all("cloned_index");
}

TEST(SPFreshTest, TestCloneRecall)
{
    using namespace SPFreshTest;

    // Prepare test data using TestDataGenerator
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    ASSERT_NE(originalIndex, nullptr);
    ASSERT_EQ(originalIndex->SaveIndex("original_index"), ErrorCode::Success);
    float originalRecall = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N);

    auto clonedIndex = originalIndex->Clone("cloned_index");
    ASSERT_NE(clonedIndex, nullptr);
    originalIndex.reset();
    clonedIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedClonedIndex;
    ASSERT_EQ(VectorIndex::LoadIndex("cloned_index", loadedClonedIndex), ErrorCode::Success);
    ASSERT_NE(loadedClonedIndex, nullptr);
    float clonedRecall = Search<int8_t>(loadedClonedIndex, queryset, vecset, addvecset, K, truth, N);
    loadedClonedIndex = nullptr;

    ASSERT_LT(std::fabs(originalRecall - clonedRecall), 1e-5)
        << "Recall mismatch between original and cloned index: "
        << "original=" << originalRecall << ", cloned=" << clonedRecall;

    std::filesystem::remove_all("original_index");
    std::filesystem::remove_all("cloned_index");
}

TEST(SPFreshTest, IndexPersistenceAndInsertSanity)
{
    using namespace SPFreshTest;

    // Prepare test data
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    // Build and save base index
    auto baseIndex = BuildIndex<int8_t>("insert_test_index", vecset, metaset);
    ASSERT_NE(baseIndex, nullptr);
    ASSERT_EQ(baseIndex->SaveIndex("insert_test_index"), ErrorCode::Success);
    baseIndex = nullptr;

    // Load the saved index
    std::shared_ptr<VectorIndex> loadedOnce;
    ASSERT_EQ(VectorIndex::LoadIndex("insert_test_index", loadedOnce), ErrorCode::Success);
    ASSERT_NE(loadedOnce, nullptr);

    // Search sanity check
    SearchOnly<int8_t>(loadedOnce, queryset, K);

    // Clone the loaded index
    auto clonedIndex = loadedOnce->Clone("insert_cloned_index");
    ASSERT_NE(clonedIndex, nullptr);

    // Save and reload the cloned index
    ASSERT_EQ(clonedIndex->SaveIndex("insert_cloned_index"), ErrorCode::Success);
    loadedOnce.reset();
    clonedIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedClone;
    ASSERT_EQ(VectorIndex::LoadIndex("insert_cloned_index", loadedClone), ErrorCode::Success);
    ASSERT_NE(loadedClone, nullptr);

    // Insert new vectors
    InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(loadedClone.get()), 1,
                          static_cast<int>(addvecset->Count()), addvecset, addmetaset);

    // Final save and reload after insert
    ASSERT_EQ(loadedClone->SaveIndex("insert_final_index"), ErrorCode::Success);
    loadedClone = nullptr;

    std::shared_ptr<VectorIndex> reloadedFinal;
    ASSERT_EQ(VectorIndex::LoadIndex("insert_final_index", reloadedFinal), ErrorCode::Success);

    // Final search sanity
    SearchOnly<int8_t>(reloadedFinal, queryset, K);
    reloadedFinal = nullptr;

    // Cleanup
    std::filesystem::remove_all("insert_test_index");
    std::filesystem::remove_all("insert_cloned_index");
    std::filesystem::remove_all("insert_final_index");
}

TEST(SPFreshTest, IndexPersistenceAndInsertMultipleThreads)
{
    using namespace SPFreshTest;

    // Prepare test data
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    // Build and save base index
    auto baseIndex = BuildIndex<int8_t>("insert_test_index_multi", vecset, metaset);
    ASSERT_NE(baseIndex, nullptr);
    ASSERT_EQ(baseIndex->SaveIndex("insert_test_index_multi"), ErrorCode::Success);
    baseIndex = nullptr;

    // Load the saved index
    std::shared_ptr<VectorIndex> loadedOnce;
    ASSERT_EQ(VectorIndex::LoadIndex("insert_test_index_multi", loadedOnce), ErrorCode::Success);
    ASSERT_NE(loadedOnce, nullptr);

    // Search sanity check
    SearchOnly<int8_t>(loadedOnce, queryset, K);

    // Clone the loaded index
    auto clonedIndex = loadedOnce->Clone("insert_cloned_index_multi");
    ASSERT_NE(clonedIndex, nullptr);

    // Save and reload the cloned index
    ASSERT_EQ(clonedIndex->SaveIndex("insert_cloned_index_multi"), ErrorCode::Success);
    loadedOnce.reset();
    clonedIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedClone;
    ASSERT_EQ(VectorIndex::LoadIndex("insert_cloned_index_multi", loadedClone), ErrorCode::Success);
    ASSERT_NE(loadedClone, nullptr);

    // Insert new vectors
    InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(loadedClone.get()), 2,
                          static_cast<int>(addvecset->Count()), addvecset, addmetaset);

    // Final save and reload after insert
    ASSERT_EQ(loadedClone->SaveIndex("insert_final_index_multi"), ErrorCode::Success);
    loadedClone = nullptr;

    std::shared_ptr<VectorIndex> reloadedFinal;
    ASSERT_EQ(VectorIndex::LoadIndex("insert_final_index_multi", reloadedFinal), ErrorCode::Success);
    ASSERT_NE(reloadedFinal, nullptr);
    // Final search sanity
    SearchOnly<int8_t>(reloadedFinal, queryset, K);
    reloadedFinal = nullptr;

    // Cleanup
    std::filesystem::remove_all("insert_test_index_multi");
    std::filesystem::remove_all("insert_cloned_index_multi");
    std::filesystem::remove_all("insert_final_index_multi");
}

TEST(SPFreshTest, IndexSaveDuringQuery)
{
    using namespace SPFreshTest;

    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    auto index = BuildIndex<int8_t>("save_during_query_index", vecset, metaset);
    ASSERT_NE(index, nullptr);

    std::atomic<bool> keepQuerying(true);
    std::thread queryThread([&]() {
        while (keepQuerying)
        {
            for (int q = 0; q < queryset->Count(); ++q)
            {
                QueryResult result(queryset->GetVector(q), K, true);
                index->SearchIndex(result);
            }
        }
    });

    // Wait a bit before saving
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ErrorCode saveStatus = index->SaveIndex("save_during_query_index");
    ASSERT_EQ(saveStatus, ErrorCode::Success);

    keepQuerying = false;
    queryThread.join();

    index = nullptr;

    std::shared_ptr<VectorIndex> reloaded;
    ASSERT_EQ(VectorIndex::LoadIndex("save_during_query_index", reloaded), ErrorCode::Success);
    ASSERT_NE(reloaded, nullptr);

    SearchOnly<int8_t>(reloaded, queryset, K);
    reloaded = nullptr;

    std::filesystem::remove_all("save_during_query_index");
}

TEST(SPFreshTest, IndexMultiThreadedQuerySanity)
{
    using namespace SPFreshTest;

    // Generate test data
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    // Build and save index
    auto index = BuildIndex<int8_t>("multi_query_index", vecset, metaset);
    ASSERT_NE(index, nullptr);
    ASSERT_EQ(index->SaveIndex("multi_query_index"), ErrorCode::Success);
    index = nullptr;

    // Reload the index
    std::shared_ptr<VectorIndex> loaded;
    ASSERT_EQ(VectorIndex::LoadIndex("multi_query_index", loaded), ErrorCode::Success);
    ASSERT_NE(loaded, nullptr);

    // Insert additional vectors
    InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(loaded.get()), 2, static_cast<int>(addvecset->Count()),
                          addvecset, addmetaset);

    // Perform multithreaded query
    const int threadCount = 4;
    std::vector<std::thread> threads;
    std::atomic<int> nextQuery(0);
    std::atomic<int> completedQueries(0);

    for (int t = 0; t < threadCount; ++t)
    {
        threads.emplace_back([&, t]() {
            QueryResult result(nullptr, K, true);
            while (true)
            {
                int i = nextQuery.fetch_add(1);
                if (i >= queryset->Count())
                    break;

                result.SetTarget(queryset->GetVector(static_cast<SizeType>(i)));
                loaded->SearchIndex(result);

                ++completedQueries;
            }
        });
    }

    for (auto &thread : threads)
    {
        thread.join();
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Multithreaded query completed: %d queries\n", completedQueries.load());
    loaded = nullptr;

    // Cleanup
    std::filesystem::remove_all("multi_query_index");
}

TEST(SPFreshTest, IndexShadowCloneLifecycleKeepLast)
{
    using namespace SPFreshTest;

    constexpr int iterations = 5;
    constexpr int insertBatchSize = 100;

    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset, addtruth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.Run(vecset, metaset, queryset, truth, addvecset, addmetaset, addtruth);

    const std::string baseIndexName = "base_index";
    ASSERT_EQ(BuildIndex<int8_t>(baseIndexName, vecset, metaset)->SaveIndex(baseIndexName), ErrorCode::Success);

    std::string previousIndexName = baseIndexName;

    for (int iter = 0; iter < iterations; ++iter)
    {
        std::string shadowIndexName = "shadow_index_" + std::to_string(iter);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[%d] Loading index: %s\n", iter, previousIndexName.c_str());

        // Load previous index
        std::shared_ptr<VectorIndex> loaded;
        ASSERT_EQ(VectorIndex::LoadIndex(previousIndexName, loaded), ErrorCode::Success);
        ASSERT_NE(loaded, nullptr);

        // Query check
        for (int i = 0; i < std::min<SizeType>(queryset->Count(), 5); ++i)
        {
            QueryResult result(queryset->GetVector(i), K, true);
            loaded->SearchIndex(result);
        }

        // Cleanup previous base index after first iteration
        if (iter == 1)
        {
            std::filesystem::remove_all(baseIndexName);
        }

        // Clone to shadow
        ASSERT_NE(loaded->Clone(shadowIndexName), nullptr);
        loaded.reset();

        std::shared_ptr<VectorIndex> shadowLoaded;
        ASSERT_EQ(VectorIndex::LoadIndex(shadowIndexName, shadowLoaded), ErrorCode::Success);
        ASSERT_NE(shadowLoaded, nullptr);
        auto *shadowIndex = static_cast<SPANN::Index<int8_t> *>(shadowLoaded.get());

        // Prepare insert batch
        const int insertOffset = (iter * insertBatchSize) % static_cast<int>(addvecset->Count());
        const int insertCount = min(insertBatchSize, static_cast<int>(addvecset->Count()) - insertOffset);

        std::vector<std::uint8_t> metaBytes;
        std::vector<std::uint64_t> offsetTable(insertCount + 1);
        std::uint64_t offset = 0;
        for (int i = 0; i < insertCount; ++i)
        {
            ByteArray meta = addmetaset->GetMetadata(insertOffset + i);
            offsetTable[i] = offset;
            metaBytes.insert(metaBytes.end(), meta.Data(), meta.Data() + meta.Length());
            offset += meta.Length();
        }
        offsetTable[insertCount] = offset;

        ByteArray metaBuf(new std::uint8_t[metaBytes.size()], metaBytes.size(), true);
        std::memcpy(metaBuf.Data(), metaBytes.data(), metaBytes.size());

        ByteArray offsetBuf(new std::uint8_t[offsetTable.size() * sizeof(std::uint64_t)],
                            offsetTable.size() * sizeof(std::uint64_t), true);
        std::memcpy(offsetBuf.Data(), offsetTable.data(), offsetTable.size() * sizeof(std::uint64_t));

        auto batchMeta = std::make_shared<MemMetadataSet>(metaBuf, offsetBuf, insertCount);
        const void *vectorStart = addvecset->GetVector(insertOffset);

        shadowIndex->AddIndex(vectorStart, insertCount, shadowIndex->GetOptions()->m_dim, batchMeta, true);

        while (!shadowIndex->AllFinished())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        ASSERT_EQ(shadowLoaded->SaveIndex(shadowIndexName), ErrorCode::Success);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[%d] Created new shadow index: %s\n", iter, shadowIndexName.c_str());
        shadowLoaded = nullptr;

        previousIndexName = shadowIndexName;
    }

    // Keep the final shadow index directory for debugging/inspection
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Kept final index: %s\n", previousIndexName.c_str());

    // Cleanup all created indexes after test
    std::filesystem::remove_all(baseIndexName);
    for (int iter = 0; iter < iterations; ++iter)
    {
        std::string shadow = "shadow_index_" + std::to_string(iter);
        std::filesystem::remove_all(shadow);
    }
}

TEST(SPFreshTest, IterativeSearch)
{
    using namespace SPFreshTest;

    constexpr int insertIterations = 5;
    constexpr int insertBatchSize = 1000;
    constexpr int dimension = 1024;
    std::shared_ptr<VectorSet> vecset = get_embeddings<float>(0, insertBatchSize, dimension, -1);
    std::shared_ptr<MetadataSet> metaset = TestUtils::TestDataGenerator<float>::GenerateMetadataSet(insertBatchSize, 0);

    auto originalIndex = BuildIndex<float>("original_index", vecset, metaset);
    ASSERT_NE(originalIndex, nullptr);
    ASSERT_EQ(originalIndex->SaveIndex("original_index"), ErrorCode::Success);
    originalIndex = nullptr;

    std::string prevPath = "original_index";
    for (int iter = 0; iter < insertIterations; iter++)
    {
        std::string clone_path = "clone_index_" + std::to_string(iter);
        std::shared_ptr<VectorIndex> prevIndex;
        ASSERT_EQ(VectorIndex::LoadIndex(prevPath, prevIndex), ErrorCode::Success);
        ASSERT_NE(prevIndex, nullptr);

        auto cloneIndex = prevIndex->Clone(clone_path);
        auto *cloneIndexPtr = static_cast<SPANN::Index<float> *>(cloneIndex.get());
        std::shared_ptr<VectorSet> tmpvecs =
            get_embeddings<float>((iter + 1) * insertBatchSize, (iter + 2) * insertBatchSize, dimension, -1);
        std::shared_ptr<MetadataSet> tmpmetas =
            TestUtils::TestDataGenerator<float>::GenerateMetadataSet(insertBatchSize, (iter + 1) * insertBatchSize);
        InsertVectors<float>(cloneIndexPtr, 1, insertBatchSize, tmpvecs, tmpmetas);

        ASSERT_EQ(cloneIndex->SaveIndex(clone_path), ErrorCode::Success);
        cloneIndex = nullptr;

        std::shared_ptr<VectorIndex> loadedIndex;
        ASSERT_EQ(VectorIndex::LoadIndex(clone_path, loadedIndex), ErrorCode::Success);
        ASSERT_NE(loadedIndex, nullptr);

        std::shared_ptr<VectorSet> embedding =
            get_embeddings<float>((1000 * iter) + 500, ((1000 * iter) + 501), dimension, -1);
        std::shared_ptr<ResultIterator> resultIterator = loadedIndex->GetIterator(embedding->GetData(), false);
        int batch = 100;
        int ri = 0;
        float current = INT_MAX, previous = INT_MAX;
        bool relaxMono = false;
        while (!relaxMono)
        {
            auto results = resultIterator->Next(batch);
            int resultCount = results->GetResultNum();
            if (resultCount <= 0)
                break;
            EXPECT_EQ(resultCount, batch);
            previous = current;
            current = 0;
            for (int j = 0; j < resultCount; j++)
            {
                std::cout << "Result[" << ri << "] VID:" << results->GetResult(j)->VID
                          << " Dist:" << results->GetResult(j)->Dist
                          << " RelaxedMono:" << results->GetResult(j)->RelaxedMono << " current:" << current
                          << " previous:" << previous << std::endl;
                relaxMono = results->GetResult(j)->RelaxedMono;
                current += results->GetResult(j)->Dist;
                ri++;
            }
            current /= resultCount;
        }
        resultIterator->Close();
        loadedIndex = nullptr;
    }

    for (int iter = 0; iter < insertIterations; iter++)
    {
        std::filesystem::remove_all("clone_index_" + std::to_string(iter));
    }
    std::filesystem::remove_all("original_index");
}

TEST(SPFreshTest, RefineIndex)
{
    using namespace SPFreshTest;

    int iterations = 5;
    int insertBatchSize = N / iterations;
    int deleteBatchSize = N / iterations;

    // Generate test data
    std::shared_ptr<VectorSet> vecset, addvecset, queryset, truth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.RunBatches(vecset, metaset, addvecset, addmetaset, queryset, N, insertBatchSize, deleteBatchSize,
                         iterations, truth);

    // Build and save index
    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    ASSERT_NE(originalIndex, nullptr);
    ASSERT_EQ(originalIndex->SaveIndex("original_index"), ErrorCode::Success);

    float recall = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N);
    std::cout << "original: recall@5=" << recall << std::endl;

    for (int iter = 0; iter < iterations; iter++)
    {

        InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(originalIndex.get()), 1, insertBatchSize, addvecset,
                              metaset, iter * insertBatchSize);
        for (int i = 0; i < deleteBatchSize; i++)
            originalIndex->DeleteIndex(iter * deleteBatchSize + i);

        recall = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N, iter + 1);
        std::cout << "iter " << iter << ": recall@5=" << recall << std::endl;
    }
    std::cout << "Before Refine:" << " recall@5=" << recall << std::endl;
    static_cast<SPANN::Index<int8_t> *>(originalIndex.get())->GetDBStat();
    ASSERT_EQ(originalIndex->SaveIndex("original_index"), ErrorCode::Success);
    originalIndex = nullptr;

    ASSERT_EQ(VectorIndex::LoadIndex("original_index", originalIndex), ErrorCode::Success);
    ASSERT_NE(originalIndex, nullptr);
    ASSERT_EQ(originalIndex->Check(), ErrorCode::Success);

    recall = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N, iterations);
    std::cout << "After Refine:" << " recall@5=" << recall << std::endl;
    static_cast<SPANN::Index<int8_t> *>(originalIndex.get())->GetDBStat();
}

TEST(SPFreshTest, IterativeSearchPerf)
{
    using namespace SPFreshTest;

    constexpr int insertIterations = 5;
    constexpr int insertBatchSize = 60000;
    constexpr int appendBatchSize = 40000;
    constexpr int dimension = 100;
    std::shared_ptr<VectorSet> vecset = get_embeddings<float>(0, insertBatchSize, dimension, -1);
    std::shared_ptr<MetadataSet> metaset = TestUtils::TestDataGenerator<float>::GenerateMetadataSet(insertBatchSize, 0);

    auto originalIndex = BuildIndex<float>("original_index", vecset, metaset);
    ASSERT_NE(originalIndex, nullptr);
    ASSERT_EQ(originalIndex->SaveIndex("original_index"), ErrorCode::Success);
    originalIndex = nullptr;

    std::string prevPath = "original_index";
    for (int iter = 0; iter < insertIterations; iter++)
    {
        std::string clone_path = "clone_index_" + std::to_string(iter);
        std::shared_ptr<VectorIndex> prevIndex;
        ASSERT_EQ(VectorIndex::LoadIndex(prevPath, prevIndex), ErrorCode::Success);
        ASSERT_NE(prevIndex, nullptr);
        auto t0 = std::chrono::high_resolution_clock::now();
        ASSERT_EQ(prevIndex->Check(), ErrorCode::Success);
        std::cout << "Check time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                           t0)
                         .count()
                  << " ms" << std::endl;

        auto cloneIndex = prevIndex->Clone(clone_path);
        auto *cloneIndexPtr = static_cast<SPANN::Index<float> *>(cloneIndex.get());
        std::shared_ptr<VectorSet> tmpvecs = get_embeddings<float>(
            insertBatchSize + iter * appendBatchSize, insertBatchSize + (iter + 1) * appendBatchSize, dimension, -1);
        std::shared_ptr<MetadataSet> tmpmetas = TestUtils::TestDataGenerator<float>::GenerateMetadataSet(
            appendBatchSize, insertBatchSize + (iter)*appendBatchSize);
        auto t1 = std::chrono::high_resolution_clock::now();
        InsertVectors<float>(cloneIndexPtr, 1, appendBatchSize, tmpvecs, tmpmetas);
        std::cout << "Insert time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                           t1)
                         .count()
                  << " ms" << std::endl;

        ASSERT_EQ(cloneIndex->SaveIndex(clone_path), ErrorCode::Success);
        cloneIndex = nullptr;
    }

    for (int iter = 0; iter < insertIterations; iter++)
    {
        std::filesystem::remove_all("clone_index_" + std::to_string(iter));
    }
    // std::filesystem::remove_all("original_index");
}

std::shared_ptr<float[]> get_embeddings(uint32_t row_id, uint32_t embedding_dim, uint32_t array_index)
{
    std::shared_ptr<float[]> retval(new float[embedding_dim], std::default_delete<float[]>());
    for (int idx = 0; idx < embedding_dim; ++idx)
    {
        retval[idx] = row_id * 17 + idx * 19 + (array_index + 1) * 23;
    }

    NormalizeVector(retval.get(), embedding_dim);
    return retval;
}

std::shared_ptr<VectorSet> get_embeddings_float32(int start_id, int end_id, int embedding_dim)
{
    std::vector<float> array_embeddings;
    for (auto i = start_id; i < end_id; ++i)
    {
        uint32_t array_size = i % 4;
        if (array_size == 0)
        {
            continue; // some null embeddings
        }

        if (array_size == 3)
            array_size = 5;
        for (int j = 0; j < array_size; ++j)
        {
            // TODO add empty sub-array
            std::shared_ptr<float[]> embeddings = get_embeddings(i, 1024, j);
            for (int j = 0; j < 1024; ++j)
            {
                array_embeddings.emplace_back(embeddings[j]);
            }
        }
    }
    ByteArray vec = ByteArray::Alloc(sizeof(float) * array_embeddings.size());
    std::memcpy(vec.Data(), array_embeddings.data(), sizeof(float) * array_embeddings.size());
    return std::make_shared<BasicVectorSet>(vec, GetEnumValueType<float>(), embedding_dim, 2 * (end_id - start_id));
}

TEST(SPFreshTest, RefineTestIdx)
{
    using namespace SPFreshTest;

    constexpr int dimension = 1024;

    std::shared_ptr<VectorSet> vecset = get_embeddings_float32(0, 500, dimension);
    std::shared_ptr<MetadataSet> metaset = TestUtils::TestDataGenerator<float>::GenerateMetadataSet(1000, 0);

    for (auto i = 0; i < 2; ++i)
    {
        void *p = vecset->GetVector(i);
        for (auto i = 0; i < dimension; ++i)
        {
            std::cout << ((float *)p)[i] << " ";
        }
        std::cout << std::endl;
    }

    auto originalIndex = BuildIndex<float>("original_index", vecset, metaset, "COSINE");
    ASSERT_NE(originalIndex, nullptr);
    ASSERT_EQ(originalIndex->SaveIndex("original_index"), ErrorCode::Success);
    originalIndex = nullptr;

    std::string prevPath = "original_index";
    for (int iter = 0; iter < 1; iter++)
    {
        std::string clone_path = "clone_index_" + std::to_string(iter);
        std::shared_ptr<VectorIndex> prevIndex;
        ASSERT_EQ(VectorIndex::LoadIndex(prevPath, prevIndex), ErrorCode::Success);
        ASSERT_NE(prevIndex, nullptr);
        auto t0 = std::chrono::high_resolution_clock::now();
        ASSERT_EQ(prevIndex->Check(), ErrorCode::Success);
        std::cout << "Check time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                           t0)
                         .count()
                  << " ms" << std::endl;

        auto cloneIndex = prevIndex->Clone(clone_path);
        auto *cloneIndexPtr = static_cast<SPANN::Index<float> *>(cloneIndex.get());
        std::shared_ptr<VectorSet> tmpvecs = get_embeddings_float32(500, 1100, dimension);
        std::shared_ptr<MetadataSet> tmpmetas = TestUtils::TestDataGenerator<float>::GenerateMetadataSet(1200, 1000);
        auto t1 = std::chrono::high_resolution_clock::now();
        InsertVectors<float>(cloneIndexPtr, 1, 1200, tmpvecs, tmpmetas);
        std::cout << "Insert time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                           t1)
                         .count()
                  << " ms" << std::endl;

        for (auto i = 1000; i < 1900; ++i)
        {
            cloneIndexPtr->DeleteIndex(i);
        }

        ASSERT_EQ(cloneIndex->SaveIndex(clone_path), ErrorCode::Success);
        cloneIndex = nullptr;
    }

    for (int iter = 0; iter < 1; iter++)
    {
        std::filesystem::remove_all("clone_index_" + std::to_string(iter));
    }
    // std::filesystem::remove_all("original_index");
}
} // namespace SPFreshTestSuite