// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/SPANNResultIterator.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common/IQuantizer.h"
#include "inc/Core/Common/PQQuantizer.h"
#include "inc/Helper/DiskIO.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Quantizer/Training.h"
#include "inc/Test.h"
#include "inc/TestDataGenerator.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <ctime>
#include <tuple>
#include <vector>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

static void segfault_handler(int sig) {
    void *array[64];
    int size = backtrace(array, 64);
    fprintf(stderr, "\n===== SEGFAULT (signal %d) =====\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    fprintf(stderr, "===== END BACKTRACE =====\n");
    fflush(stderr);
    _exit(1);
}

static __attribute__((constructor)) void install_segfault_handler() {
    struct sigaction sa;
    sa.sa_handler = segfault_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
}

using namespace SPTAG;

namespace SPFreshTest
{
SizeType N = 10000;
DimensionType M = 100;
int K = 10;
int queries = 10;

std::shared_ptr<VectorSet> ConvertToFloatVectorSet(const std::shared_ptr<VectorSet>& src)
{
    if (!src)
        return nullptr;

    if (src->GetValueType() == VectorValueType::Float)
        return src;

    SizeType count = src->Count();
    DimensionType dim = src->Dimension();
    ByteArray bytes = ByteArray::Alloc(sizeof(float) * (size_t)count * (size_t)dim);
    float* out = reinterpret_cast<float*>(bytes.Data());

    switch (src->GetValueType())
    {
    case VectorValueType::Int8:
    {
        auto* in = reinterpret_cast<const std::int8_t*>(src->GetData());
        for (size_t i = 0; i < (size_t)count * (size_t)dim; ++i)
            out[i] = static_cast<float>(in[i]);
        break;
    }
    case VectorValueType::UInt8:
    {
        auto* in = reinterpret_cast<const std::uint8_t*>(src->GetData());
        for (size_t i = 0; i < (size_t)count * (size_t)dim; ++i)
            out[i] = static_cast<float>(in[i]);
        break;
    }
    case VectorValueType::Int16:
    {
        auto* in = reinterpret_cast<const std::int16_t*>(src->GetData());
        for (size_t i = 0; i < (size_t)count * (size_t)dim; ++i)
            out[i] = static_cast<float>(in[i]);
        break;
    }
    default:
        return nullptr;
    }

    return std::make_shared<BasicVectorSet>(bytes, VectorValueType::Float, dim, count);
}

std::shared_ptr<COMMON::IQuantizer> EnsurePQQuantizer(const std::string& quantizerFile,
                                                     const std::shared_ptr<VectorSet>& trainVectors,
                                                     DimensionType quantizedDim,
                                                     int threadNum)
{
    if (!trainVectors)
        return nullptr;

    std::shared_ptr<COMMON::IQuantizer> quantizer;
    if (fileexists(quantizerFile.c_str())) {
        auto ptr = SPTAG::f_createIO();
        if (ptr->Initialize(quantizerFile.c_str(), std::ios::binary | std::ios::in))
        {
            quantizer = COMMON::IQuantizer::LoadIQuantizer(ptr);
            BOOST_REQUIRE(quantizer != nullptr);
            return quantizer;
        }
    }

    if (quantizedDim <= 0 || (trainVectors->Dimension() % quantizedDim) != 0)
        return nullptr;

    auto options = std::make_shared<QuantizerOptions>(
        trainVectors->Count(), false, 0.0f, QuantizerType::PQQuantizer, quantizerFile, quantizedDim, "", "");
    options->m_dimension = trainVectors->Dimension();
    options->m_threadNum = threadNum;
    options->m_inputValueType = VectorValueType::Float;
    options->m_trainingSamples = trainVectors->Count();

    ByteArray pq_vector_array = ByteArray::Alloc(sizeof(std::uint8_t) * (size_t)quantizedDim * (size_t)trainVectors->Count());
    auto pq_vectors = std::make_shared<BasicVectorSet>(pq_vector_array, VectorValueType::UInt8, quantizedDim, trainVectors->Count());

    auto codebooks = TrainPQQuantizer<float>(options, trainVectors, pq_vectors);
    if (!codebooks)
        return nullptr;

    quantizer = std::make_shared<COMMON::PQQuantizer<float>>(
        quantizedDim, 256, trainVectors->Dimension() / quantizedDim, false, std::move(codebooks));

    auto fp = SPTAG::f_createIO();
    if (fp != nullptr && fp->Initialize(quantizerFile.c_str(), std::ios::binary | std::ios::out))
        quantizer->SaveQuantizer(fp);

    return quantizer;
}

template <typename T>
std::shared_ptr<VectorIndex> BuildIndex(const std::string &outDirectory, std::shared_ptr<VectorSet> vecset,
                                        std::shared_ptr<MetadataSet> metaset, const std::string &distMethod = "L2", int searchthread = 2)
{
    auto vecIndex = VectorIndex::CreateInstance(IndexAlgoType::SPANN, GetEnumValueType<T>());
    int maxthreads = std::thread::hardware_concurrency();
    int postingLimit = 4 * sizeof(T);
    std::string configuration = R"(
        [Base]
            DistCalcMethod=)" + distMethod + R"(
            IndexAlgoType=BKT
            ValueType=)" + Helper::Convert::ConvertToString(GetEnumValueType<T>()) + 
                                R"(
            Dim=)" + std::to_string(M) +
                                R"(
            IndexDirectory=)" + outDirectory +
                                R"(

        [SelectHead]
            isExecute=true
            NumberOfThreads=)" + std::to_string(maxthreads) + R"(
            SelectHeadType=BKT
            SelectThreshold=0
            SplitFactor=0
            SplitThreshold=0
            Ratio=0.2

        [BuildHead]
            isExecute=true
            NumberOfThreads=)" + std::to_string(maxthreads) + R"(

        [BuildSSDIndex]
            isExecute=true
            BuildSsdIndex=true
            InternalResultNum=64
            SearchInternalResultNum=64
            NumberOfThreads=)" + std::to_string(maxthreads) + R"(
	        PostingPageLimit=)" + std::to_string(postingLimit) + R"(
            SearchPostingPageLimit=)" + std::to_string(postingLimit) + R"(
            TmpDir=tmpdir
            Storage=FILEIO
            SpdkBatchSize=64
            ExcludeHead=false
            ResultNum=10
            SearchThreadNum=)" + std::to_string(searchthread) + R"(
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
            BufferLength=)" + std::to_string(postingLimit) + R"(
            InPlace=true
            StartFileSizeGB=1
            OneClusterCutMax=true
            ConsistencyCheck=true
            ChecksumCheck=true
            ChecksumInRead=false
            AsyncMergeInSearch=false
            DeletePercentageForRefine=0.4
            AsyncAppendQueueSize=0
            AllowZeroReplica=false
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
std::shared_ptr<VectorIndex> BuildLargeIndex(const std::string &outDirectory, std::string &pvecset,
                                        std::string& pmetaset, std::string& pmetaidx, Helper::IniReader& iniReader, const std::string &distMethod = "L2",
                                        int searchthread = 2, int insertthread = 2, int layers = 1,
                                        std::shared_ptr<COMMON::IQuantizer> quantizer = nullptr, std::string quantizerFilePath = "quantizer.bin")
{
    auto vecIndex = VectorIndex::CreateInstance(IndexAlgoType::SPANN, GetEnumValueType<T>());
    int maxthreads = std::thread::hardware_concurrency();
    int postingLimit = 4 * sizeof(T);
    std::string configuration = R"(
        [Base]
            DistCalcMethod=)" + distMethod + R"(
            IndexAlgoType=BKT
            VectorPath=)" + pvecset + R"(
            ValueType=)" + Helper::Convert::ConvertToString(GetEnumValueType<T>()) +
                                R"(
            Dim=)" + std::to_string(M) +
                                R"(
            IndexDirectory=)" + outDirectory +
                                R"(

        [SelectHead]
            isExecute=true
            NumberOfThreads=)" + std::to_string(maxthreads) + R"(
            SelectHeadType=BKT
            SelectThreshold=0
            SplitFactor=0
            SplitThreshold=0
            Ratio=0.2
            ParallelBKTBuild=true

        [BuildHead]
            isExecute=true
            AddCountForRebuild=10000
            NumberOfThreads=)" + std::to_string(maxthreads) + R"(

        [BuildSSDIndex]
            isExecute=true
            BuildSsdIndex=true
            InternalResultNum=64
            SearchInternalResultNum=64
            NumberOfThreads=)" + std::to_string(maxthreads) + R"(
	        PostingPageLimit=)" + std::to_string(postingLimit) +
                                R"(
            SearchPostingPageLimit=)" +
                                std::to_string(postingLimit) + R"(
            TmpDir=tmpdir
            Storage=FILEIO
            SpdkBatchSize=64
            ExcludeHead=false
            ResultNum=10
            SearchThreadNum=)" + std::to_string(searchthread) + R"(
            Update=true
            SteadyState=true
            InsertThreadNum=1
            AppendThreadNum=)" + std::to_string(insertthread) + R"(
            ReassignThreadNum=0
            DisableReassign=false
            ReassignK=64
            LatencyLimit=50.0
            SearchDuringUpdate=true
            MergeThreshold=10
            Sampling=4
            BufferLength=)" + std::to_string(postingLimit) +  R"(
            InPlace=true
            StartFileSizeGB=1
            OneClusterCutMax=true
            ConsistencyCheck=false
            ChecksumCheck=false
            ChecksumInRead=false
            AsyncMergeInSearch=false
            DeletePercentageForRefine=0.4
            AsyncAppendQueueSize=0
            AllowZeroReplica=false
            ShareDB=true            
            Layers=)" + std::to_string(layers) + R"(
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

    for (const auto &sec : sections)
    {
        auto params = iniReader.GetParameters(sec.c_str());
        for (const auto &[key, val] : params)
        {
            vecIndex->SetParameter(key.c_str(), val.c_str(), sec.c_str());
        }
    }

    if (quantizer)
    {
        vecIndex->SetParameter("QuantizerFilePath", quantizerFilePath.c_str(), "Base");
        vecIndex->SetQuantizer(quantizer);
        vecIndex->SetQuantizerADC(false);
        vecIndex->SetParameter("Dim", std::to_string(quantizer->GetNumSubvectors()).c_str(), "Base");
    }
    auto buildStatus = vecIndex->BuildIndex();
    if (buildStatus != ErrorCode::Success)
        return nullptr;

    vecIndex->SetMetadata(new SPTAG::FileMetadataSet(pmetaset, pmetaidx));
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
float Search(std::shared_ptr<VectorIndex> &vecIndex, std::shared_ptr<VectorSet> &queryset,
             std::shared_ptr<VectorSet> &baseVec, std::shared_ptr<VectorSet> &addVec, int k,
             std::shared_ptr<VectorSet> &truth, SizeType baseCount, int batch, int totalbatches)
{
    auto results = SearchOnly<T>(vecIndex, queryset, k);
    return TestUtils::TestDataGenerator<T>::EvaluateRecall(results, truth, k, k, batch, totalbatches);
}

template <typename ValueType>
void InsertVectors(SPANN::Index<ValueType> *p_index, int insertThreads, int step,
                   std::shared_ptr<VectorSet> addset, std::shared_ptr<MetadataSet> &metaset, int searchThreads = 0, std::shared_ptr<VectorSet> queryset = nullptr, int numQueries = 0, int k = 5, std::ostream* benchmarkData = nullptr, int start = 0)
{
    p_index->ForceCompaction();
    p_index->GetDBStat();

    std::vector<std::thread> threads;

    int printstep = step / 50;
    std::atomic_size_t vectorsSent(start);
    auto func = [&]() {
        size_t index = start;
        while (true)
        {
            index = vectorsSent.fetch_add(1);
            if (index < start + step)
            {
                if ((index % (printstep - 1)) == 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Sent %.2lf%%...\n", index * 100.0 / step);
                    p_index->GetDBStat();
                }
                ByteArray p_meta = metaset->GetMetadata((SizeType)index);
                std::uint64_t *offsets = new std::uint64_t[2]{0, p_meta.Length()};
                std::shared_ptr<MetadataSet> meta(new MemMetadataSet(
                    p_meta, ByteArray((std::uint8_t *)offsets, 2 * sizeof(std::uint64_t), true), 1));
                // For quantized index, pass GetFeatureDim() which returns reconstruct dimension
                ErrorCode ret = p_index->AddIndex(addset->GetVector((SizeType)index), 1, addset->Dimension(), meta, true);
                if (ret != ErrorCode::Success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "AddIndex failed. VID:%zu Dim:%d IndexDim:%d Storage:%s Error:%d\n",
                                 index,
                                 addset->Dimension(),
                                 p_index->GetFeatureDim(),
                                 p_index->GetParameter("Storage", "BuildSSDIndex").c_str(),
                                 static_cast<int>(ret));
                }
                BOOST_REQUIRE(ret == ErrorCode::Success);
            }
            else
            {
                return;
            }
        }
    };

    if (searchThreads > 0 && queryset != nullptr && numQueries != 0 && benchmarkData != nullptr) {
        std::vector<float> latencies(numQueries);
        std::vector<QueryResult> results(numQueries);
        std::vector<float> duration(searchThreads);

        for (int i = 0; i < numQueries; i++)
        {
            results[i] = QueryResult((const ValueType *)queryset->GetVector(i), k, false);
        }

        std::atomic_size_t queriesSent(0);
        auto search = [&](int tid) {
            auto s1 = std::chrono::high_resolution_clock::now();
            size_t qid;
            while ((qid = queriesSent.fetch_add(1)) < numQueries)
            {
                auto t1 = std::chrono::high_resolution_clock::now();
                p_index->SearchIndex(results[qid]);
                auto t2 = std::chrono::high_resolution_clock::now();
                latencies[qid] = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
            }
            auto s2 = std::chrono::high_resolution_clock::now();
            duration[tid] = std::chrono::duration_cast<std::chrono::microseconds>(s2 - s1).count() / 1000.0f;
        };

        for (int j = 0; j < insertThreads; j++)
        {
            threads.emplace_back(func);
        }
        for (int j = 0; j < searchThreads; j++)
        {
            threads.emplace_back(search, j);
        }
        for (auto &thread : threads)
        {
            thread.join();
        }

        // Calculate statistics
        float mean = 0, minLat = (std::numeric_limits<float>::max)(), maxLat = 0;
        for (int i = 0; i < numQueries; i++)
        {
            mean += latencies[i];
            minLat = (std::min)(minLat, latencies[i]);
            maxLat = (std::max)(maxLat, latencies[i]);
        }
        mean /= numQueries;

        std::sort(latencies.begin(), latencies.end());
        float p50 = latencies[static_cast<size_t>(numQueries * 0.50)];
        float p90 = latencies[static_cast<size_t>(numQueries * 0.90)];
        float p95 = latencies[static_cast<size_t>(numQueries * 0.95)];
        float p99 = latencies[static_cast<size_t>(numQueries * 0.99)];
        float maxBatchLatency = 1e-6;
        for (int i = 0; i < searchThreads; i++)
            if (maxBatchLatency < duration[i]) maxBatchLatency = duration[i];
        float qps = numQueries / maxBatchLatency;

        *benchmarkData << "        \"numQueries\": " << numQueries << ",\n";
        *benchmarkData << "        \"meanLatency\": " << mean << ",\n";
        *benchmarkData << "        \"p50\": " << p50 << ",\n";
        *benchmarkData << "        \"p90\": " << p90 << ",\n";
        *benchmarkData << "        \"p95\": " << p95 << ",\n";
        *benchmarkData << "        \"p99\": " << p99 << ",\n";
        *benchmarkData << "        \"minLatency\": " << minLat << ",\n";
        *benchmarkData << "        \"maxLatency\": " << maxLat << ",\n";
        *benchmarkData << "        \"qps\": " << qps << ",\n";
    }
    while (!p_index->AllFinished())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}


template <typename T>
void BenchmarkQueryPerformance(std::shared_ptr<VectorIndex> &index, std::shared_ptr<VectorSet> &queryset,
                               std::shared_ptr<VectorSet> &truth, const std::string &truthPath,
                               SizeType baseVectorCount, int topK, int searchK, int numThreads, int numQueries, int batches, int totalbatches,
                               std::ostream &benchmarkData, std::string prefix = "")
{
    // Benchmark: Query performance with detailed latency stats
    std::vector<float> latencies(numQueries);
    std::atomic_size_t queriesSent(0);
    std::vector<QueryResult> results(numQueries);

    for (int i = 0; i < numQueries; i++)
    {
        results[i] = QueryResult((const T *)queryset->GetVector(i), searchK, false);
    }

    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    auto batchStart = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numThreads; i++)
    {
        threads.emplace_back([&]() {
            size_t qid;
            while ((qid = queriesSent.fetch_add(1)) < numQueries)
            {
                auto t1 = std::chrono::high_resolution_clock::now();
                index->SearchIndex(results[qid]);
                auto t2 = std::chrono::high_resolution_clock::now();
                latencies[qid] = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
            }
        });
    }

    for (auto &thread : threads)
        thread.join();

    auto batchEnd = std::chrono::high_resolution_clock::now();
    float batchLatency =
        std::chrono::duration_cast<std::chrono::microseconds>(batchEnd - batchStart).count() / 1000000.0f;

    // Calculate statistics
    float mean = 0, minLat = (std::numeric_limits<float>::max)(), maxLat = 0;
    for (int i = 0; i < numQueries; i++)
    {
        mean += latencies[i];
        minLat = (std::min)(minLat, latencies[i]);
        maxLat = (std::max)(maxLat, latencies[i]);
    }
    mean /= numQueries;

    std::sort(latencies.begin(), latencies.end());
    float p50 = latencies[static_cast<size_t>(numQueries * 0.50)];
    float p90 = latencies[static_cast<size_t>(numQueries * 0.90)];
    float p95 = latencies[static_cast<size_t>(numQueries * 0.95)];
    float p99 = latencies[static_cast<size_t>(numQueries * 0.99)];
    float qps = numQueries / batchLatency;

    BOOST_TEST_MESSAGE("  Queries: " << numQueries);
    BOOST_TEST_MESSAGE("  Mean Latency: " << mean << " ms");
    BOOST_TEST_MESSAGE("  P50 Latency:  " << p50 << " ms");
    BOOST_TEST_MESSAGE("  P90 Latency:  " << p90 << " ms");
    BOOST_TEST_MESSAGE("  P95 Latency:  " << p95 << " ms");
    BOOST_TEST_MESSAGE("  P99 Latency:  " << p99 << " ms");
    BOOST_TEST_MESSAGE("  Min Latency:  " << minLat << " ms");
    BOOST_TEST_MESSAGE("  Max Latency:  " << maxLat << " ms");
    BOOST_TEST_MESSAGE("  QPS:          " << qps);

    // Collect JSON data for Benchmark
    benchmarkData << std::fixed << std::setprecision(4);
    benchmarkData << prefix << "{\n";
    benchmarkData << prefix << "      \"numQueries\": " << numQueries << ",\n";
    benchmarkData << prefix << "      \"meanLatency\": " << mean << ",\n";
    benchmarkData << prefix << "      \"p50\": " << p50 << ",\n";
    benchmarkData << prefix << "      \"p90\": " << p90 << ",\n";
    benchmarkData << prefix << "      \"p95\": " << p95 << ",\n";
    benchmarkData << prefix << "      \"p99\": " << p99 << ",\n";
    benchmarkData << prefix << "      \"minLatency\": " << minLat << ",\n";
    benchmarkData << prefix << "      \"maxLatency\": " << maxLat << ",\n";
    benchmarkData << prefix << "      \"qps\": " << qps << ",\n";
    

    // Recall evaluation (if truth file provided)
    if (!truth || truthPath.empty() || truthPath == "none")
    {
        BOOST_TEST_MESSAGE("  Recall evaluation skipped (no truth data)");
        benchmarkData << prefix << "      \"recall\": null\n";
        benchmarkData << prefix << "    }";
        return;
    }

    BOOST_TEST_MESSAGE("Checking for truth file: " << truthPath);
    std::shared_ptr<VectorSet> pvecset, paddvecset;
    float avgRecall = TestUtils::TestDataGenerator<T>::EvaluateRecall(results, truth, topK, searchK, batches, totalbatches);
    BOOST_TEST_MESSAGE("  Recall" << topK << "@" << searchK << " = " << (avgRecall * 100.0f) << "%");
    BOOST_TEST_MESSAGE("  (Evaluated on " << numQueries << " queries against base vectors)");
    benchmarkData << std::fixed << std::setprecision(4);
    benchmarkData << prefix << "      \"recall\": {\n";
    benchmarkData << prefix << "        \"recallAtK\": " << avgRecall << ",\n";
    benchmarkData << prefix << "        \"k\": " << topK << ",\n";
    benchmarkData << prefix << "        \"numQueries\": " << numQueries << "\n";
    benchmarkData << prefix << "      }\n";
    benchmarkData << prefix << "    }";
}

ErrorCode QuantizeVectors(const std::shared_ptr<COMMON::IQuantizer>& quantizer,
                     const std::shared_ptr<VectorSet>& srcVectors,
                     ByteArray& quantizedBytes)
{
    if (!quantizer || !srcVectors)
        return ErrorCode::Fail;

    int maxthreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.reserve(maxthreads);
    std::atomic_size_t vectorsSent(0);
    auto func = [&]() {
        size_t index = 0;
        while (true)
        {
            index = vectorsSent.fetch_add(1);
            if (index < srcVectors->Count())
            {
                quantizer->QuantizeVector(srcVectors->GetVector(index), quantizedBytes.Data() + index * (size_t)(quantizer->GetNumSubvectors()), false);
            }
            else
            {
                return;
            }
        }
    };
    for (int j = 0; j < maxthreads; j++)
    {
        threads.emplace_back(func);
    }
    for (auto &thread : threads)
    {
        thread.join();
    }
    return ErrorCode::Success;
}

template <typename T>
void RunBenchmark(const std::string &vectorPath, const std::string &queryPath, const std::string &truthPath,
                  DistCalcMethod distMethod, const std::string &indexPath, int dimension, int baseVectorCount,
                  int insertVectorCount, int deleteVectorCount, int batches, int topK, int numSearchThreads, int numInsertThreads, int numSearchDuringInsertThreads, int numQueries, Helper::IniReader& iniReader,
                  const std::string &outputFile = "output.json", const bool rebuild = true, const int resume = -1,
                  const std::string &quantizerFilePath = std::string(""), int quantizedDim = 0, int layers = 1)
{
    int oldM = M, oldK = K, oldN = N, oldQueries = queries;
    N = baseVectorCount;
    queries = numQueries;
    M = dimension;
    K = topK;
    std::string dist = Helper::Convert::ConvertToString(distMethod);
    int insertBatchSize = insertVectorCount / max(batches, 1);
    int deleteBatchSize = deleteVectorCount / max(batches, 1);

    // Variables to collect JSON output data
    std::ostringstream tmpbenchmark;

    // Generate test data
    bool generateTruth = !(truthPath.empty() || truthPath == "none");
    bool enableQuantization = !quantizerFilePath.empty();
    std::string pvecset, paddset, pqueryset, ptruth, pmeta, pmetaidx, paddmeta, paddmetaidx;
    TestUtils::TestDataGenerator<T> generator(N, queries, M, K, dist, insertVectorCount, false, vectorPath, queryPath);
    generator.RunLargeBatches(pvecset, pmeta, pmetaidx, paddset, paddmeta, paddmetaidx, pqueryset, N, insertBatchSize,
                              deleteBatchSize, batches, ptruth, generateTruth);

    std::ofstream jsonFile(outputFile);
    BOOST_REQUIRE(jsonFile.is_open());

    jsonFile << std::fixed << std::setprecision(4);

    // Get current timestamp
    auto time_t_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm_now;
#if defined(_MSC_VER)
    localtime_s(&tm_now, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_now);
#endif

    std::ostringstream timestampStream;
    timestampStream << std::put_time(&tm_now, "%Y-%m-%dT%H:%M:%S");
    std::string timestamp = timestampStream.str();

    jsonFile << "{\n";
    jsonFile << "  \"timestamp\": \"" << timestamp << "\",\n";
    jsonFile << "  \"config\": {\n";
    jsonFile << "    \"vectorPath\": \"" << vectorPath << "\",\n";
    jsonFile << "    \"queryPath\": \"" << queryPath << "\",\n";
    jsonFile << "    \"truthPath\": \"" << truthPath << "\",\n";
    jsonFile << "    \"indexPath\": \"" << indexPath << "\",\n";
    jsonFile << "    \"quantizerPath\": \"" << quantizerFilePath << "\",\n";
    jsonFile << "    \"ValueType\": \"" << Helper::Convert::ConvertToString(GetEnumValueType<T>()) << "\",\n";
    jsonFile << "    \"dimension\": " << dimension << ",\n";
    jsonFile << "    \"baseVectorCount\": " << baseVectorCount << ",\n";
    jsonFile << "    \"insertVectorCount\": " << insertVectorCount << ",\n";
    jsonFile << "    \"DeleteVectorCount\": " << deleteVectorCount << ",\n";
    jsonFile << "    \"BatchNum\": " << batches << ",\n";
    jsonFile << "    \"topK\": " << topK << ",\n";
    jsonFile << "    \"numQueries\": " << numQueries << ",\n";
    jsonFile << "    \"numSearchThreads\": " << numSearchThreads << ",\n";
    jsonFile << "    \"numInsertThreads\": " << numInsertThreads << ",\n";
    jsonFile << "    \"layers\": " << layers << ",\n";
    jsonFile << "    \"DistMethod\": \"" << Helper::Convert::ConvertToString(distMethod) << "\"\n";
    jsonFile << "  },\n";
    jsonFile << "  \"results\": {\n";

    int SearchK = enableQuantization? topK * 4 : topK;
    std::shared_ptr<VectorIndex> index;
    std::shared_ptr<COMMON::IQuantizer> quantizer;
    
    // Build initial index
    BOOST_TEST_MESSAGE("\n=== Building Index ===");
    if (rebuild || !direxists(indexPath.c_str())) {
        auto buildstart = std::chrono::high_resolution_clock::now();

        if (enableQuantization)
        {
            auto baseVectorsRaw = TestUtils::TestDataGenerator<T>::LoadVectorSet(pvecset, M);
            BOOST_REQUIRE(baseVectorsRaw != nullptr);

            auto baseVectorsFloat = ConvertToFloatVectorSet(baseVectorsRaw);
            BOOST_REQUIRE(baseVectorsFloat != nullptr);

            if (quantizedDim <= 0) quantizedDim = dimension / 2;
            BOOST_REQUIRE(quantizedDim > 0 && (dimension % quantizedDim) == 0);

            quantizer = EnsurePQQuantizer(quantizerFilePath, baseVectorsFloat, (DimensionType)quantizedDim, numSearchThreads);
            BOOST_REQUIRE(quantizer != nullptr);

            std::string pquanvecset = "perftest_quanvectors.bin";
            {
                ByteArray quantizedBaseBytes = ByteArray::Alloc((size_t)baseVectorCount * (size_t)quantizer->GetNumSubvectors());
                BOOST_REQUIRE(QuantizeVectors(quantizer, baseVectorsFloat, quantizedBaseBytes) == ErrorCode::Success);
                auto quantizedBase = std::make_shared<BasicVectorSet>(quantizedBaseBytes, VectorValueType::UInt8, quantizer->GetNumSubvectors(), baseVectorCount);
                quantizedBase->Save(pquanvecset);
            }

            index = BuildLargeIndex<uint8_t>(indexPath, pquanvecset, pmeta, pmetaidx, iniReader, dist, numSearchThreads, numInsertThreads, layers, quantizer, "quantizer.bin");
            BOOST_REQUIRE(index != nullptr);
            index->SetQuantizerADC(true);
        }
        else
        {
            index = BuildLargeIndex<T>(indexPath, pvecset, pmeta, pmetaidx, iniReader, dist, numSearchThreads, numInsertThreads, layers);
            BOOST_REQUIRE(index != nullptr);
        }

        auto buildend = std::chrono::high_resolution_clock::now();
        double buildseconds =
            std::chrono::duration_cast<std::chrono::microseconds>(buildend - buildstart).count() / 1000000.0f;
        jsonFile << "    \"build timeSeconds\": " << buildseconds << ",\n";
        BOOST_TEST_MESSAGE("Index built successfully with " << baseVectorCount << " vectors");
    }
    else
    {
        BOOST_REQUIRE(VectorIndex::LoadIndex(indexPath, index) == ErrorCode::Success);
        BOOST_REQUIRE(index != nullptr);
    }

    auto queryset = TestUtils::TestDataGenerator<T>::LoadVectorSet(pqueryset, M);
    BOOST_REQUIRE(queryset != nullptr);

    if (enableQuantization)
    {
        if (!quantizer)
        {
            quantizer = index->GetQuantizer();
        }
        BOOST_REQUIRE(quantizer != nullptr);
        queryset = ConvertToFloatVectorSet(queryset);
    }

    std::shared_ptr<VectorSet> truth;
    if (generateTruth)
    {
        truth = TestUtils::TestDataGenerator<float>::LoadVectorSet(ptruth, K);
    }

    // Benchmark 0: Query performance before insertions (round 1 — cold cache)
    BOOST_TEST_MESSAGE("\n=== Benchmark 0: Query Before Insertions (Round 1) ===");
    BenchmarkQueryPerformance<T>(index, queryset, truth, truthPath, baseVectorCount, topK, SearchK,
                                 numSearchThreads, numQueries, 0, batches, tmpbenchmark);
    jsonFile << "    \"benchmark0_query_before_insert\": ";
    BenchmarkQueryPerformance<T>(index, queryset, truth, truthPath, baseVectorCount, topK, SearchK,
                                 numSearchThreads, numQueries, 0, batches, jsonFile);
    jsonFile << ",\n";
    jsonFile.flush();

    // Benchmark 0b: Query performance before insertions (round 2 — warm cache)
    BOOST_TEST_MESSAGE("\n=== Benchmark 0b: Query Before Insertions (Round 2) ===");
    BenchmarkQueryPerformance<T>(index, queryset, truth, truthPath, baseVectorCount, topK, SearchK,
                                 numSearchThreads, numQueries, 0, batches, tmpbenchmark);
    jsonFile << "    \"benchmark0b_query_before_insert_round2\": ";
    BenchmarkQueryPerformance<T>(index, queryset, truth, truthPath, baseVectorCount, topK, SearchK,
                                 numSearchThreads, numQueries, 0, batches, jsonFile);
    jsonFile << ",\n";
    jsonFile.flush();

    BOOST_REQUIRE(index->SaveIndex(indexPath) == ErrorCode::Success);
    index = nullptr;


    // Benchmark 1: Insert performance
    if (insertBatchSize > 0)
    {
        BOOST_TEST_MESSAGE("\n=== Benchmark 1: Insert Performance ===");
        {
            jsonFile << "    \"benchmark1_insert\": {\n";

            // Auto-resume: if resume == -1, check for checkpoint file
            int effectiveResume = resume;
            std::string checkpointFile = indexPath + "/checkpoint.txt";
            if (effectiveResume < 0) {
                std::ifstream cpIn(checkpointFile);
                if (cpIn.is_open()) {
                    int lastBatch = -1;
                    if (cpIn >> lastBatch && lastBatch >= 0) {
                        // Verify the saved index directory exists
                        std::string savedPath = indexPath + "_" + std::to_string(lastBatch);
                        if (std::filesystem::exists(savedPath)) {
                            effectiveResume = lastBatch;
                            BOOST_TEST_MESSAGE("Auto-resuming from checkpoint: batch " << lastBatch << "/" << batches);
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Auto-resuming from checkpoint: batch %d/%d\n", lastBatch, batches);
                        }
                    }
                    cpIn.close();
                }
            }

            std::string prevPath = indexPath;
            if (effectiveResume >= 0)
            {
                prevPath = indexPath + "_" + std::to_string(effectiveResume);
            }
            for (int iter = effectiveResume + 1; iter < batches; iter++)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "========== BATCH %d/%d (insert %d vectors) ==========\n", iter + 1, batches, insertBatchSize);
                BOOST_TEST_MESSAGE("\n========== BATCH " << iter + 1 << "/" << batches << " ==========");
                jsonFile << "      \"batch_" << iter + 1 << "\": {\n";

                std::string clonePath = indexPath + "_" + std::to_string(iter);
                if (std::filesystem::exists(clonePath))
                {
                    std::filesystem::remove_all(clonePath);
                }
                std::shared_ptr<VectorIndex> prevIndex, cloneIndex;
                auto start = std::chrono::high_resolution_clock::now();
                BOOST_REQUIRE(VectorIndex::LoadIndex(prevPath, prevIndex) == ErrorCode::Success);
                auto end = std::chrono::high_resolution_clock::now();
                BOOST_REQUIRE(prevIndex != nullptr);
                BOOST_REQUIRE(prevIndex->Check() == ErrorCode::Success);

                double seconds =
                    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0f;
                int vectorCount = prevIndex->GetNumSamples();
                BOOST_TEST_MESSAGE("  Load Time: " << seconds << " seconds");
                BOOST_TEST_MESSAGE("  Index vectors after reload: " << vectorCount);

                // Collect JSON data for Benchmark 4
                jsonFile << "        \"Load timeSeconds\": " << seconds << ",\n";
                jsonFile << "        \"Load vectorCount\": " << vectorCount << ",\n";

                start = std::chrono::high_resolution_clock::now();
                cloneIndex = prevIndex->Clone(clonePath);
                end = std::chrono::high_resolution_clock::now();
                seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0f;
                jsonFile << "        \"Clone timeSeconds\": " << seconds << ",\n";
                
                prevIndex = nullptr;
                
                // If using quantization, update dimension after clone
                if (enableQuantization)
                {
                    cloneIndex->SetParameter("Dim", std::to_string(quantizer->GetNumSubvectors()).c_str(), "Base");
                }
                
                ErrorCode cloneret = cloneIndex->Check();
                BOOST_REQUIRE(cloneret == ErrorCode::Success);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Cloned index from %s to %s, check:%d, time: %f seconds\n",
                             prevPath.c_str(), clonePath.c_str(), (int)(cloneret == ErrorCode::Success), seconds);

                int insertStart = iter * insertBatchSize;
                {
                    std::shared_ptr<VectorSet> addset = TestUtils::TestDataGenerator<T>::LoadVectorSet(paddset, M, insertStart, insertBatchSize);
                    ByteArray quantizedAddBytes;
                    if (enableQuantization) {
                        auto addFloat = ConvertToFloatVectorSet(addset);
                        BOOST_REQUIRE(addFloat != nullptr);
                        quantizedAddBytes = ByteArray::Alloc((size_t)addFloat->Count() * (size_t)(quantizer->GetNumSubvectors()));
                        BOOST_REQUIRE(QuantizeVectors(quantizer, addFloat, quantizedAddBytes) == ErrorCode::Success);
                        addset = std::make_shared<BasicVectorSet>(quantizedAddBytes,
                                                                 VectorValueType::UInt8,
                                                                 quantizer->GetNumSubvectors(),
                                                                 addFloat->Count());
                    }
                    std::shared_ptr<MetadataSet> addmetaset = TestUtils::TestDataGenerator<T>::LoadMetadataSet(paddmeta, paddmetaidx, insertStart, insertBatchSize);
                    start = std::chrono::high_resolution_clock::now();
                    InsertVectors<T>(static_cast<SPANN::Index<T> *>(cloneIndex.get()), numInsertThreads, insertBatchSize,
                                     addset, addmetaset, numSearchDuringInsertThreads, queryset, numQueries, SearchK, &jsonFile, 0);
                    end = std::chrono::high_resolution_clock::now();
                }
                seconds =
                    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0f;
                double throughput = insertBatchSize / seconds;

                BOOST_TEST_MESSAGE("  Inserted: " << insertBatchSize << " vectors");
                BOOST_TEST_MESSAGE("  Time: " << seconds << " seconds");
                BOOST_TEST_MESSAGE("  Throughput: " << throughput << " vectors/sec");

                // Collect JSON data for Benchmark 1               
                jsonFile << "        \"inserted\": " << insertBatchSize << ",\n";
                jsonFile << "        \"insert timeSeconds\": " << seconds << ",\n";
                jsonFile << "        \"insert throughput\": " << throughput << ",\n";

                if (deleteBatchSize > 0)
                {
                    std::vector<std::thread> threads;
                    threads.reserve(numInsertThreads);

                    int startidx = iter * deleteBatchSize;
                    std::atomic_size_t vectorsSent(startidx);
                    int totaldeleted = startidx + deleteBatchSize;
                    int printstep = deleteBatchSize / 50;
                    auto func = [&]() {
                        size_t idx = startidx;
                        while (true)
                        {
                            idx = vectorsSent.fetch_add(1);
                            if (idx < totaldeleted)
                            {
                                if ((idx % (printstep - 1)) == 0)
                                {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Sent %.2lf%%...\n",
                                                 (idx - startidx) * 100.0 / deleteBatchSize);
                                }
                                auto ret = cloneIndex->DeleteIndex(idx);
                                if (ret != ErrorCode::Success) {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "DeleteIndex(%d) failed: %d\n", (int)idx, (int)ret);
                                }
                            }
                            else
                            {
                                return;
                            }
                        }
                    };

                    start = std::chrono::high_resolution_clock::now();
                    for (int j = 0; j < numInsertThreads; j++)
                    {
                        threads.emplace_back(func);
                    }
                    for (auto &thread : threads)
                    {
                        thread.join();
                    }
                    end = std::chrono::high_resolution_clock::now();
                    double seconds =
                        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0f;
                    double throughput = deleteBatchSize / seconds;

                    jsonFile << "        \"deleted\": " << deleteVectorCount << ",\n";
                    jsonFile << "        \"delete timeSeconds\": " << seconds << ",\n";
                    jsonFile << "        \"delete throughput\": " << throughput << ",\n";
                }

                BOOST_TEST_MESSAGE("\n=== Benchmark 2: Query After Insertions and Deletions ===");
                jsonFile << "        \"search\":";
                BenchmarkQueryPerformance<T>(cloneIndex, queryset, truth, truthPath, baseVectorCount, topK, SearchK, numSearchThreads,
                                             numQueries, iter + 1, batches, tmpbenchmark, "    ");
                BenchmarkQueryPerformance<T>(cloneIndex, queryset, truth, truthPath, baseVectorCount,
                                             topK, SearchK, numSearchThreads, numQueries, iter + 1, batches, jsonFile, "    ");
                jsonFile << ",\n";

                BOOST_TEST_MESSAGE("\n=== Benchmark 2b: Query After Insertions and Deletions (Round 2) ===");
                jsonFile << "        \"search_round2\":";
                BenchmarkQueryPerformance<T>(cloneIndex, queryset, truth, truthPath, baseVectorCount, topK, SearchK, numSearchThreads,
                                             numQueries, iter + 1, batches, tmpbenchmark, "    ");
                BenchmarkQueryPerformance<T>(cloneIndex, queryset, truth, truthPath, baseVectorCount,
                                             topK, SearchK, numSearchThreads, numQueries, iter + 1, batches, jsonFile, "    ");
                jsonFile << ",\n";

                start = std::chrono::high_resolution_clock::now();
                BOOST_REQUIRE(cloneIndex->SaveIndex(clonePath) == ErrorCode::Success);
                end = std::chrono::high_resolution_clock::now();

                seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0f;
                BOOST_TEST_MESSAGE("  Save Time: " << seconds << " seconds");
                BOOST_TEST_MESSAGE("  Save completed successfully");

                // Write checkpoint file after successful save
                {
                    std::ofstream cpOut(checkpointFile, std::ios::trunc);
                    if (cpOut.is_open()) {
                        cpOut << iter << std::endl;
                        cpOut.close();
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Checkpoint saved: batch %d/%d\n", iter + 1, batches);
                    }
                }

                // Collect JSON data for Benchmark 3
                jsonFile << "        \"save timeSeconds\": " << seconds << "\n";

                if (iter != batches - 1)
                    jsonFile << "      },\n";
                else
                    jsonFile << "      }\n";

                cloneIndex = nullptr;
                prevPath = clonePath;
                jsonFile.flush();

                if (iter > 0)
                    std::filesystem::remove_all(indexPath + "_" + std::to_string(iter - 1));
            }
        }
        jsonFile << "    }\n";
    }

    jsonFile << "  }\n";
    jsonFile << "}\n";
    jsonFile.close();

    M = oldM;
    K = oldK;
    N = oldN;
    queries = oldQueries;
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

BOOST_AUTO_TEST_SUITE(SPFreshTest)

BOOST_AUTO_TEST_CASE(TestLoadAndSave)
{
    using namespace SPFreshTest;

    // Prepare test data using TestDataGenerator
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.RunBatches(vecset, metaset, addvecset, addmetaset, queryset, N, N, 0, 1, truth);

    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    originalIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedIndex;
    BOOST_REQUIRE(VectorIndex::LoadIndex("original_index", loadedIndex) == ErrorCode::Success);
    BOOST_REQUIRE(loadedIndex != nullptr);
    BOOST_REQUIRE(loadedIndex->SaveIndex("loaded_and_saved_index") == ErrorCode::Success);
    loadedIndex = nullptr;

    std::unordered_set<std::string> exceptions = {"indexloader.ini"};

    // Compare files in both directories
    BOOST_REQUIRE_MESSAGE(CompareDirectoriesWithLogging("original_index", "loaded_and_saved_index", exceptions),
                          "Saved index does not match loaded-then-saved index");

    std::filesystem::remove_all("original_index");
    std::filesystem::remove_all("loaded_and_saved_index");
}

BOOST_AUTO_TEST_CASE(TestReopenIndexRecall)
{
    using namespace SPFreshTest;

    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.RunBatches(vecset, metaset, addvecset, addmetaset, queryset, N, N, 0, 1, truth);

    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    float recall1 = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N, 0, 1);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);    
    originalIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedOnce;
    BOOST_REQUIRE(VectorIndex::LoadIndex("original_index", loadedOnce) == ErrorCode::Success);
    BOOST_REQUIRE(loadedOnce != nullptr);
    BOOST_REQUIRE(loadedOnce->SaveIndex("reopened_index") == ErrorCode::Success);
    loadedOnce = nullptr;

    std::shared_ptr<VectorIndex> loadedTwice;
    BOOST_REQUIRE(VectorIndex::LoadIndex("reopened_index", loadedTwice) == ErrorCode::Success);
    BOOST_REQUIRE(loadedTwice != nullptr);
    float recall2 = Search<int8_t>(loadedTwice, queryset, vecset, addvecset, K, truth, N, 0, 1);
    loadedTwice = nullptr;

    BOOST_REQUIRE_MESSAGE(std::fabs(recall1 - recall2) < 0.02, "Recall mismatch between original and reopened index");

    std::filesystem::remove_all("original_index");
    std::filesystem::remove_all("reopened_index");
}

BOOST_AUTO_TEST_CASE(TestInsertAndSearch)
{
    using namespace SPFreshTest;

    // Prepare test data using TestDataGenerator
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.RunBatches(vecset, metaset, addvecset, addmetaset, queryset, N, N, 0, 1, truth);

    // Build base index
    auto index = BuildIndex<int8_t>("insert_test_index", vecset, metaset);
    BOOST_REQUIRE(index != nullptr);
    BOOST_REQUIRE(index->SaveIndex("insert_test_index") == ErrorCode::Success);
    index = nullptr;

    std::shared_ptr<VectorIndex> loadedOnce;
    BOOST_REQUIRE(VectorIndex::LoadIndex("insert_test_index", loadedOnce) == ErrorCode::Success);
    BOOST_REQUIRE(loadedOnce != nullptr);

    InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(loadedOnce.get()), 2, 1000, addvecset, addmetaset);
    SearchOnly<int8_t>(loadedOnce, queryset, K);
    loadedOnce = nullptr;

    std::filesystem::remove_all("insert_test_index");
}

BOOST_AUTO_TEST_CASE(TestClone)
{
    using namespace SPFreshTest;

    // Prepare test data using TestDataGenerator
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.RunBatches(vecset, metaset, addvecset, addmetaset, queryset, N, N, 0, 1, truth);

    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);

    auto clonedIndex = originalIndex->Clone("cloned_index");
    BOOST_REQUIRE(clonedIndex != nullptr);
    BOOST_REQUIRE(clonedIndex->SaveIndex("cloned_index") == ErrorCode::Success);
    originalIndex.reset();
    clonedIndex = nullptr;

    std::unordered_set<std::string> exceptions = {"indexloader.ini"};

    // Compare files in both directories
    BOOST_REQUIRE_MESSAGE(CompareDirectoriesWithLogging("original_index", "cloned_index", exceptions),
                          "Saved index does not match loaded-then-saved index");

    std::filesystem::remove_all("original_index");
    std::filesystem::remove_all("cloned_index");
}

BOOST_AUTO_TEST_CASE(TestCloneRecall)
{
    using namespace SPFreshTest;

    // Prepare test data using TestDataGenerator
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.RunBatches(vecset, metaset, addvecset, addmetaset, queryset, N, N, 0, 1, truth);

    auto originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    float originalRecall = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N, 0, 1);
    
    auto clonedIndex = originalIndex->Clone("cloned_index");
    BOOST_REQUIRE(clonedIndex != nullptr);
    originalIndex.reset();
    clonedIndex = nullptr;

    std::shared_ptr<VectorIndex> loadedClonedIndex;
    BOOST_REQUIRE(VectorIndex::LoadIndex("cloned_index", loadedClonedIndex) == ErrorCode::Success);
    BOOST_REQUIRE(loadedClonedIndex != nullptr);
    float clonedRecall = Search<int8_t>(loadedClonedIndex, queryset, vecset, addvecset, K, truth, N, 0, 1);
    loadedClonedIndex = nullptr;

    BOOST_REQUIRE_MESSAGE(std::fabs(originalRecall - clonedRecall) < 0.02,
                          "Recall mismatch between original and cloned index: "
                              << "original=" << originalRecall << ", cloned=" << clonedRecall);

    std::filesystem::remove_all("original_index");
    std::filesystem::remove_all("cloned_index");
}

BOOST_AUTO_TEST_CASE(IndexPersistenceAndInsertSanity)
{
    using namespace SPFreshTest;

    // Prepare test data
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.RunBatches(vecset, metaset, addvecset, addmetaset, queryset, N, N, 0, 1, truth);

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
    InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(loadedClone.get()), 1,
                          static_cast<int>(addvecset->Count()), addvecset, addmetaset);

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
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.RunBatches(vecset, metaset, addvecset, addmetaset, queryset, N, N, 0, 1, truth);

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
    InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(loadedClone.get()), 2,
                          static_cast<int>(addvecset->Count()), addvecset, addmetaset);

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
}

BOOST_AUTO_TEST_CASE(IndexSaveDuringQuery)
{
    using namespace SPFreshTest;

    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.RunBatches(vecset, metaset, addvecset, addmetaset, queryset, N, N, 0, 1, truth);

    auto index = BuildIndex<int8_t>("save_during_query_index", vecset, metaset);
    BOOST_REQUIRE(index != nullptr);

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
    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.RunBatches(vecset, metaset, addvecset, addmetaset, queryset, N, N, 0, 1, truth);

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
    InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(loaded.get()), 2,
                          static_cast<int>(addvecset->Count()), addvecset, addmetaset);

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

BOOST_AUTO_TEST_CASE(IndexShadowCloneLifecycleKeepLast)
{
    using namespace SPFreshTest;

    constexpr int iterations = 5;
    constexpr int insertBatchSize = 100;

    std::shared_ptr<VectorSet> vecset, queryset, truth, addvecset;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.RunBatches(vecset, metaset, addvecset, addmetaset, queryset, N, N, 0, 1, truth);

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
        BOOST_REQUIRE(loaded->Clone(shadowIndexName) != nullptr);
        loaded.reset();

        std::shared_ptr<VectorIndex> shadowLoaded;
        BOOST_REQUIRE(VectorIndex::LoadIndex(shadowIndexName, shadowLoaded) == ErrorCode::Success);
        BOOST_REQUIRE(shadowLoaded != nullptr);
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

        BOOST_REQUIRE(shadowLoaded->SaveIndex(shadowIndexName) == ErrorCode::Success);
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

BOOST_AUTO_TEST_CASE(IterativeSearch)
{
    using namespace SPFreshTest;

    constexpr int insertIterations = 5;
    constexpr int insertBatchSize = 1000;
    constexpr int dimension = 1024;
    std::shared_ptr<VectorSet> vecset = get_embeddings<float>(0, insertBatchSize, dimension, -1);
    std::shared_ptr<MetadataSet> metaset =
        TestUtils::TestDataGenerator<float>::GenerateMetadataSet(insertBatchSize, 0);

    auto originalIndex = BuildIndex<float>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    originalIndex = nullptr;

    std::string prevPath = "original_index";
    for (int iter = 0; iter < insertIterations; iter++)
    {
        std::string clone_path = "clone_index_" + std::to_string(iter);
        std::shared_ptr<VectorIndex> prevIndex;
        BOOST_REQUIRE(VectorIndex::LoadIndex(prevPath, prevIndex) == ErrorCode::Success);
        BOOST_REQUIRE(prevIndex != nullptr);

        auto cloneIndex = prevIndex->Clone(clone_path);
        auto *cloneIndexPtr = static_cast<SPANN::Index<float> *>(cloneIndex.get());
        std::shared_ptr<VectorSet> tmpvecs =
            get_embeddings<float>((iter + 1) * insertBatchSize, (iter + 2) * insertBatchSize, dimension, -1);
        std::shared_ptr<MetadataSet> tmpmetas =
            TestUtils::TestDataGenerator<float>::GenerateMetadataSet(insertBatchSize, (iter + 1) * insertBatchSize);
        InsertVectors<float>(cloneIndexPtr, 1, insertBatchSize, tmpvecs, tmpmetas);

        BOOST_REQUIRE(cloneIndex->SaveIndex(clone_path) == ErrorCode::Success);
        cloneIndex = nullptr;

        std::shared_ptr<VectorIndex> loadedIndex;
        BOOST_REQUIRE(VectorIndex::LoadIndex(clone_path, loadedIndex) == ErrorCode::Success);
        BOOST_REQUIRE(loadedIndex != nullptr);

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

BOOST_AUTO_TEST_CASE(RefineIndex)
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
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);

    float recall = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N, 0, iterations);
    std::cout << "original: recall@" << K << "= " << recall << std::endl;

    for (int iter = 0; iter < iterations; iter++)
    {

        InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(originalIndex.get()), 1, insertBatchSize, addvecset,
                              metaset, iter * insertBatchSize);
        for (int i = 0; i < deleteBatchSize; i++)
            originalIndex->DeleteIndex(iter * deleteBatchSize + i);

        recall = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N, iter + 1, iterations);
        std::cout << "iter " << iter << ": recall@" << K << "=" << recall << std::endl;
    }
    std::cout << "Before Refine:" << " recall@" << K << "=" << recall << std::endl;
    static_cast<SPANN::Index<int8_t> *>(originalIndex.get())->GetDBStat();
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    originalIndex = nullptr;

    BOOST_REQUIRE(VectorIndex::LoadIndex("original_index", originalIndex) == ErrorCode::Success);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->Check() == ErrorCode::Success);

    recall = Search<int8_t>(originalIndex, queryset, vecset, addvecset, K, truth, N, iterations, iterations);
    std::cout << "After Refine:" << " recall@" << K << "=" << recall << std::endl;
    static_cast<SPANN::Index<int8_t> *>(originalIndex.get())->GetDBStat();
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    originalIndex = nullptr;

    std::filesystem::remove_all("original_index");
}

BOOST_AUTO_TEST_CASE(CacheTest)
{
    using namespace SPFreshTest;

    int iterations = 5;
    int insertBatchSize = N / iterations;
    int deleteBatchSize = N / iterations;

    // Generate test data
    std::shared_ptr<VectorSet> vecset, addvecset, queryset, truth;
    std::shared_ptr<MetadataSet> metaset, addmetaset;

    std::srand(10);
    TestUtils::TestDataGenerator<int8_t> generator(N, queries, M, K, "L2");
    generator.RunBatches(vecset, metaset, addvecset, addmetaset, queryset, N, insertBatchSize, deleteBatchSize,
                         iterations, truth);

    // Build and save index
    std::shared_ptr<VectorIndex> originalIndex, finalIndex;
    
    std::filesystem::remove_all("original_index");

    originalIndex = BuildIndex<int8_t>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    originalIndex = nullptr;

   
    for (int iter = 0; iter < iterations; iter++)
    {
        if (direxists(("clone_index_" + std::to_string(iter)).c_str()))
        {
            std::filesystem::remove_all("clone_index_" + std::to_string(iter));
        }
    }
    
    std::string prevPath = "original_index";
    float recall = 0.0;
    
    std::cout << "=================No Cache===================" << std::endl;
    
    for (int iter = 0; iter < iterations; iter++)
    {
        std::string clone_path = "clone_index_" + std::to_string(iter);
        std::shared_ptr<VectorIndex> prevIndex;
        BOOST_REQUIRE(VectorIndex::LoadIndex(prevPath, prevIndex) == ErrorCode::Success);
        BOOST_REQUIRE(prevIndex != nullptr);
        auto t0 = std::chrono::high_resolution_clock::now();
        BOOST_REQUIRE(prevIndex->Check() == ErrorCode::Success);
        std::cout << "[INFO] Check time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t0).count()
                  << " ms" << std::endl;
        

        auto cloneIndex = prevIndex->Clone(clone_path);
        prevIndex = nullptr;
        BOOST_REQUIRE(cloneIndex->Check() == ErrorCode::Success);
        
        recall = Search<int8_t>(cloneIndex, queryset, vecset, addvecset, K, truth, N, iter, iterations);
        std::cout << "[INFO] After Save, Clone and Load:" << " recall@" << K << "=" << recall << std::endl;
        static_cast<SPANN::Index<int8_t> *>(cloneIndex.get())->GetDBStat();

        auto t1 = std::chrono::high_resolution_clock::now();
        InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(cloneIndex.get()), 1, insertBatchSize, addvecset,
                              metaset, iter * insertBatchSize);
        std::cout << "[INFO] Insert time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count()
                  << " ms" << std::endl;
        
        for (int i = 0; i < deleteBatchSize; i++)
            cloneIndex->DeleteIndex(iter * deleteBatchSize + i);

        recall = Search<int8_t>(cloneIndex, queryset, vecset, addvecset, K, truth, N, iter + 1, iterations);
        std::cout << "[INFO] After iter " << iter << ": recall@" << K << "=" << recall << std::endl;
        static_cast<SPANN::Index<int8_t> *>(cloneIndex.get())->GetDBStat();

        BOOST_REQUIRE(cloneIndex->SaveIndex(clone_path) == ErrorCode::Success);
        cloneIndex = nullptr;
        prevPath = clone_path;
    }
 
    BOOST_REQUIRE(VectorIndex::LoadIndex(prevPath, finalIndex) == ErrorCode::Success);
    BOOST_REQUIRE(finalIndex != nullptr);
    auto t = std::chrono::high_resolution_clock::now();
    BOOST_REQUIRE(finalIndex->Check() == ErrorCode::Success);
    std::cout << "[INFO] Check time for iteration " << iterations << ": "
                << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t).count()
                << " ms" << std::endl;
    
    recall = Search<int8_t>(finalIndex, queryset, vecset, addvecset, K, truth, N, iterations, iterations);
    std::cout << "[INFO] After Save and Load:" << " recall@" << K << "=" << recall << std::endl;
    static_cast<SPANN::Index<int8_t> *>(finalIndex.get())->GetDBStat();
    finalIndex = nullptr;
    for (int iter = 0; iter < iterations; iter++)
    {
        std::filesystem::remove_all("clone_index_" + std::to_string(iter));
    }
    
    std::cout << "=================Enable Cache===================" << std::endl;
    prevPath = "original_index";
    for (int iter = 0; iter < iterations; iter++)
    {
        std::string clone_path = "clone_index_" + std::to_string(iter);
        std::shared_ptr<VectorIndex> prevIndex;
        BOOST_REQUIRE(VectorIndex::LoadIndex(prevPath, prevIndex) == ErrorCode::Success);
        BOOST_REQUIRE(prevIndex != nullptr);
        auto t0 = std::chrono::high_resolution_clock::now();
        BOOST_REQUIRE(prevIndex->Check() == ErrorCode::Success);
        std::cout << "[INFO] Check time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t0).count()
                  << " ms" << std::endl;
        

        prevIndex->SetParameter("CacheSizeGB", "4", "BuildSSDIndex");
        prevIndex->SetParameter("CacheShards", "2", "BuildSSDIndex");
        
        BOOST_REQUIRE(prevIndex->SaveIndex(prevPath) == ErrorCode::Success);
        auto cloneIndex = prevIndex->Clone(clone_path);

        recall = Search<int8_t>(cloneIndex, queryset, vecset, addvecset, K, truth, N, iter, iterations);
        std::cout << "[INFO] After Save, Clone and Load:" << " recall@" << K << "=" << recall << std::endl;
        static_cast<SPANN::Index<int8_t> *>(cloneIndex.get())->GetDBStat();

        auto t1 = std::chrono::high_resolution_clock::now();
        InsertVectors<int8_t>(static_cast<SPANN::Index<int8_t> *>(cloneIndex.get()), 1, insertBatchSize, addvecset,
                              metaset, iter * insertBatchSize);
        std::cout << "[INFO] Insert time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count()
                  << " ms" << std::endl;

        for (int i = 0; i < deleteBatchSize; i++)
            cloneIndex->DeleteIndex(iter * deleteBatchSize + i);

        recall = Search<int8_t>(cloneIndex, queryset, vecset, addvecset, K, truth, N, iter + 1, iterations);
        std::cout << "[INFO] After iter " << iter << ": recall@" << K << "=" << recall << std::endl;
        static_cast<SPANN::Index<int8_t> *>(cloneIndex.get())->GetDBStat();

        BOOST_REQUIRE(cloneIndex->SaveIndex(clone_path) == ErrorCode::Success);
        cloneIndex = nullptr;
        prevPath = clone_path;
    }
    BOOST_REQUIRE(VectorIndex::LoadIndex(prevPath, finalIndex) == ErrorCode::Success);
    BOOST_REQUIRE(finalIndex != nullptr);
    auto tt = std::chrono::high_resolution_clock::now();
    BOOST_REQUIRE(finalIndex->Check() == ErrorCode::Success);
    std::cout << "[INFO] Check time for iteration " << iterations << ": "
                << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - tt).count()
                << " ms" << std::endl;
    
    recall = Search<int8_t>(finalIndex, queryset, vecset, addvecset, K, truth, N, iterations, iterations);
    std::cout << "[INFO] After Save and Load:" << " recall@" << K << "=" << recall << std::endl;
    static_cast<SPANN::Index<int8_t> *>(finalIndex.get())->GetDBStat();
    finalIndex = nullptr;

    for (int iter = 0; iter < iterations; iter++)
    {
        std::filesystem::remove_all("clone_index_" + std::to_string(iter));
    }
    std::filesystem::remove_all("original_index");
}

BOOST_AUTO_TEST_CASE(IterativeSearchPerf)
{
    using namespace SPFreshTest;

    constexpr int insertIterations = 5;
    constexpr int insertBatchSize = 60000;
    constexpr int appendBatchSize = 40000;
    constexpr int dimension = 100;
    std::shared_ptr<VectorSet> vecset = get_embeddings<float>(0, insertBatchSize, dimension, -1);
    std::shared_ptr<MetadataSet> metaset = TestUtils::TestDataGenerator<float>::GenerateMetadataSet(insertBatchSize, 0);

    auto originalIndex = BuildIndex<float>("original_index", vecset, metaset);
    BOOST_REQUIRE(originalIndex != nullptr);
    BOOST_REQUIRE(originalIndex->SaveIndex("original_index") == ErrorCode::Success);
    originalIndex = nullptr;

    std::string prevPath = "original_index";
    for (int iter = 0; iter < insertIterations; iter++)
    {
        std::string clone_path = "clone_index_" + std::to_string(iter);
        std::shared_ptr<VectorIndex> prevIndex;
        BOOST_REQUIRE(VectorIndex::LoadIndex(prevPath, prevIndex) == ErrorCode::Success);
        BOOST_REQUIRE(prevIndex != nullptr);
        auto t0 = std::chrono::high_resolution_clock::now();
        BOOST_REQUIRE(prevIndex->Check() == ErrorCode::Success);
        std::cout << "Check time for iteration " << iter << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t0).count()
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

        BOOST_REQUIRE(cloneIndex->SaveIndex(clone_path) == ErrorCode::Success);
        cloneIndex = nullptr;
    }

    for (int iter = 0; iter < insertIterations; iter++)
    {
        std::filesystem::remove_all("clone_index_" + std::to_string(iter));
    }
    std::filesystem::remove_all("original_index");
}

BOOST_AUTO_TEST_CASE(BenchmarkFromConfig)
{
    using namespace SPFreshTest;

    // Check if benchmark config is provided via environment variable
    const char *configPath = std::getenv("BENCHMARK_CONFIG");
    if (configPath == nullptr)
    {
        BOOST_TEST_MESSAGE("Skipping benchmark test - BENCHMARK_CONFIG environment variable not set");
        return;
    }

    BOOST_TEST_MESSAGE("Running benchmark with config: " << configPath);

    // Read benchmark configuration
    Helper::IniReader iniReader;
    if (ErrorCode::Success != iniReader.LoadIniFile(configPath))
    {
        BOOST_FAIL("Failed to load benchmark config file: " << configPath);
        return;
    }

    // Parse config parameters
    std::string vectorPath = iniReader.GetParameter("Benchmark", "VectorPath", std::string(""));
    std::string queryPath = iniReader.GetParameter("Benchmark", "QueryPath", std::string(""));
    std::string truthPath = iniReader.GetParameter("Benchmark", "TruthPath", std::string(""));
    std::string indexPath = iniReader.GetParameter("Benchmark", "IndexPath", std::string("benchmark_index"));
    std::string quantizerFilePath = iniReader.GetParameter("Benchmark", "QuantizerFilePath", std::string(""));
    int quantizedDim = iniReader.GetParameter("Benchmark", "QuantizedDim", 0);

    VectorValueType valueType = VectorValueType::Float;
    std::string valueTypeStr = iniReader.GetParameter("Benchmark", "ValueType", std::string("Float"));
    if (valueTypeStr == "Float")
        valueType = VectorValueType::Float;
    else if (valueTypeStr == "Int8")
        valueType = VectorValueType::Int8;
    else if (valueTypeStr == "UInt8")
        valueType = VectorValueType::UInt8;

    int dimension = iniReader.GetParameter("Benchmark", "Dimension", 128);
    int baseVectorCount = iniReader.GetParameter("Benchmark", "BaseVectorCount", 8000);
    int insertVectorCount = iniReader.GetParameter("Benchmark", "InsertVectorCount", 2000);
    int deleteVectorCount = iniReader.GetParameter("Benchmark", "DeleteVectorCount", 2000);
    int batchNum = iniReader.GetParameter("Benchmark", "BatchNum", 100);
    int topK = iniReader.GetParameter("Benchmark", "TopK", 10);
    int numSearchThreads = iniReader.GetParameter("Benchmark", "NumSearchThreads", 8);
    int numInsertThreads = iniReader.GetParameter("Benchmark", "NumInsertThreads", 8);
    int appendThreadNum = iniReader.GetParameter("Benchmark", "AppendThreadNum", 0);
    int numSearchDuringInsertThreads = iniReader.GetParameter("Benchmark", "NumSearchDuringInsertThreads", 1);
    int numQueries = iniReader.GetParameter("Benchmark", "NumQueries", 1000);
    int layers = iniReader.GetParameter("Benchmark", "Layers", 1);
    DistCalcMethod distMethod = iniReader.GetParameter("Benchmark", "DistMethod", DistCalcMethod::L2);
    bool rebuild = (iniReader.GetParameter("Benchmark", "Rebuild", true) || iniReader.GetParameter("Benchmark", "RebuildSSDOnly", false));
    int resume = iniReader.GetParameter("Benchmark", "Resume", -1);

    BOOST_TEST_MESSAGE("=== Benchmark Configuration ===");
    BOOST_TEST_MESSAGE("Vector Path: " << vectorPath);
    BOOST_TEST_MESSAGE("Query Path: " << queryPath);
    BOOST_TEST_MESSAGE("Base Vectors: " << baseVectorCount);
    BOOST_TEST_MESSAGE("Insert Vectors: " << insertVectorCount);
    BOOST_TEST_MESSAGE("Dimension: " << dimension);
    BOOST_TEST_MESSAGE("Batch Number: " << batchNum);
    BOOST_TEST_MESSAGE("Top-K: " << topK);
    BOOST_TEST_MESSAGE("SearchThreads: " << numSearchThreads);
    BOOST_TEST_MESSAGE("InsertThreads: " << numInsertThreads);
    BOOST_TEST_MESSAGE("SearchDuringInsertThreads: " << numSearchDuringInsertThreads);
    BOOST_TEST_MESSAGE("Queries: " << numQueries);
    BOOST_TEST_MESSAGE("Layers: " << layers);
    BOOST_TEST_MESSAGE("DistMethod: " << Helper::Convert::ConvertToString(distMethod));
    if (!quantizerFilePath.empty())
    {
        BOOST_TEST_MESSAGE("QuantizerFilePath: " << quantizerFilePath);
        BOOST_TEST_MESSAGE("QuantizedDim: " << quantizedDim);
    }

    // Get output file path from environment variable or use default
    const char *outputPath = std::getenv("BENCHMARK_OUTPUT");
    std::string outputFile = outputPath ? std::string(outputPath) : "output.json";
    BOOST_TEST_MESSAGE("Output File: " << outputFile);

    // Dispatch to appropriate type
    if (valueType == VectorValueType::Float)
    {
        RunBenchmark<float>(vectorPath, queryPath, truthPath, distMethod, indexPath, dimension, baseVectorCount,
                    insertVectorCount, deleteVectorCount, batchNum, topK, numSearchThreads, numInsertThreads, numSearchDuringInsertThreads, numQueries, iniReader, 
                    outputFile, rebuild, resume, quantizerFilePath, quantizedDim, layers);
    }
    else if (valueType == VectorValueType::Int8)
    {
        RunBenchmark<std::int8_t>(vectorPath, queryPath, truthPath, distMethod, indexPath, dimension, baseVectorCount,
                      insertVectorCount, deleteVectorCount, batchNum, topK, numSearchThreads, numInsertThreads, numSearchDuringInsertThreads, numQueries, iniReader,
                      outputFile, rebuild, resume, quantizerFilePath, quantizedDim, layers);
    }
    else if (valueType == VectorValueType::UInt8)
    {
        RunBenchmark<std::uint8_t>(vectorPath, queryPath, truthPath, distMethod, indexPath, dimension, baseVectorCount,
                       insertVectorCount, deleteVectorCount, batchNum, topK, numSearchThreads, numInsertThreads, numSearchDuringInsertThreads, numQueries, iniReader,
                       outputFile, rebuild, resume, quantizerFilePath, quantizedDim, layers);
    }

    //std::filesystem::remove_all(indexPath);
}
BOOST_AUTO_TEST_SUITE_END()
