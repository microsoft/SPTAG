// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/Distributed/WorkerNode.h"
#include "inc/Core/SPANN/Distributed/DispatcherNode.h"
#include "inc/Core/SPANN/ExtraDynamicSearcher.h"
#include "inc/Core/SPANN/ExtraTiKVController.h"
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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <ctime>
#include <tuple>
#include <vector>

#ifndef _MSC_VER
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
#endif

using namespace SPTAG;

// Helper: parse "host:port,host:port,..." into vector of pairs.
static std::vector<std::pair<std::string, std::string>> ParseNodeAddrs(const std::string& addrStr) {
    std::vector<std::pair<std::string, std::string>> result;
    auto parts = Helper::StrUtils::SplitString(addrStr, ",");
    for (auto& part : parts) {
        auto hp = Helper::StrUtils::SplitString(part, ":");
        if (hp.size() == 2) result.emplace_back(hp[0], hp[1]);
    }
    return result;
}

// Helper: bind a WorkerNode to ALL ExtraDynamicSearcher layers inside a VectorIndex.
// Calls SetWorker() which wires up append, head-sync, and remote-lock callbacks.
// All layers must have the worker bound so that AddIDCapacity (called per-layer) sees
// the correct numNodes and grows each layer's TiKVVersionMap to cover the full global
// VID space (capa * numNodes), not just this node's slice.
template <typename T>
static void BindWorkerToIndex(SPANN::WorkerNode* worker, std::shared_ptr<VectorIndex>& index) {
    auto* spannIndex = dynamic_cast<SPANN::Index<T>*>(index.get());
    if (!spannIndex) return;
    for (int layer = 0; ; ++layer) {
        auto diskIndex = spannIndex->GetDiskIndex(layer);
        if (!diskIndex) break;
        auto* searcher = dynamic_cast<SPANN::ExtraDynamicSearcher<T>*>(diskIndex.get());
        if (searcher) searcher->SetWorker(worker);
    }
}

// Helper: same as BindWorkerToIndex but takes a raw SPANN::Index<T>* directly
// (for sites that have already extracted the spannIndex pointer).
template <typename T>
static void BindWorkerToAllLayers(SPANN::WorkerNode* worker, SPANN::Index<T>* spannIndex) {
    if (!spannIndex) return;
    for (int layer = 0; ; ++layer) {
        auto diskIndex = spannIndex->GetDiskIndex(layer);
        if (!diskIndex) break;
        auto* searcher = dynamic_cast<SPANN::ExtraDynamicSearcher<T>*>(diskIndex.get());
        if (searcher) searcher->SetWorker(worker);
    }
}

// Configuration for distributed mode, read from [Distributed] ini section.
struct DistributedConfig {
    bool enabled = false;
    int workerIndex = 0;          // 0-based: 0 = driver (dispatcher + worker 0), 1+ = remote worker
    std::string dispatcherAddr;   // "host:port"
    std::string workerAddrs;      // "host:port,host:port,..."
    std::string storeAddrs;       // "addr,addr,..."
    std::string pdAddrs;          // "host:port,host:port,..." (per-worker PD)

    // Number of workers (for query/insert partitioning)
    int GetNumWorkers() const {
        if (!enabled || workerAddrs.empty()) return 1;
        return (int)std::count(workerAddrs.begin(), workerAddrs.end(), ',') + 1;
    }

    // Parse dispatcher address into host:port pair
    std::pair<std::string, std::string> GetDispatcherAddr() const {
        auto hp = Helper::StrUtils::SplitString(dispatcherAddr, ":");
        if (hp.size() == 2) return {hp[0], hp[1]};
        return {"", ""};
    }

    // Get PD address for this worker (falls back to global TiKVPDAddresses)
    std::string GetLocalPDAddr() const {
        if (pdAddrs.empty()) return "";
        auto addrs = Helper::StrUtils::SplitString(pdAddrs, ",");
        if (workerIndex < (int)addrs.size()) return addrs[workerIndex];
        return addrs[0];
    }

    static DistributedConfig FromIni(Helper::IniReader& ini) {
        DistributedConfig cfg;
        cfg.enabled = ini.GetParameter("Distributed", "Enabled", false);
        cfg.dispatcherAddr = ini.GetParameter("Distributed", "DispatcherAddr", std::string(""));
        cfg.workerAddrs = ini.GetParameter("Distributed", "WorkerAddrs", std::string(""));
        cfg.storeAddrs = ini.GetParameter("Distributed", "StoreAddrs", std::string(""));
        cfg.pdAddrs = ini.GetParameter("Distributed", "PDAddrs", std::string(""));

        // Worker index from env var (0 = driver, 1+ = remote worker)
        const char* wiEnv = std::getenv("WORKER_INDEX");
        cfg.workerIndex = wiEnv ? std::atoi(wiEnv) : 0;

        return cfg;
    }
};

namespace SPFreshTest
{
SizeType N = 10000;
DimensionType M = 100;
int K = 10;
int queries = 10;

struct LatencySummary
{
    double mean = 0;
    double p50 = 0;
    double p90 = 0;
    double p95 = 0;
    double p99 = 0;
};

LatencySummary SummarizeLatencyValues(std::vector<double> values)
{
    LatencySummary summary;
    if (values.empty()) return summary;

    for (double value : values) summary.mean += value;
    summary.mean /= values.size();

    std::sort(values.begin(), values.end());
    auto percentile = [&values](double ratio) -> double {
        size_t index = static_cast<size_t>(values.size() * ratio);
        if (index >= values.size()) index = values.size() - 1;
        return values[index];
    };

    summary.p50 = percentile(0.50);
    summary.p90 = percentile(0.90);
    summary.p95 = percentile(0.95);
    summary.p99 = percentile(0.99);
    return summary;
}

template <typename Getter>
LatencySummary SummarizeSearchStats(const std::vector<SPANN::SearchStats>& stats, Getter getter)
{
    std::vector<double> values;
    values.reserve(stats.size());
    for (const auto& stat : stats) values.push_back(getter(stat));
    return SummarizeLatencyValues(std::move(values));
}

template <typename Getter>
double MeanSearchStats(const std::vector<SPANN::SearchStats>& stats, Getter getter)
{
    if (stats.empty()) return 0;
    double total = 0;
    for (const auto& stat : stats) total += getter(stat);
    return total / stats.size();
}

void WriteLatencySummary(std::ostream& out, const std::string& prefix, const char* name, const LatencySummary& summary, bool comma = true)
{
    out << prefix << "        \"" << name << "\": {"
        << "\"mean\": " << summary.mean
        << ", \"p50\": " << summary.p50
        << ", \"p90\": " << summary.p90
        << ", \"p95\": " << summary.p95
        << ", \"p99\": " << summary.p99
        << "}" << (comma ? "," : "") << "\n";
}

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
                                        std::string& pmetaset, std::string& pmetaidx, const std::string &distMethod = "L2",
                                        int searchthread = 2, int insertthread = 2, int layers = 1,
                                        std::shared_ptr<COMMON::IQuantizer> quantizer = nullptr, std::string quantizerFilePath = "quantizer.bin",
                                        const std::map<std::string, std::string>& ssdOverrides = {},
                                        bool ssdOnly = false,
                                        SPANN::WorkerNode* p_worker = nullptr)
{
    auto vecIndex = VectorIndex::CreateInstance(IndexAlgoType::SPANN, GetEnumValueType<T>());
    int maxthreads = std::thread::hardware_concurrency();
    int postingLimit = 4 * sizeof(T);
    remove((outDirectory + FolderSep + "ssdmapping_0_postings").c_str());
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

    // Apply overrides (e.g., Storage, TiKV settings, SelectHead/BuildHead params)
    for (const auto &[key, val] : ssdOverrides)
    {
        // Keys prefixed with "SectionName." are routed to the corresponding section
        auto dotPos = key.find('.');
        if (dotPos != std::string::npos) {
            std::string section = key.substr(0, dotPos);
            std::string param = key.substr(dotPos + 1);
            vecIndex->SetParameter(param.c_str(), val.c_str(), section.c_str());
        } else {
            vecIndex->SetParameter(key.c_str(), val.c_str(), "BuildSSDIndex");
        }
    }

    // SSD-only mode: skip SelectHead and BuildHead, resume from specified layer
    if (ssdOnly)
    {
        // Allow explicit ResumeLayer from config/overrides; otherwise default to layer 0
        // (rebuild SSD for all layers, reusing existing head indexes)
        int resumeLayer = 0;
        vecIndex->SetParameter("ResumeLayer", std::to_string(resumeLayer).c_str(), "BuildSSDIndex");
    }

    if (quantizer)
    {
        vecIndex->SetParameter("QuantizerFilePath", quantizerFilePath.c_str(), "Base");
        vecIndex->SetQuantizer(quantizer);
        vecIndex->SetQuantizerADC(false);
        vecIndex->SetParameter("Dim", std::to_string(quantizer->GetNumSubvectors()).c_str(), "Base");
    }

    // Bind a routing worker (if any) to the freshly-created SSD searcher
    // before BuildIndex runs. Build itself does not route postings any more
    // (shared TiKV cluster — driver writes directly), so in buildOnly mode
    // the workerPtr will simply be nullptr and this block is a no-op.
    if (p_worker) {
        if (auto* spannIdx = dynamic_cast<SPANN::Index<T>*>(vecIndex.get())) {
            spannIdx->SetWorker(p_worker);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "BuildLargeIndex: bound routing worker (numNodes=%d)\n",
                p_worker->GetNumNodes());
        }
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

template <typename T>
double ExecutePartitionedSearch(VectorIndex* index,
                                std::shared_ptr<VectorSet>& queryset,
                                int myStart, int myCount,
                                int searchK, int numThreads,
                                std::vector<QueryResult>& results,
                                std::vector<float>* latenciesOut,
                                std::vector<SPANN::SearchStats>* statsOut);

template <typename ValueType>
void InsertVectors(SPANN::Index<ValueType> *p_index, int insertThreads, int step,
                   std::shared_ptr<VectorSet> addset, std::shared_ptr<MetadataSet> &metaset, int searchThreads = 0, std::shared_ptr<VectorSet> queryset = nullptr, int numQueries = 0, int k = 5, std::ostream* benchmarkData = nullptr, int start = 0,
                   SPANN::WorkerNode* router = nullptr)
{
    p_index->ForceCompaction();
    p_index->GetDBStat();

    std::vector<std::thread> threads;

    int printstep = step / 50;

    // Bulk path: single AddIndex call amortizes remote-append RPCs into one AppendBatchAsync.
    // Per-vector RNGSelection is parallelized inside ExtraDynamicSearcher::AddIndex so we
    // keep insertThreads-way parallelism while saving N-1 RPCs.
    bool useBulk = (router && router->GetNumNodes() > 1);

    // Per-vector insert (original path): each thread grabs one vector at a time
    std::atomic_size_t vectorsSent(start);
    auto perVecFunc = [&]() {
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

    // Bulk insert (router path): single call, parallelism inside SPANNIndex::AddIndex
    auto bulkFunc = [&]() {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "InsertVectors: bulk AddIndex for %d vectors (router enabled)\n", step);
        ErrorCode ret = p_index->AddIndex(addset->GetVector((SizeType)start), step, addset->Dimension(), metaset, true);
        if (ret != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "AddIndex bulk failed. start:%d count:%d Dim:%d Error:%d\n",
                         start, step, addset->Dimension(), static_cast<int>(ret));
        }
        BOOST_REQUIRE(ret == ErrorCode::Success);
    };

    std::function<void()> func;
    int insertThreadCount;
    if (useBulk) {
        func = bulkFunc;
        insertThreadCount = 1;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "InsertVectors: bulk path - driver launcher=1, internal parallelism comes from "
                     "[BuildSSDIndex] AppendThreadNum (user-supplied InsertThreadNum=%d is unused on this path)\n",
                     insertThreads);
    } else {
        func = perVecFunc;
        insertThreadCount = insertThreads;
    }

    bool withSearch = (searchThreads > 0 && queryset != nullptr && numQueries != 0 && benchmarkData != nullptr);

    for (int j = 0; j < insertThreadCount; j++)
    {
        threads.emplace_back(func);
    }

    std::vector<float> latencies;
    std::vector<QueryResult> results;
    double searchWallSeconds = 0.0;
    std::thread searchThread;
    if (withSearch) {
        searchThread = std::thread([&]() {
            searchWallSeconds = ExecutePartitionedSearch<ValueType>(
                p_index, queryset, /*myStart=*/0, numQueries, k, searchThreads,
                results, &latencies, /*statsOut=*/nullptr);
        });
    }

    for (auto &thread : threads)
    {
        thread.join();
    }
    if (withSearch) {
        searchThread.join();

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
        float qps = numQueries / std::max(static_cast<float>(searchWallSeconds), 1e-6f);

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
    auto barrierStart = std::chrono::high_resolution_clock::now();
    size_t barrierPolls = 0;
    while (!p_index->AllFinished())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        barrierPolls++;
    }
    auto barrierEnd = std::chrono::high_resolution_clock::now();
    double barrierSeconds = std::chrono::duration_cast<std::chrono::microseconds>(barrierEnd - barrierStart).count() / 1000000.0;
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "[DIAG] BatchBarrierWait seconds=%.6f polls=%zu\n",
                 barrierSeconds, barrierPolls);
    if (benchmarkData != nullptr)
    {
        *benchmarkData << "        \"batch barrier waitSeconds\": " << barrierSeconds << ",\n";
    }
}


// Dump per-query top-K results to disk in the format consumed by the offline
// union-recall / cross-validation tooling (added in #452):
//   [int64 numQueries][int32 topK][numQueries * topK * (int64 VID + float Dist)]
// Out-of-range / empty slots are written as VID=-1 / Dist=FLT_MAX.
//
// In a single-node run `results` covers the whole query set, so the file is a
// complete baseline. In a distributed run each node only holds its own
// contiguous query slice [myStart, myStart+count); callers pass count = slice
// length and suffix the path with ".node<idx>", so the disjoint per-node files
// can be concatenated in node order offline and diffed against the single-node
// baseline to independently validate head-replication / posting consistency.
static void DumpSearchResultFile(const std::string& path,
                                 const std::vector<QueryResult>& results,
                                 int count, int topK, int searchK)
{
    std::ofstream fout(path, std::ios::binary | std::ios::trunc);
    if (!fout.is_open()) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
            "[SearchResult] Failed to open dump path: %s\n", path.c_str());
        return;
    }
    std::int64_t nq = count;
    std::int32_t tk = topK;
    fout.write(reinterpret_cast<const char*>(&nq), sizeof(nq));
    fout.write(reinterpret_cast<const char*>(&tk), sizeof(tk));
    for (int q = 0; q < count; ++q) {
        for (int kk = 0; kk < topK; ++kk) {
            const auto* rr = results[q].GetResult(kk);
            std::int64_t vid = (rr && kk < searchK) ? static_cast<std::int64_t>(rr->VID) : -1;
            float dist = (rr && kk < searchK) ? rr->Dist : (std::numeric_limits<float>::max)();
            fout.write(reinterpret_cast<const char*>(&vid), sizeof(vid));
            fout.write(reinterpret_cast<const char*>(&dist), sizeof(dist));
        }
    }
    fout.close();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
        "[SearchResult] Dumped %d queries x %d topK to %s\n", count, topK, path.c_str());
}

template <typename T>
void BenchmarkQueryPerformance(std::shared_ptr<VectorIndex> &index, std::shared_ptr<VectorSet> &queryset,
                               std::shared_ptr<VectorSet> &truth, const std::string &truthPath,
                               SizeType baseVectorCount, int topK, int searchK, int numThreads, int numQueries, int batches, int totalbatches,
                               std::ostream &benchmarkData, std::string prefix = "",
                               int nodeIndex = 0, SPANN::WorkerNode* router = nullptr,
                               SPANN::DispatcherNode* dispatcher = nullptr,
                               const std::string& searchResultPath = "")
{
    // Use hash ring node count (workers only) for partitioning, not GetNumNodes() (includes dispatcher)
    auto ring = (router && router->IsEnabled()) ? router->GetHashRing() : nullptr;
    int nodeCount = ring ? static_cast<int>(ring->NodeCount()) : 1;
    bool distributed = (dispatcher != nullptr && router != nullptr && router->IsEnabled() && nodeCount > 1);

    // Determine this node's query range (balanced contiguous partition)
    int myStart = 0, myCount = numQueries;
    if (distributed) {
        myStart = (int)((long long)nodeIndex * numQueries / nodeCount);
        int myEnd = (int)((long long)(nodeIndex + 1) * numQueries / nodeCount);
        myCount = myEnd - myStart;
    }

    // Dispatch search command to all workers via TCP (distributed only)
    std::int64_t dispatchId = -1;
    int round = 0;
    if (distributed) {
        static std::atomic<int> s_searchRound{0};
        round = s_searchRound.fetch_add(1);
        dispatchId = dispatcher->BroadcastDispatchCommand(
            SPANN::DispatchCommand::Type::Search, static_cast<std::uint32_t>(round));
    }

    // Run this node's share of queries.
    std::vector<QueryResult> results;
    std::vector<float> latencies;
    std::vector<SPANN::SearchStats> searchStats;
    double localWallTime = ExecutePartitionedSearch<T>(
        index.get(), queryset, myStart, myCount, searchK, numThreads,
        results, &latencies, &searchStats);
    float batchLatency = static_cast<float>(localWallTime);
    auto* spannIndex = dynamic_cast<SPANN::Index<T>*>(index.get());

    if (distributed) {
        // Driver also runs searches against its local node, so it can have
        // outgoing merge hints queued. Drain before we move on.
        if (router) {
            router->FlushRemoteMerges();
        }
        // Collect worker timings via TCP; QPS is governed by the slowest node.
        auto workerTimes = dispatcher->WaitForAllResults(dispatchId, 300);
        for (double wt : workerTimes) {
            batchLatency = std::max(batchLatency, static_cast<float>(wt));
        }
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
            "BenchmarkQueryPerformance round %d: local=%.1fms (%d queries), max=%.1fms, QPS=%.1f\n",
            round, localWallTime * 1000, myCount, batchLatency * 1000, numQueries / batchLatency);
    }

    // Cross-validation dump: persist this node's per-query top-K so a
    // distributed run can be diffed offline against a single-node baseline.
    // Single-node writes the full set to <path> (baseline); each distributed
    // node writes only its contiguous slice to <path>.node<idx>.
    if (!searchResultPath.empty()) {
        std::string dumpPath = distributed
            ? (searchResultPath + ".node" + std::to_string(nodeIndex))
            : searchResultPath;
        DumpSearchResultFile(dumpPath, results, myCount, topK, searchK);
    }

    // Calculate statistics (from this node's queries)
    int statsCount = myCount;
    float mean = 0, minLat = (std::numeric_limits<float>::max)(), maxLat = 0;
    for (int i = 0; i < statsCount; i++)
    {
        mean += latencies[i];
        minLat = (std::min)(minLat, latencies[i]);
        maxLat = (std::max)(maxLat, latencies[i]);
    }
    mean /= statsCount;

    std::sort(latencies.begin(), latencies.end());
    float p50 = latencies[static_cast<size_t>(statsCount * 0.50)];
    float p90 = latencies[static_cast<size_t>(statsCount * 0.90)];
    float p95 = latencies[static_cast<size_t>(statsCount * 0.95)];
    float p99 = latencies[static_cast<size_t>(statsCount * 0.99)];
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

    if (spannIndex != nullptr)
    {
        auto postingReadSummary = SummarizeSearchStats(searchStats, [](const SPANN::SearchStats& stat) { return stat.m_diskReadLatency; });
        auto versionMapSummary = SummarizeSearchStats(searchStats, [](const SPANN::SearchStats& stat) { return stat.m_versionMapLatency; });
        auto compSummary = SummarizeSearchStats(searchStats, [](const SPANN::SearchStats& stat) { return stat.m_compLatency; });
        auto setupSummary = SummarizeSearchStats(searchStats, [](const SPANN::SearchStats& stat) { return stat.m_exSetUpLatency; });

        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "[SearchBreakdown] postingReadMs mean=%.4lf p50=%.4lf p95=%.4lf p99=%.4lf, versionMapMs mean=%.4lf p50=%.4lf p95=%.4lf p99=%.4lf\n",
                     postingReadSummary.mean, postingReadSummary.p50, postingReadSummary.p95, postingReadSummary.p99,
                     versionMapSummary.mean, versionMapSummary.p50, versionMapSummary.p95, versionMapSummary.p99);

        std::vector<int> activeLayers;
        for (int layer = 0; layer < SPANN::SearchStats::kSearchLatencyBreakdownLayers; layer++)
        {
            bool active = false;
            for (const auto& stat : searchStats)
            {
                if (stat.m_layerPostingCount[layer] > 0 ||
                    stat.m_layerListElementsCount[layer] > 0 ||
                    stat.m_layerVersionCheckCount[layer] > 0 ||
                    stat.m_layerTotalLatency[layer] > 0)
                {
                    active = true;
                    break;
                }
            }
            if (active) activeLayers.push_back(layer);
        }

        benchmarkData << prefix << "      \"latencyBreakdown\": {\n";
        std::string breakdownPrefix = prefix + "  ";
        WriteLatencySummary(benchmarkData, breakdownPrefix, "postingReadMs", postingReadSummary);
        WriteLatencySummary(benchmarkData, breakdownPrefix, "versionMapMs", versionMapSummary);
        WriteLatencySummary(benchmarkData, breakdownPrefix, "compMs", compSummary);
        WriteLatencySummary(benchmarkData, breakdownPrefix, "setupMs", setupSummary);
        benchmarkData << prefix << "        \"layers\": {\n";
        for (size_t layerIndex = 0; layerIndex < activeLayers.size(); layerIndex++)
        {
            int layer = activeLayers[layerIndex];
            auto layerPostingReadSummary = SummarizeSearchStats(searchStats, [layer](const SPANN::SearchStats& stat) { return stat.m_layerPostingReadLatency[layer]; });
            auto layerVersionMapSummary = SummarizeSearchStats(searchStats, [layer](const SPANN::SearchStats& stat) { return stat.m_layerVersionMapLatency[layer]; });
            auto layerCompSummary = SummarizeSearchStats(searchStats, [layer](const SPANN::SearchStats& stat) { return stat.m_layerCompLatency[layer]; });
            auto layerSetupSummary = SummarizeSearchStats(searchStats, [layer](const SPANN::SearchStats& stat) { return stat.m_layerSetupLatency[layer]; });
            auto layerTotalSummary = SummarizeSearchStats(searchStats, [layer](const SPANN::SearchStats& stat) { return stat.m_layerTotalLatency[layer]; });

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                         "[SearchBreakdown][Layer %d] postingReadMs mean=%.4lf p95=%.4lf, versionMapMs mean=%.4lf p95=%.4lf, compMs mean=%.4lf, setupMs mean=%.4lf, totalMs mean=%.4lf\n",
                         layer,
                         layerPostingReadSummary.mean, layerPostingReadSummary.p95,
                         layerVersionMapSummary.mean, layerVersionMapSummary.p95,
                         layerCompSummary.mean,
                         layerSetupSummary.mean,
                         layerTotalSummary.mean);

            std::string layerPrefix = prefix + "          ";
            benchmarkData << prefix << "          \"" << layer << "\": {\n";
            WriteLatencySummary(benchmarkData, layerPrefix, "postingReadMs", layerPostingReadSummary);
            WriteLatencySummary(benchmarkData, layerPrefix, "versionMapMs", layerVersionMapSummary);
            WriteLatencySummary(benchmarkData, layerPrefix, "compMs", layerCompSummary);
            WriteLatencySummary(benchmarkData, layerPrefix, "setupMs", layerSetupSummary);
            WriteLatencySummary(benchmarkData, layerPrefix, "totalMs", layerTotalSummary);
            benchmarkData << prefix << "            \"postingCountMean\": " << MeanSearchStats(searchStats, [layer](const SPANN::SearchStats& stat) { return stat.m_layerPostingCount[layer]; }) << ",\n";
            benchmarkData << prefix << "            \"versionCheckCountMean\": " << MeanSearchStats(searchStats, [layer](const SPANN::SearchStats& stat) { return stat.m_layerVersionCheckCount[layer]; }) << ",\n";
            benchmarkData << prefix << "            \"listElementsMean\": " << MeanSearchStats(searchStats, [layer](const SPANN::SearchStats& stat) { return stat.m_layerListElementsCount[layer]; }) << "\n";
            benchmarkData << prefix << "          }" << (layerIndex + 1 < activeLayers.size() ? "," : "") << "\n";
        }
        benchmarkData << prefix << "        }\n";
        benchmarkData << prefix << "      },\n";
    }

    // Recall evaluation
    if (!truth || truthPath.empty() || truthPath == "none")
    {
        BOOST_TEST_MESSAGE("  Recall evaluation skipped (no truth data)");
        benchmarkData << prefix << "      \"recall\": null\n";
        benchmarkData << prefix << "    }";
        return;
    }

    BOOST_TEST_MESSAGE("Checking for truth file: " << truthPath);
    std::shared_ptr<VectorSet> pvecset, paddvecset;
    // In distributed mode, this node only searched queries [myStart, myStart+myCount).
    // Pass the global query count and this node's offset so EvaluateRecall indexes
    // the truth file in global terms (BATCH > 0 reads the wrong truth rows otherwise).
    int recallTotalQueries = distributed ? numQueries : -1;
    int recallQueryOffset = distributed ? myStart : 0;
    float avgRecall = TestUtils::TestDataGenerator<T>::EvaluateRecall(results, truth, topK, searchK, batches, totalbatches,
                                                                      recallTotalQueries, recallQueryOffset);
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

// Run [myStart, myStart+myCount) queries against `index` using `numThreads` workers.
// Returns wall time in seconds. Fills `results` and (when non-null) per-query
// `latenciesOut` (ms) and `statsOut` (SPANN SearchStats). When `statsOut` is
// non-null and the index is a SPANN index, the stats overload of SearchIndex
// is used; otherwise the plain SearchIndex path runs.
template <typename T>
double ExecutePartitionedSearch(VectorIndex* index,
                                std::shared_ptr<VectorSet>& queryset,
                                int myStart, int myCount,
                                int searchK, int numThreads,
                                std::vector<QueryResult>& results,
                                std::vector<float>* latenciesOut,
                                std::vector<SPANN::SearchStats>* statsOut)
{
    auto* spannIndex = dynamic_cast<SPANN::Index<T>*>(index);
    bool useStats = (statsOut != nullptr && spannIndex != nullptr);

    results.resize(myCount);
    for (int i = 0; i < myCount; i++) {
        results[i] = QueryResult((const T*)queryset->GetVector(myStart + i), searchK, false);
    }
    if (useStats) statsOut->assign(myCount, SPANN::SearchStats());
    if (latenciesOut) latenciesOut->assign(myCount, 0.0f);

    std::atomic_size_t queriesSent(0);
    int nThreads = std::min(numThreads, std::max(myCount, 1));
    std::vector<std::thread> threads;
    threads.reserve(nThreads);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nThreads; i++) {
        threads.emplace_back([&]() {
            size_t qid;
            while ((qid = queriesSent.fetch_add(1)) < static_cast<size_t>(myCount)) {
                auto t1 = std::chrono::high_resolution_clock::now();
                if (useStats) {
                    spannIndex->SearchIndex(results[qid], &(*statsOut)[qid]);
                } else if (spannIndex != nullptr) {
                    spannIndex->SearchIndex(results[qid]);
                } else {
                    index->SearchIndex(results[qid]);
                }
                auto t2 = std::chrono::high_resolution_clock::now();
                if (latenciesOut) {
                    (*latenciesOut)[qid] =
                        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
                }
            }
        });
    }
    for (auto& t : threads) t.join();
    auto t3 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t3 - t0).count() / 1000000.0;
}

ErrorCode QuantizeVectors(const std::shared_ptr<COMMON::IQuantizer>& quantizer,
                          const std::shared_ptr<VectorSet>& source,
                          ByteArray& dest);

template <typename T>
void LoadAndInsertBatch(SPANN::Index<T>* spannIndex,
                        const std::string& paddset,
                        const std::string& paddmeta,
                        const std::string& paddmetaidx,
                        int dimension,
                        int insertStart, int loadCount, int perNodeBatch,
                        int numInsertThreads,
                        SPANN::WorkerNode* router,
                        std::shared_ptr<COMMON::IQuantizer> quantizer,
                        int searchDuringInsertThreads,
                        std::shared_ptr<VectorSet> queryset,
                        int numQueries, int searchK,
                        std::ostream* benchmarkData,
                        const char* logPrefix)
{
    auto addset = TestUtils::TestDataGenerator<T>::LoadVectorSet(paddset, dimension, insertStart, loadCount);
    if (quantizer) {
        auto addFloat = ConvertToFloatVectorSet(addset);
        BOOST_REQUIRE(addFloat != nullptr);
        ByteArray quantizedAddBytes =
            ByteArray::Alloc((size_t)addFloat->Count() * (size_t)(quantizer->GetNumSubvectors()));
        BOOST_REQUIRE(QuantizeVectors(quantizer, addFloat, quantizedAddBytes) == ErrorCode::Success);
        addset = std::make_shared<BasicVectorSet>(quantizedAddBytes,
                                                  VectorValueType::UInt8,
                                                  quantizer->GetNumSubvectors(),
                                                  addFloat->Count());
    }
    auto addmetaset = TestUtils::TestDataGenerator<T>::LoadMetadataSet(paddmeta, paddmetaidx, insertStart, loadCount);
    InsertVectors<T>(spannIndex, numInsertThreads, perNodeBatch,
                     addset, addmetaset,
                     searchDuringInsertThreads, queryset, numQueries, searchK,
                     benchmarkData, 0, router);
    if (router) {
        router->FlushRemoteAppends();
        router->FlushRemoteMerges();
        router->LogRouteStats(" (batch flush)");
        router->ResetRouteStats();
    }
}

template <typename T>
void LogCheckpointLayerStats(const std::shared_ptr<VectorIndex>& index, int layers, int currentBatch, int totalBatches)
{
    auto spannIndex = std::dynamic_pointer_cast<SPANN::Index<T>>(index);
    if (!spannIndex) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                     "Checkpoint layer stats: batch %d/%d unable to cast index for layer stats\n",
                     currentBatch, totalBatches);
        return;
    }

    for (int layer = 0; layer <= layers; layer++) {
        std::vector<SizeType> headMapping;
        ErrorCode mappingRet = spannIndex->GetHeadIndexMapping(layer, headMapping);
        long long headMappingSize = mappingRet == ErrorCode::Success ? static_cast<long long>(headMapping.size()) : -1;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "Checkpoint layer stats: batch %d/%d layer=%d samples=%lld deleted=%lld headMapping=%lld\n",
                     currentBatch, totalBatches, layer,
                     static_cast<long long>(spannIndex->GetNumSamples(layer)),
                     static_cast<long long>(spannIndex->GetNumDeleted(layer)),
                     headMappingSize);
    }
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
                  int insertVectorCount, int deleteVectorCount, int batches, int topK, int numSearchThreads, int numInsertThreads, int numSearchDuringInsertThreads, int numQueries,
                  const std::string &outputFile = "output.json", const bool rebuild = true, const int resume = -1,
                  const std::string &quantizerFilePath = std::string(""), int quantizedDim = 0, int layers = 1,
                  const std::map<std::string, std::string>& ssdOverrides = {},
                  bool rebuildSsdOnly = false,
                  bool buildOnly = false,
                  const DistributedConfig& distCfg = {})
{
    int oldM = M, oldK = K, oldN = N, oldQueries = queries;
    N = baseVectorCount;
    queries = numQueries;
    M = dimension;
    K = topK;
    std::string dist = Helper::Convert::ConvertToString(distMethod);
    int insertBatchSize = insertVectorCount / max(batches, 1);
    int deleteBatchSize = deleteVectorCount / max(batches, 1);

    // Optional cross-validation dump path (Benchmark/SearchResult).  Re-read
    // from the config file since RunBenchmark isn't handed the IniReader.  When
    // set, each BenchmarkQueryPerformance call persists its per-query top-K;
    // see DumpSearchResultFile.  Last search round wins (final index state).
    std::string searchResultPath;
    if (const char* cfgPath = std::getenv("BENCHMARK_CONFIG")) {
        Helper::IniReader srIni;
        if (srIni.LoadIniFile(cfgPath) == ErrorCode::Success) {
            searchResultPath = srIni.GetParameter("Benchmark", "SearchResult", std::string(""));
        }
    }

    // Use distributed config for multi-node partitioning
    int nodeIndex = distCfg.workerIndex;
    int numNodes = distCfg.GetNumWorkers();
    int myInsertStart = (numNodes > 1) ? (nodeIndex * insertBatchSize) / numNodes : 0;
    int myInsertEnd = (numNodes > 1) ? ((nodeIndex + 1) * insertBatchSize) / numNodes : insertBatchSize;
    int perNodeBatch = myInsertEnd - myInsertStart;
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "RunBenchmark: nodeIndex=%d numNodes=%d insertBatchSize=%d myInsertStart=%d myInsertEnd=%d perNodeBatch=%d\n",
                 nodeIndex, numNodes, insertBatchSize, myInsertStart, myInsertEnd, perNodeBatch);

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
    // Distributed routing: dispatcher + local worker (driver node is both)
    std::unique_ptr<SPANN::DispatcherNode> dispatcher;
    std::unique_ptr<SPANN::WorkerNode> worker;
    SPANN::WorkerNode* workerPtr = nullptr;  // convenience alias
    std::shared_ptr<VectorIndex> index;
    std::shared_ptr<COMMON::IQuantizer> quantizer;

    // Distributed setup: when running a non-buildOnly distributed benchmark
    // (i.e. the search/insert run phase), create the dispatcher + worker0
    // so the driver can broadcast the hash ring and accept remote callbacks.
    // BuildOnly mode skips this entirely — build runs single-node and writes
    // straight to the shared TiKV cluster (PD routes each key to the owning
    // store), so no dispatcher / worker plumbing is needed for the build
    // path.
    if (distCfg.enabled && !buildOnly) {
        auto dispAddr = distCfg.GetDispatcherAddr();
        auto workerAddrs = ParseNodeAddrs(distCfg.workerAddrs);
        auto storeAddrs = Helper::StrUtils::SplitString(distCfg.storeAddrs, ",");

        dispatcher.reset(new SPANN::DispatcherNode());
        BOOST_REQUIRE_MESSAGE(dispatcher->Initialize(dispAddr, workerAddrs),
            "DispatcherNode initialization failed (build-phase setup)");
        BOOST_REQUIRE(dispatcher->Start());

        worker.reset(new SPANN::WorkerNode());
        // Pre-build: pass nullptr DB. After BuildIndex, swap in the real DB
        // via SetDB() (or rebuild the worker on top of it for run mode).
        BOOST_REQUIRE_MESSAGE(
            worker->Initialize(nullptr, 0, dispAddr, workerAddrs, storeAddrs),
            "WorkerNode initialization failed (build-phase setup)");
        BOOST_REQUIRE(worker->Start());
        workerPtr = worker.get();

        dispatcher->SetLocalWorkerIndex(worker->GetLocalNodeIndex());
        worker->SetHashRing(dispatcher->GetHashRing());

        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
            "Pre-build: waiting for all peer connections...\n");
        BOOST_REQUIRE_MESSAGE(dispatcher->WaitForAllPeersConnected(180),
            "Timed out waiting for peer connections (build-phase)");

        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(180);
        while (std::chrono::steady_clock::now() < deadline) {
            if (dispatcher->AllWorkersAcked()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        BOOST_REQUIRE_MESSAGE(dispatcher->AllWorkersAcked(),
            "Timed out waiting for workers to ACK ring (build-phase)");
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
            "Pre-build: all %d workers connected and ring synchronized\n", numNodes);

        // Start heartbeat pump so remote workers can detect driver failure
        // and exit cleanly instead of relying on a fixed wall-clock receiver
        // timeout. Worker side enforces HeartbeatTimeoutSec (default 180s).
        // Interval is fixed at 30s; six missed pings before worker bails.
        dispatcher->StartHeartbeat(30);
    }

    // Build initial index
    BOOST_TEST_MESSAGE("\n=== Building Index ===");
    if (rebuild || rebuildSsdOnly || !direxists(indexPath.c_str())) {
        if (!rebuildSsdOnly) {
            // Allow empty or non-existent directories; block only if index files already exist
            if (direxists(indexPath.c_str()) && fileexists((indexPath + FolderSep + "indexloader.ini").c_str())) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "Index directory '%s' already exists with index files. Refusing to delete. "
                    "Remove it manually or use RebuildSSDOnly=true to resume.\n",
                    indexPath.c_str());
                BOOST_FAIL("Index directory already exists: " + indexPath);
                return;
            }
        }
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

            index = BuildLargeIndex<uint8_t>(indexPath, pquanvecset, pmeta, pmetaidx, dist, numSearchThreads, numInsertThreads, layers, quantizer, "quantizer.bin", ssdOverrides, rebuildSsdOnly, workerPtr);
            BOOST_REQUIRE(index != nullptr);
            index->SetQuantizerADC(true);
        }
        else
        {
            index = BuildLargeIndex<T>(indexPath, pvecset, pmeta, pmetaidx, dist, numSearchThreads, numInsertThreads, layers, nullptr, "quantizer.bin", ssdOverrides, rebuildSsdOnly, workerPtr);
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

    // Set up distributed routing for RUN mode if configured.
    // (Build-phase needs no dispatcher/worker; the run-phase dispatcher+worker
    // were created in the pre-build block above.) The driver node is both
    // dispatcher (ring management) and worker 0 (compute).
    if (distCfg.enabled && !buildOnly) {
        // Bind worker to ALL searcher layers (wires append + headsync + lock + fetch callbacks).
        // Every layer must see the worker so AddIDCapacity grows each layer's
        // version map by capa * numNodes (not just capa).
        auto* spannIndex = dynamic_cast<SPANN::Index<T>*>(index.get());
        BOOST_REQUIRE(spannIndex != nullptr);
        BindWorkerToAllLayers<T>(workerPtr, spannIndex);

        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
            "Run mode: worker bound to all %d layers\n",
            (int)spannIndex->GetOptions()->m_layers);
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
        // Goal: allow pointing TruthPath at an arbitrary pre-computed truth file
        // instead of always loading the auto-generated perftest_batchtruth.* name.
        // The file is expected to be in the same format the generator writes (a
        // saved BasicVectorSet), so the standard loader parses it directly.
        std::string truthFile = ptruth;
        if (!truthPath.empty() && truthPath != "none" && fileexists(truthPath.c_str())) {
            BOOST_TEST_MESSAGE("Using TruthPath from config (overriding auto-generated name): " << truthPath);
            truthFile = truthPath;
        }
        truth = TestUtils::TestDataGenerator<float>::LoadVectorSet(truthFile, K);
    }

    // Benchmark 0/0b: query performance before insertions. Skip in BuildOnly
    // mode (no point measuring queries when we're about to exit; queries also
    // require workers to be running for distributed scatter-gather).
    if (!buildOnly) {
        // Benchmark 0: Query performance before insertions (round 1 — cold cache)
        BOOST_TEST_MESSAGE("\n=== Benchmark 0: Query Before Insertions (Round 1) ===");
        BenchmarkQueryPerformance<T>(index, queryset, truth, truthPath, baseVectorCount, topK, SearchK,
                                     numSearchThreads, numQueries, 0, batches, tmpbenchmark, "",
                                     nodeIndex, workerPtr, dispatcher.get(), searchResultPath);
        jsonFile << "    \"benchmark0_query_before_insert\": ";
        BenchmarkQueryPerformance<T>(index, queryset, truth, truthPath, baseVectorCount, topK, SearchK,
                                     numSearchThreads, numQueries, 0, batches, jsonFile, "",
                                     nodeIndex, workerPtr, dispatcher.get(), searchResultPath);
        jsonFile << ",\n";
        jsonFile.flush();

        // Benchmark 0b: Query performance before insertions (round 2 — warm cache)
        BOOST_TEST_MESSAGE("\n=== Benchmark 0b: Query Before Insertions (Round 2) ===");
        BenchmarkQueryPerformance<T>(index, queryset, truth, truthPath, baseVectorCount, topK, SearchK,
                                     numSearchThreads, numQueries, 0, batches, tmpbenchmark, "",
                                     nodeIndex, workerPtr, dispatcher.get(), searchResultPath);
        jsonFile << "    \"benchmark0b_query_before_insert_round2\": ";
        BenchmarkQueryPerformance<T>(index, queryset, truth, truthPath, baseVectorCount, topK, SearchK,
                                     numSearchThreads, numQueries, 0, batches, jsonFile, "",
                                     nodeIndex, workerPtr, dispatcher.get(), searchResultPath);
        jsonFile << ",\n";
        jsonFile.flush();
    } else {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "BuildOnly=true: skipping Benchmark 0/0b query rounds\n");
        jsonFile << "    \"benchmark0_query_before_insert\": {},\n";
        jsonFile << "    \"benchmark0b_query_before_insert_round2\": {},\n";
        jsonFile.flush();
    }

    BOOST_REQUIRE(index->SaveIndex(indexPath) == ErrorCode::Success);
    index = nullptr;


    // Benchmark 1: Insert performance
    if (buildOnly) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "BuildOnly=true: skipping insert batches, index saved to %s\n", indexPath.c_str());
        jsonFile << "    \"benchmark1_insert\": {}\n";
    }
    else if (insertBatchSize > 0)
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

                // Re-bind the worker to ALL layers of the new cloned index's searchers
                // (every layer must see the worker so AddIDCapacity grows each layer's
                // version map by capa * numNodes).
                if (workerPtr) {
                    BindWorkerToIndex<T>(workerPtr, cloneIndex);
                }

                // Dispatch insert command to workers via TCP
                std::uint64_t insertDispatchId = 0;
                if (dispatcher && numNodes > 1) {
                    insertDispatchId = dispatcher->BroadcastDispatchCommand(
                        SPANN::DispatchCommand::Type::Insert, static_cast<std::uint32_t>(iter));
                }

                // Each node inserts its contiguous slice
                // [iter*batchSize + myInsertStart, +perNodeBatch).
                int insertStart = iter * insertBatchSize + myInsertStart;
                int loadCount = perNodeBatch;
                {
                    std::string driverTag = "RunBenchmark iter=" + std::to_string(iter);
                    start = std::chrono::high_resolution_clock::now();
                    LoadAndInsertBatch<T>(static_cast<SPANN::Index<T>*>(cloneIndex.get()),
                                          paddset, paddmeta, paddmetaidx, M,
                                          insertStart, loadCount, perNodeBatch,
                                          numInsertThreads, workerPtr,
                                          enableQuantization ? quantizer : nullptr,
                                          numSearchDuringInsertThreads, queryset,
                                          numQueries, SearchK, &jsonFile,
                                          driverTag.c_str());
                }

                // Wait for all worker nodes to finish this batch via TCP.
                if (insertDispatchId > 0) {
                    auto workerTimes = dispatcher->WaitForAllResults(insertDispatchId, 7200);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Driver: all %d workers finished batch %d\n",
                                 (int)workerTimes.size(), iter + 1);
                }

                end = std::chrono::high_resolution_clock::now();
                seconds =
                    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0f;
                double throughput = insertBatchSize / seconds;

                BOOST_TEST_MESSAGE("  Inserted: " << insertBatchSize << " vectors (" << perNodeBatch << " local)");
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
                                             numQueries, iter + 1, batches, tmpbenchmark, "    ",
                                             nodeIndex, workerPtr, dispatcher.get(), searchResultPath);
                BenchmarkQueryPerformance<T>(cloneIndex, queryset, truth, truthPath, baseVectorCount,
                                             topK, SearchK, numSearchThreads, numQueries, iter + 1, batches, jsonFile, "    ",
                                             nodeIndex, workerPtr, dispatcher.get(), searchResultPath);
                jsonFile << ",\n";

                BOOST_TEST_MESSAGE("\n=== Benchmark 2b: Query After Insertions and Deletions (Round 2) ===");
                jsonFile << "        \"search_round2\":";
                BenchmarkQueryPerformance<T>(cloneIndex, queryset, truth, truthPath, baseVectorCount, topK, SearchK, numSearchThreads,
                                             numQueries, iter + 1, batches, tmpbenchmark, "    ",
                                             nodeIndex, workerPtr, dispatcher.get(), searchResultPath);
                BenchmarkQueryPerformance<T>(cloneIndex, queryset, truth, truthPath, baseVectorCount,
                                             topK, SearchK, numSearchThreads, numQueries, iter + 1, batches, jsonFile, "    ",
                                             nodeIndex, workerPtr, dispatcher.get(), searchResultPath);
                jsonFile << ",\n";

                start = std::chrono::high_resolution_clock::now();
                BOOST_REQUIRE(cloneIndex->SaveIndex(clonePath) == ErrorCode::Success);
                end = std::chrono::high_resolution_clock::now();

                seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0f;
                BOOST_TEST_MESSAGE("  Save Time: " << seconds << " seconds");
                BOOST_TEST_MESSAGE("  Save completed successfully");

                if (enableQuantization)
                    LogCheckpointLayerStats<uint8_t>(cloneIndex, layers, iter + 1, batches);
                else
                    LogCheckpointLayerStats<T>(cloneIndex, layers, iter + 1, batches);

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

    // Stop workers in distributed mode
    if (dispatcher && numNodes > 1) {
        // Stop the heartbeat pump first so we don't race a stray Heartbeat
        // packet against the Stop dispatch on the same connection.
        dispatcher->StopHeartbeat();
        auto dispatchId = dispatcher->BroadcastDispatchCommand(SPANN::DispatchCommand::Type::Stop, 0);
        // Wait briefly for ACKs so workers exit cleanly before the driver
        // tears down the network (which would force-kill in-flight RPCs).
        dispatcher->WaitForAllResults(dispatchId, 60);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Driver: sent Stop command to all workers\n");
    }

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

// Forward declaration
template <typename T>
void RunWorker(const std::string& indexPath, int dimension, int baseVectorCount,
               int insertVectorCount, int batches, int topK, int numSearchThreads,
               int numInsertThreads, int numQueries, VectorValueType valueType,
               const std::map<std::string, std::string>& ssdOverrides,
               const DistributedConfig& distCfg, int workerTimeout);

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
    int numSearchDuringInsertThreads = iniReader.GetParameter("Benchmark", "NumSearchDuringInsertThreads", 1);
    int appendThreadNum = iniReader.GetParameter("Benchmark", "AppendThreadNum", 0);
    int numQueries = iniReader.GetParameter("Benchmark", "NumQueries", 1000);
    int layers = iniReader.GetParameter("Benchmark", "Layers", 1);
    DistCalcMethod distMethod = iniReader.GetParameter("Benchmark", "DistMethod", DistCalcMethod::L2);
    bool rebuild = iniReader.GetParameter("Benchmark", "Rebuild", true);
    bool rebuildSsdOnly = iniReader.GetParameter("Benchmark", "RebuildSSDOnly", false);
    bool buildOnly = iniReader.GetParameter("Benchmark", "BuildOnly", false);
    int resume = iniReader.GetParameter("Benchmark", "Resume", -1);

    // Read storage backend overrides for BuildSSDIndex
    std::map<std::string, std::string> ssdOverrides;
    std::string storage = iniReader.GetParameter("Benchmark", "Storage", std::string(""));
    if (!storage.empty()) {
        ssdOverrides["Storage"] = storage;
    }
    std::string tikvKeyPrefix = iniReader.GetParameter("Benchmark", "TiKVKeyPrefix", std::string(""));
    if (!tikvKeyPrefix.empty()) {
        ssdOverrides["TiKVKeyPrefix"] = tikvKeyPrefix;
    }
    if (appendThreadNum > 0) {
        ssdOverrides["AppendThreadNum"] = std::to_string(appendThreadNum);
    }

    // Pass through any [BuildSSDIndex] section params from the ini as overrides
    auto buildSSDParams = iniReader.GetParameters("BuildSSDIndex");
    for (const auto &[key, val] : buildSSDParams) {
        ssdOverrides[key] = val;
    }

    // Read distributed config from [Distributed] section
    auto distCfg = DistributedConfig::FromIni(iniReader);

    // Shared TiKV raft cluster: every compute node connects to the FULL PD
    // endpoint list. The TiKV client uses PD-raft to route reads/writes to
    // whichever store owns the region, so any compute can access any posting.
    if (!distCfg.pdAddrs.empty()) {
        ssdOverrides["TiKVPDAddresses"] = distCfg.pdAddrs;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
            "Using PD address: %s (workerIndex=%d)\n",
            distCfg.pdAddrs.c_str(), distCfg.workerIndex);
    }

    // Pass through [SelectHead] and [BuildHead] params as overrides too
    auto selectHeadParams = iniReader.GetParameters("SelectHead");
    for (const auto &[key, val] : selectHeadParams) {
        ssdOverrides["SelectHead." + key] = val;
    }
    auto buildHeadParams = iniReader.GetParameters("BuildHead");
    for (const auto &[key, val] : buildHeadParams) {
        ssdOverrides["BuildHead." + key] = val;
    }

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

    // Worker node path: if distributed and workerIndex > 0, run as remote worker and return
    if (distCfg.enabled && distCfg.workerIndex > 0) {
        int workerTimeout = iniReader.GetParameter("Benchmark", "WorkerTimeout", 3600);
        BOOST_TEST_MESSAGE("Running as worker node " << distCfg.workerIndex);
        if (valueType == VectorValueType::Float)
            RunWorker<float>(indexPath, dimension, baseVectorCount, insertVectorCount, batchNum, topK, numSearchThreads, numInsertThreads, numQueries, valueType, ssdOverrides, distCfg, workerTimeout);
        else if (valueType == VectorValueType::Int8)
            RunWorker<std::int8_t>(indexPath, dimension, baseVectorCount, insertVectorCount, batchNum, topK, numSearchThreads, numInsertThreads, numQueries, valueType, ssdOverrides, distCfg, workerTimeout);
        else if (valueType == VectorValueType::UInt8)
            RunWorker<std::uint8_t>(indexPath, dimension, baseVectorCount, insertVectorCount, batchNum, topK, numSearchThreads, numInsertThreads, numQueries, valueType, ssdOverrides, distCfg, workerTimeout);
        return;
    }

    // Get output file path from environment variable or use default
    const char *outputPath = std::getenv("BENCHMARK_OUTPUT");
    std::string outputFile = outputPath ? std::string(outputPath) : "output.json";
    BOOST_TEST_MESSAGE("Output File: " << outputFile);

    // Driver path (nodeIndex == 0 or single-node mode)
    if (valueType == VectorValueType::Float)
    {
        RunBenchmark<float>(vectorPath, queryPath, truthPath, distMethod, indexPath, dimension, baseVectorCount,
                    insertVectorCount, deleteVectorCount, batchNum, topK, numSearchThreads, numInsertThreads, numSearchDuringInsertThreads, numQueries, outputFile, 
                    rebuild, resume, quantizerFilePath, quantizedDim, layers, ssdOverrides, rebuildSsdOnly, buildOnly, distCfg);
    }
    else if (valueType == VectorValueType::Int8)
    {
        RunBenchmark<std::int8_t>(vectorPath, queryPath, truthPath, distMethod, indexPath, dimension, baseVectorCount,
                      insertVectorCount, deleteVectorCount, batchNum, topK, numSearchThreads, numInsertThreads, numSearchDuringInsertThreads, numQueries,
                      outputFile, rebuild, resume, quantizerFilePath, quantizedDim, layers, ssdOverrides, rebuildSsdOnly, buildOnly, distCfg);
    }
    else if (valueType == VectorValueType::UInt8)
    {
        RunBenchmark<std::uint8_t>(vectorPath, queryPath, truthPath, distMethod, indexPath, dimension, baseVectorCount,
                       insertVectorCount, deleteVectorCount, batchNum, topK, numSearchThreads, numInsertThreads, numSearchDuringInsertThreads, numQueries,
                       outputFile, rebuild, resume, quantizerFilePath, quantizedDim, layers, ssdOverrides, rebuildSsdOnly, buildOnly, distCfg);
    }
}

/// Worker node path for distributed benchmark (nodeIndex > 0).
/// Loads a pre-built head index, connects to TiKV, starts WorkerNode,
/// and waits for TCP dispatch commands from the driver node.
template <typename T>
void RunWorker(const std::string& indexPath, int dimension, int baseVectorCount,
               int insertVectorCount, int batches, int topK, int numSearchThreads,
               int numInsertThreads, int numQueries, VectorValueType valueType,
               const std::map<std::string, std::string>& ssdOverrides,
               const DistributedConfig& distCfg, int workerTimeout)
{
    int oldN = N, oldM = M, oldK = K, oldQ = queries;
    N = baseVectorCount; M = dimension; K = topK; queries = numQueries;

    int nodeIndex = distCfg.workerIndex;
    int numNodes = distCfg.GetNumWorkers();
    int insertBatchSize = insertVectorCount / std::max(batches, 1);
    int myInsertStart = (numNodes > 1) ? (nodeIndex * insertBatchSize) / numNodes : 0;
    int myInsertEnd = (numNodes > 1) ? ((nodeIndex + 1) * insertBatchSize) / numNodes : insertBatchSize;
    int perNodeBatch = myInsertEnd - myInsertStart;

    BOOST_TEST_MESSAGE("Worker node " << nodeIndex << ": Loading index from " << indexPath);
    std::shared_ptr<VectorIndex> index;
    // IMPORTANT: Pass ssdOverrides through LoadIndex so that worker-specific settings
    // (especially TiKVPDAddresses pointing at this worker's local PD) are applied
    // BEFORE the underlying TiKV connection is constructed in PrepareDB. Without this,
    // the worker would inherit the driver's PD address from the saved indexloader.ini
    // and route every KV write back to the driver's TiKV instead of its own.
    BOOST_REQUIRE(VectorIndex::LoadIndex(indexPath, ssdOverrides, index) == ErrorCode::Success);
    BOOST_REQUIRE(index != nullptr);

    // Create WorkerNode
    auto dispAddr = distCfg.GetDispatcherAddr();
    auto workerAddrs = ParseNodeAddrs(distCfg.workerAddrs);
    auto storeAddrs = Helper::StrUtils::SplitString(distCfg.storeAddrs, ",");

    auto* spannIndex = dynamic_cast<SPANN::Index<T>*>(index.get());
    BOOST_REQUIRE_MESSAGE(spannIndex != nullptr, "Failed to cast to SPANN::Index<T>");
    auto diskIndex = spannIndex->GetDiskIndex(0);
    BOOST_REQUIRE(diskIndex != nullptr);
    auto* searcher = dynamic_cast<SPANN::ExtraDynamicSearcher<T>*>(diskIndex.get());
    BOOST_REQUIRE(searcher != nullptr);
    auto workerDb = searcher->GetDB();
    BOOST_REQUIRE_MESSAGE(workerDb != nullptr, "Worker: could not extract db from index");

    SPANN::WorkerNode workerNode;
    BOOST_REQUIRE_MESSAGE(workerNode.Initialize(workerDb, nodeIndex, dispAddr, workerAddrs, storeAddrs),
                          "WorkerNode initialization failed");
    BOOST_REQUIRE(workerNode.Start());
    auto* router = &workerNode;

    // Bind worker to ALL searcher layers (every layer must see the worker so
    // AddIDCapacity grows each layer's version map by capa * numNodes).
    BindWorkerToAllLayers<T>(router, spannIndex);

    // Wait for ring from dispatcher
    BOOST_REQUIRE_MESSAGE(router->WaitForRing(120),
                          "Worker: Timed out waiting for ring from dispatcher");

    BOOST_TEST_MESSAGE("Worker " << nodeIndex << ": Ready, numNodes=" << numNodes
                       << " perNodeBatch=" << perNodeBatch);

    // Build data file names
    std::string typeStr = Helper::Convert::ConvertToString(valueType);
    std::string paddset = "perftest_addvector.bin." + typeStr + "_" + std::to_string(insertVectorCount) + "_" + std::to_string(dimension);
    std::string paddmeta = "perftest_addmeta.bin." + std::to_string(baseVectorCount) + "_" + std::to_string(insertVectorCount);
    std::string paddmetaidx = "perftest_addmetaidx.bin." + std::to_string(baseVectorCount) + "_" + std::to_string(insertVectorCount);

    // Load query set
    int searchK = topK;
    // Optional cross-validation dump path (Benchmark/SearchResult); re-read
    // from config since RunWorker isn't handed the IniReader.  Each Search
    // round this worker dumps its contiguous query slice to <path>.node<idx>.
    std::string searchResultPath;
    if (const char* cfgPath = std::getenv("BENCHMARK_CONFIG")) {
        Helper::IniReader srIni;
        if (srIni.LoadIniFile(cfgPath) == ErrorCode::Success) {
            searchResultPath = srIni.GetParameter("Benchmark", "SearchResult", std::string(""));
        }
    }
    std::string pqueryset = "perftest_query.bin." + typeStr + "_" + std::to_string(numQueries) + "_" + std::to_string(dimension);
    auto queryset = TestUtils::TestDataGenerator<T>::LoadVectorSet(pqueryset, dimension);
    BOOST_REQUIRE_MESSAGE(queryset != nullptr, "Worker: Failed to load query set from " << pqueryset);

    // Register dispatch callback
    std::promise<void> stopPromise;
    auto stopFuture = stopPromise.get_future();
    std::once_flag stopOnce;

    router->SetDispatchCallback([&](const SPANN::DispatchCommand& cmd) -> SPANN::DispatchResult {
        SPANN::DispatchResult result;
        result.m_dispatchId = cmd.m_dispatchId;
        result.m_round = cmd.m_round;

        if (cmd.m_type == SPANN::DispatchCommand::Type::Stop) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Worker %d: Stop command received\n", nodeIndex);
            std::call_once(stopOnce, [&]() { stopPromise.set_value(); });
            result.m_status = SPANN::DispatchResult::Status::Success;
            return result;
        }

        if (cmd.m_type == SPANN::DispatchCommand::Type::Heartbeat) {
            // Driver sends a Heartbeat every HeartbeatIntervalSec; the result
            // is dropped by DispatchCoordinator. Acknowledge silently so we
            // don't log noise every 30s during the insert phase.
            result.m_status = SPANN::DispatchResult::Status::Success;
            return result;
        }

        if (cmd.m_type == SPANN::DispatchCommand::Type::Search) {
            int myStart = (int)((long long)nodeIndex * numQueries / numNodes);
            int myEnd = (int)((long long)(nodeIndex + 1) * numQueries / numNodes);
            int myCount = myEnd - myStart;

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Worker %d: Search round %u - %d queries [%d, %d)\n",
                         nodeIndex, cmd.m_round, myCount, myStart, myEnd);

            std::vector<QueryResult> results;
            double wallTime = ExecutePartitionedSearch<T>(
                index.get(), queryset, myStart, myCount, searchK,
                std::min(numSearchThreads, myCount),
                results, /*latenciesOut=*/nullptr, /*statsOut=*/nullptr);

            // Cross-validation dump: this worker's contiguous query slice.
            // Overwritten each round, so the file reflects the final index
            // state (last search round before Stop), aligned with the driver.
            if (!searchResultPath.empty()) {
                DumpSearchResultFile(searchResultPath + ".node" + std::to_string(nodeIndex),
                                     results, myCount, topK, searchK);
            }

            // Drain merge hints accumulated during this search round.
            // Search-side AsyncMergeInSearch on remote-owned heads enqueues
            // notifications via QueueRemoteMerge; auto-flush only fires when
            // a per-target bucket reaches kMergeAutoFlushThreshold, so the
            // tail of every round (and any sparse rounds) needs an explicit
            // drain to guarantee no hint is dropped.
            router->FlushRemoteMerges();

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Worker %d: Search round %u done - %.1fms\n",
                         nodeIndex, cmd.m_round, wallTime * 1000);
            result.m_status = SPANN::DispatchResult::Status::Success;
            result.m_wallTime = wallTime;
            return result;
        }

        if (cmd.m_type == SPANN::DispatchCommand::Type::Insert) {
            int insertStart = cmd.m_round * insertBatchSize + myInsertStart;
            int loadCount = perNodeBatch;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Worker %d: Batch %u - inserting %d vectors (offset %d)\n",
                         nodeIndex, cmd.m_round + 1, perNodeBatch, insertStart);

            auto t1 = std::chrono::high_resolution_clock::now();
            std::string workerTag =
                "Worker " + std::to_string(nodeIndex) + " batch=" + std::to_string(cmd.m_round + 1);
            LoadAndInsertBatch<T>(spannIndex, paddset, paddmeta, paddmetaidx, dimension,
                                  insertStart, loadCount, perNodeBatch,
                                  numInsertThreads, router,
                                  /*quantizer=*/nullptr,
                                  /*searchDuringInsertThreads=*/0,
                                  /*queryset=*/nullptr,
                                  /*numQueries=*/0, /*searchK=*/5,
                                  /*benchmarkData=*/nullptr,
                                  workerTag.c_str());
            auto t2 = std::chrono::high_resolution_clock::now();
            double secs = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Worker %d: Batch %u done - %d vectors in %.2f s (%.1f vec/s)\n",
                         nodeIndex, cmd.m_round + 1, perNodeBatch, secs, perNodeBatch / secs);

            result.m_status = SPANN::DispatchResult::Status::Success;
            result.m_wallTime = secs;
            return result;
        }

        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "Worker %d: Unknown command type %d\n",
                     nodeIndex, (int)cmd.m_type);
        result.m_status = SPANN::DispatchResult::Status::Failed;
        result.m_errorCode = static_cast<std::int32_t>(SPTAG::ErrorCode::Undefined);
        return result;
    });

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Worker %d: Waiting for dispatch commands\n", nodeIndex);

    auto status = stopFuture.wait_for(std::chrono::seconds(workerTimeout));
    if (status == std::future_status::timeout) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "Worker %d: Timeout after %ds\n", nodeIndex, workerTimeout);
    }

    router->ClearDispatchCallback();
    N = oldN; M = oldM; K = oldK; queries = oldQ;
    BOOST_TEST_MESSAGE("Worker " << nodeIndex << ": Shutting down");
}
BOOST_AUTO_TEST_SUITE_END()
