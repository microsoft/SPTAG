// Native concurrent benchmark for the tenant-aware SPANN unfilter path.
//
// This deliberately invokes TenantIndexManager::SearchWithACL rather than the
// generic AnnIndex batch API, so it measures the same unfilter-tail path as
// the canonical filtered-curve runner without Python's GIL.

#include "inc/CoreInterface.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace SPTAG;

namespace {

struct Options
{
    std::string indexDir;
    std::string queryFile;
    std::string truthFile;
    std::size_t maxQueries = 0;
    std::size_t warmupQueries = 200;
    std::size_t latencyLimitMs = 0;
    int tenant = 0;
    int topk = 100;
    std::vector<int> threadCounts = {1};
};

struct UInt8Dataset
{
    std::size_t count = 0;
    std::size_t dimension = 0;
    std::vector<std::uint8_t> values;
};

struct TruthSet
{
    std::size_t count = 0;
    std::size_t dimension = 0;
    std::vector<std::uint32_t> ids;
};

enum class Phase
{
    Ready,
    Warmup,
    Measure,
    Stop,
};

struct SharedState
{
    std::mutex mutex;
    std::condition_variable cv;
    Phase phase = Phase::Ready;
    int readyWorkers = 0;
    int completedWorkers = 0;
    std::atomic<std::size_t> nextQuery{0};
};

struct TrialResult
{
    int threads = 0;
    double elapsedSeconds = 0.0;
    double recallPercent = 0.0;
    double qps = 0.0;
    double averageLatencyUs = 0.0;
    double p50LatencyUs = 0.0;
    double p95LatencyUs = 0.0;
    double p99LatencyUs = 0.0;
    std::size_t failedQueries = 0;
    std::uint64_t postingPageReads = 0;
    std::uint64_t postingLogicalBytes = 0;
    std::uint64_t postingPhysicalBytes = 0;
    std::uint64_t adcScannedVectors = 0;
    std::uint64_t adcSurvivors = 0;
    std::uint64_t rerankCandidates = 0;
    std::uint64_t rerankReadRequests = 0;
    std::uint64_t rerankPhysicalBytes = 0;
};

struct SearchWorkTotals
{
    std::uint64_t postingPageReads = 0;
    std::uint64_t postingLogicalBytes = 0;
    std::uint64_t postingPhysicalBytes = 0;
    std::uint64_t adcScannedVectors = 0;
    std::uint64_t adcSurvivors = 0;
    std::uint64_t rerankCandidates = 0;
    std::uint64_t rerankReadRequests = 0;
    std::uint64_t rerankPhysicalBytes = 0;

    void Add(const VectorIndex::PostingScanStats& stats)
    {
        postingPageReads += stats.m_postingPageReads;
        postingLogicalBytes += stats.m_postingLogicalBytes;
        postingPhysicalBytes += stats.m_postingPhysicalBytes;
        adcScannedVectors += stats.m_adcScannedVectors;
        adcSurvivors += stats.m_adcSurvivors;
        rerankCandidates += stats.m_rerankCandidates;
        rerankReadRequests += stats.m_rerankReadRequests;
        rerankPhysicalBytes += stats.m_rerankPhysicalBytes;
    }
};

void Usage(const char* program)
{
    std::cerr
        << "Usage: " << program
        << " --index <index-dir> --query <query.u8bin> --truth <gt.ibin>"
        << " [--threads N | --thread-grid N1,N2,...] [--warmup N] [--max-queries N]"
        << " [--topk K] [--tenant ID] [--latency-limit-ms N]\n";
}

bool ParseUnsigned(const char* text, std::size_t& value)
{
    if (text == nullptr || *text == '\0') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0'
        || parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return true;
}

std::size_t ReadPositiveEnvironment(const char* name, std::size_t fallback)
{
    const char* value = std::getenv(name);
    std::size_t parsed = 0;
    return value != nullptr && ParseUnsigned(value, parsed) && parsed > 0 ? parsed : fallback;
}

bool ParseInt(const char* text, int& value)
{
    std::size_t parsed = 0;
    if (!ParseUnsigned(text, parsed)
        || parsed > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool ParseThreadGrid(const char* text, std::vector<int>& threadCounts)
{
    if (text == nullptr || *text == '\0') return false;

    std::vector<int> parsed;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        int threads = 0;
        if (!ParseInt(token.c_str(), threads) || threads <= 0
            || std::find(parsed.begin(), parsed.end(), threads) != parsed.end()) {
            return false;
        }
        parsed.push_back(threads);
    }
    if (parsed.empty()) return false;
    threadCounts = std::move(parsed);
    return true;
}

bool ParseArgs(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            Usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << arg << "\n";
            return false;
        }

        const char* value = argv[++i];
        if (std::strcmp(arg, "--index") == 0) {
            options.indexDir = value;
        } else if (std::strcmp(arg, "--query") == 0) {
            options.queryFile = value;
        } else if (std::strcmp(arg, "--truth") == 0) {
            options.truthFile = value;
        } else if (std::strcmp(arg, "--threads") == 0) {
            int threads = 0;
            if (!ParseInt(value, threads) || threads <= 0) return false;
            options.threadCounts = {threads};
        } else if (std::strcmp(arg, "--thread-grid") == 0) {
            if (!ParseThreadGrid(value, options.threadCounts)) return false;
        } else if (std::strcmp(arg, "--warmup") == 0) {
            if (!ParseUnsigned(value, options.warmupQueries)) return false;
        } else if (std::strcmp(arg, "--max-queries") == 0) {
            if (!ParseUnsigned(value, options.maxQueries)) return false;
        } else if (std::strcmp(arg, "--latency-limit-ms") == 0) {
            if (!ParseUnsigned(value, options.latencyLimitMs) || options.latencyLimitMs == 0) return false;
        } else if (std::strcmp(arg, "--topk") == 0) {
            if (!ParseInt(value, options.topk) || options.topk <= 0) return false;
        } else if (std::strcmp(arg, "--tenant") == 0) {
            if (!ParseInt(value, options.tenant) || options.tenant < 0) return false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    if (options.indexDir.empty() || options.queryFile.empty() || options.truthFile.empty()) {
        return false;
    }
    return true;
}

template <typename T>
bool ReadDataset(const std::string& path, std::vector<T>& values, std::size_t& count, std::size_t& dimension)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        std::cerr << "Cannot open " << path << "\n";
        return false;
    }

    const std::streamoff fileSize = input.tellg();
    input.seekg(0);
    std::int32_t rawCount = 0;
    std::int32_t rawDimension = 0;
    if (!input.read(reinterpret_cast<char*>(&rawCount), sizeof(rawCount))
        || !input.read(reinterpret_cast<char*>(&rawDimension), sizeof(rawDimension))
        || rawCount <= 0 || rawDimension <= 0) {
        std::cerr << "Invalid dataset header in " << path << "\n";
        return false;
    }

    count = static_cast<std::size_t>(rawCount);
    dimension = static_cast<std::size_t>(rawDimension);
    if (count > std::numeric_limits<std::size_t>::max() / dimension
        || count * dimension > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        std::cerr << "Dataset dimensions overflow in " << path << "\n";
        return false;
    }

    const std::size_t valueCount = count * dimension;
    const std::size_t expectedSize = sizeof(rawCount) + sizeof(rawDimension) + valueCount * sizeof(T);
    if (fileSize < 0 || static_cast<std::uintmax_t>(fileSize) != expectedSize) {
        std::cerr << "Unexpected size for " << path << ": got " << fileSize
                  << ", expected " << expectedSize << "\n";
        return false;
    }

    values.resize(valueCount);
    if (!input.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(valueCount * sizeof(T)))) {
        std::cerr << "Failed to read values from " << path << "\n";
        return false;
    }
    return true;
}

bool ReadQueries(const std::string& path, UInt8Dataset& dataset)
{
    return ReadDataset(path, dataset.values, dataset.count, dataset.dimension);
}

bool ReadTruth(const std::string& path, TruthSet& truth)
{
    return ReadDataset(path, truth.ids, truth.count, truth.dimension);
}

double Percentile(std::vector<double> values, double fraction)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t index = std::min(
        values.size() - 1,
        static_cast<std::size_t>(fraction * static_cast<double>(values.size())));
    return values[index];
}

std::size_t CalculateRecallHits(const TruthSet& truth,
                                const std::vector<std::int32_t>& resultIds,
                                std::size_t warmupQueries,
                                std::size_t measuredQueries,
                                int topk)
{
    std::size_t hits = 0;
    for (std::size_t measuredIndex = 0; measuredIndex < measuredQueries; ++measuredIndex) {
        const std::size_t queryIndex = warmupQueries + measuredIndex;
        const std::uint32_t* truthRow = truth.ids.data() + queryIndex * truth.dimension;
        const std::int32_t* resultRow = resultIds.data() + measuredIndex * static_cast<std::size_t>(topk);

        std::unordered_set<std::uint32_t> expected;
        const std::size_t expectedCount = std::min<std::size_t>(topk, truth.dimension);
        expected.reserve(expectedCount);
        for (std::size_t i = 0; i < expectedCount; ++i) {
            expected.insert(truthRow[i]);
        }

        std::unordered_set<std::uint32_t> seen;
        seen.reserve(static_cast<std::size_t>(topk));
        for (int i = 0; i < topk; ++i) {
            const std::int32_t id = resultRow[i];
            if (id >= 0 && seen.insert(static_cast<std::uint32_t>(id)).second
                && expected.find(static_cast<std::uint32_t>(id)) != expected.end()) {
                ++hits;
            }
        }
    }
    return hits;
}

int ReadNprobe()
{
    const char* value = std::getenv("SPTAG_FIXED_NPROBE");
    int nprobe = 0;
    return ParseInt(value, nprobe) ? nprobe : 0;
}

TrialResult RunConcurrentTrial(TenantIndexManager& manager,
                               const UInt8Dataset& queries,
                               const TruthSet& truth,
                               const Options& options,
                               std::size_t totalQueries,
                               std::uint8_t* emptyTag,
                               std::size_t serialWarmupQueries,
                               int threadCount)
{
    const std::size_t measuredQueries = totalQueries - options.warmupQueries;
    SharedState shared;
    std::vector<double> latenciesUs(measuredQueries, 0.0);
    std::vector<std::int32_t> resultIds(
        measuredQueries * static_cast<std::size_t>(options.topk),
        -1);
    std::vector<std::size_t> nullResults(static_cast<std::size_t>(threadCount), 0);
    std::vector<SearchWorkTotals> workTotals(static_cast<std::size_t>(threadCount));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threadCount));

    auto runRange = [&](std::size_t begin, std::size_t end, bool record, int workerId) {
        while (true) {
            const std::size_t queryIndex = shared.nextQuery.fetch_add(1, std::memory_order_relaxed);
            if (queryIndex >= end) return;

            const auto start = std::chrono::steady_clock::now();
            const auto result = manager.SearchWithACL(
                ByteArray(const_cast<std::uint8_t*>(
                              queries.values.data() + queryIndex * queries.dimension),
                          queries.dimension,
                          false),
                options.tenant,
                options.topk,
                ByteArray(emptyTag, 0, false),
                0);
            const auto finish = std::chrono::steady_clock::now();

            if (!record) continue;

            const std::size_t measuredIndex = queryIndex - begin;
            workTotals[static_cast<std::size_t>(workerId)].Add(
                VectorIndex::GetThreadLocalPostingScanStats());
            latenciesUs[measuredIndex] =
                std::chrono::duration<double, std::micro>(finish - start).count();
            std::int32_t* output =
                resultIds.data() + measuredIndex * static_cast<std::size_t>(options.topk);
            if (result == nullptr) {
                ++nullResults[static_cast<std::size_t>(workerId)];
                continue;
            }
            const int resultCount = std::min(result->GetResultNum(), options.topk);
            for (int i = 0; i < resultCount; ++i) {
                const auto* item = result->GetResult(i);
                if (item != nullptr && item->VID >= 0
                    && item->VID <= static_cast<SizeType>(std::numeric_limits<std::int32_t>::max())) {
                    output[i] = static_cast<std::int32_t>(item->VID);
                }
            }
        }
    };

    for (int workerId = 0; workerId < threadCount; ++workerId) {
        workers.emplace_back([&, workerId] {
            {
                std::unique_lock<std::mutex> lock(shared.mutex);
                ++shared.readyWorkers;
                shared.cv.notify_all();
                shared.cv.wait(lock, [&] { return shared.phase != Phase::Ready; });
                if (shared.phase == Phase::Stop) return;
            }

            runRange(serialWarmupQueries, options.warmupQueries, false, workerId);

            {
                std::unique_lock<std::mutex> lock(shared.mutex);
                ++shared.completedWorkers;
                shared.cv.notify_all();
                shared.cv.wait(lock, [&] { return shared.phase != Phase::Warmup; });
                if (shared.phase == Phase::Stop) return;
            }

            runRange(options.warmupQueries, totalQueries, true, workerId);

            {
                std::unique_lock<std::mutex> lock(shared.mutex);
                ++shared.completedWorkers;
                shared.cv.notify_all();
                shared.cv.wait(lock, [&] { return shared.phase == Phase::Stop; });
            }
        });
    }

    double elapsedSeconds = 0.0;
    {
        std::unique_lock<std::mutex> lock(shared.mutex);
        shared.cv.wait(lock, [&] { return shared.readyWorkers == threadCount; });
        shared.nextQuery.store(serialWarmupQueries, std::memory_order_relaxed);
        shared.phase = Phase::Warmup;
        shared.cv.notify_all();
        shared.cv.wait(lock, [&] { return shared.completedWorkers == threadCount; });

        shared.completedWorkers = 0;
        shared.nextQuery.store(options.warmupQueries, std::memory_order_relaxed);
        shared.phase = Phase::Measure;
        const auto start = std::chrono::steady_clock::now();
        shared.cv.notify_all();
        shared.cv.wait(lock, [&] { return shared.completedWorkers == threadCount; });
        const auto finish = std::chrono::steady_clock::now();
        elapsedSeconds = std::chrono::duration<double>(finish - start).count();
        shared.phase = Phase::Stop;
        shared.cv.notify_all();
    }

    for (auto& worker : workers) {
        worker.join();
    }

    const std::size_t recallHits = CalculateRecallHits(
        truth, resultIds, options.warmupQueries, measuredQueries, options.topk);
    const std::size_t recallDenominator =
        measuredQueries * static_cast<std::size_t>(options.topk);
    TrialResult output;
    output.threads = threadCount;
    output.elapsedSeconds = elapsedSeconds;
    output.recallPercent =
        100.0 * static_cast<double>(recallHits) / static_cast<double>(recallDenominator);
    output.qps = static_cast<double>(measuredQueries) / elapsedSeconds;
    output.averageLatencyUs = std::accumulate(
        latenciesUs.begin(), latenciesUs.end(), 0.0) / static_cast<double>(measuredQueries);
    output.p50LatencyUs = Percentile(latenciesUs, 0.50);
    output.p95LatencyUs = Percentile(latenciesUs, 0.95);
    output.p99LatencyUs = Percentile(latenciesUs, 0.99);
    output.failedQueries =
        std::accumulate(nullResults.begin(), nullResults.end(), std::size_t{0});
    for (const auto& work : workTotals) {
        output.postingPageReads += work.postingPageReads;
        output.postingLogicalBytes += work.postingLogicalBytes;
        output.postingPhysicalBytes += work.postingPhysicalBytes;
        output.adcScannedVectors += work.adcScannedVectors;
        output.adcSurvivors += work.adcSurvivors;
        output.rerankCandidates += work.rerankCandidates;
        output.rerankReadRequests += work.rerankReadRequests;
        output.rerankPhysicalBytes += work.rerankPhysicalBytes;
    }
    return output;
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!ParseArgs(argc, argv, options)) {
        Usage(argv[0]);
        return 2;
    }

    UInt8Dataset queries;
    TruthSet truth;
    if (!ReadQueries(options.queryFile, queries) || !ReadTruth(options.truthFile, truth)) {
        return 1;
    }
    if (queries.count != truth.count || queries.dimension == 0) {
        std::cerr << "Query/truth count mismatch or invalid query dimension\n";
        return 1;
    }
    if (truth.dimension < static_cast<std::size_t>(options.topk)) {
        std::cerr << "Truth width is smaller than --topk\n";
        return 1;
    }

    const std::size_t totalQueries = options.maxQueries == 0
        ? queries.count
        : std::min(options.maxQueries, queries.count);
    if (options.warmupQueries == 0 || options.warmupQueries >= totalQueries) {
        std::cerr << "--warmup must be positive and smaller than the selected query count\n";
        return 1;
    }
    const std::size_t measuredQueries = totalQueries - options.warmupQueries;
    const std::size_t sharedAioContexts =
        ReadPositiveEnvironment("SPTAG_SHARED_AIO_CONTEXTS", 4);
    const std::size_t sharedAioEvents =
        ReadPositiveEnvironment("SPTAG_SHARED_AIO_EVENTS", 1024);

    TenantIndexManager manager(
        static_cast<DimensionType>(queries.dimension),
        "SPANN",
        "UInt8");
    if (!manager.LoadAll(options.indexDir.c_str())) {
        std::cerr << "LoadAll failed: " << options.indexDir << "\n";
        return 1;
    }
    if (options.latencyLimitMs > 0) {
        const std::string latencyLimit = std::to_string(options.latencyLimitMs);
        manager.SetSearchParam("LatencyLimit", latencyLimit.c_str(), "BuildSSDIndex");
    }

    std::uint8_t emptyTag = 0;

    for (std::size_t trialOrder = 0; trialOrder < options.threadCounts.size(); ++trialOrder) {
        // The first warmup query initializes the lazy search structures before
        // workers concurrently enter the loaded manager. It counts toward each
        // trial's advertised warmup total.
        const auto preflight = manager.SearchWithACL(
            ByteArray(queries.values.data(), queries.dimension, false),
            options.tenant,
            options.topk,
            ByteArray(&emptyTag, 0, false),
            0);
        if (preflight == nullptr) {
            std::cerr << "Preflight SearchWithACL failed\n";
            return 1;
        }

        const TrialResult result = RunConcurrentTrial(
            manager,
            queries,
            truth,
            options,
            totalQueries,
            &emptyTag,
            1,
            options.threadCounts[trialOrder]);
        const auto perMeasuredQuery = [&](std::uint64_t value) {
            return static_cast<double>(value) / static_cast<double>(measuredQueries);
        };
        std::cout << std::fixed << std::setprecision(6)
                  << "RESULT {"
                  << "\"engine\":\"spann\","
                  << "\"path\":\"SearchWithACL-unfilter\","
                  << "\"nprobe\":" << ReadNprobe() << ","
                  << "\"threads\":" << result.threads << ","
                  << "\"trial_order\":" << trialOrder << ","
                  << "\"manager_reused\":true,"
                  << "\"warmup_queries\":" << options.warmupQueries << ","
                  << "\"serial_warmup_queries\":1,"
                  << "\"latency_limit_override_ms\":" << options.latencyLimitMs << ","
                  << "\"shared_aio_contexts\":" << sharedAioContexts << ","
                  << "\"shared_aio_events\":" << sharedAioEvents << ","
                  << "\"measured_queries\":" << measuredQueries << ","
                  << "\"topk\":" << options.topk << ","
                  << "\"recall_percent\":" << result.recallPercent << ","
                  << "\"qps\":" << result.qps << ","
                  << "\"avg_latency_us\":" << result.averageLatencyUs << ","
                  << "\"p50_latency_us\":" << result.p50LatencyUs << ","
                  << "\"p95_latency_us\":" << result.p95LatencyUs << ","
                  << "\"p99_latency_us\":" << result.p99LatencyUs << ","
                  << "\"failed_queries\":" << result.failedQueries << ","
                  << "\"posting_page_reads_per_query\":"
                  << perMeasuredQuery(result.postingPageReads) << ","
                  << "\"posting_logical_bytes_per_query\":"
                  << perMeasuredQuery(result.postingLogicalBytes) << ","
                  << "\"posting_physical_bytes_per_query\":"
                  << perMeasuredQuery(result.postingPhysicalBytes) << ","
                  << "\"adc_scanned_vectors_per_query\":"
                  << perMeasuredQuery(result.adcScannedVectors) << ","
                  << "\"adc_survivors_per_query\":"
                  << perMeasuredQuery(result.adcSurvivors) << ","
                  << "\"rerank_candidates_per_query\":"
                  << perMeasuredQuery(result.rerankCandidates) << ","
                  << "\"rerank_read_requests_per_query\":"
                  << perMeasuredQuery(result.rerankReadRequests) << ","
                  << "\"rerank_physical_bytes_per_query\":"
                  << perMeasuredQuery(result.rerankPhysicalBytes)
                  << "}" << std::endl;
    }
    return 0;
}
