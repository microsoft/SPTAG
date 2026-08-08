// Native BKT-head diagnostic for comparing static SPANN head graph snapshots.

#include "inc/Core/BKT/Index.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/SimpleIniReader.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sys/stat.h>

using namespace SPTAG;

namespace {

constexpr int kRecall50 = 50;
constexpr int kRecall104 = 104;

struct Options
{
    std::string rootIni;
    std::string newHeadIndex;
    std::string oldHeadIndex;
    std::string queryFile;
    std::string outputFile;
    std::string exactHeadsOutputFile;
    std::size_t offset = 20;
    std::size_t count = 20;
    std::size_t warmupCount = 20;
    std::size_t exactThreads = 0;
    int maxK = kRecall104;
};

struct ExactHeadsFileHeader
{
    std::uint64_t magic;
    std::uint32_t version;
    std::uint32_t headerBytes;
    std::uint64_t queryOffset;
    std::uint64_t queryCount;
    std::uint32_t headK;
    std::uint32_t idBytes;
};

constexpr std::uint64_t kExactHeadsMagic = 0x3153444145485053ULL; // "SPHEADS1", little-endian.
constexpr std::uint32_t kExactHeadsVersion = 1;
static_assert(sizeof(ExactHeadsFileHeader) == 40, "Unexpected exact-head file header layout");

struct QuerySlice
{
    std::size_t totalCount = 0;
    std::size_t dimension = 0;
    std::vector<std::uint8_t> warmup;
    std::vector<std::uint8_t> measured;
};

struct Candidate
{
    SizeType id = -1;
    std::uint32_t distance = 0;
};

struct CandidateBetter
{
    bool operator()(const Candidate& p_left, const Candidate& p_right) const
    {
        return p_left.distance < p_right.distance ||
            (p_left.distance == p_right.distance && p_left.id < p_right.id);
    }
};

struct GraphMetrics
{
    std::string name;
    std::size_t failedQueries = 0;
    std::uint64_t candidatesFound = 0;
    std::uint64_t checkedLeaves = 0;
    std::uint64_t overlap50 = 0;
    std::uint64_t overlap104 = 0;
    std::uint64_t overlapResultMaxK = 0;
};

void Usage(const char* p_program)
{
    std::cerr << "Usage: " << p_program
              << " --root-ini <tenant/indexloader.ini>"
              << " --new-head <HeadIndex>"
              << " --old-head <HeadIndex.headrebuild.stage.N>"
              << " --query <query.u8bin>"
              << " [--offset 20] [--count 20] [--warmup-count 20]"
              << " [--exact-threads N] [--max-k N] [--output result.jsonl]"
              << " [--exact-heads-output exact_heads.bin]\n";
}

bool ParseSize(const char* p_text, std::size_t& p_value)
{
    if (p_text == nullptr || *p_text == '\0') return false;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(p_text, &end, 10);
    if (end == p_text || *end != '\0' ||
        value > static_cast<unsigned long long>((std::numeric_limits<std::size_t>::max)())) {
        return false;
    }
    p_value = static_cast<std::size_t>(value);
    return true;
}

bool ParseArgs(int p_argc, char** p_argv, Options& p_options)
{
    for (int i = 1; i < p_argc; ++i) {
        const char* arg = p_argv[i];
        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            Usage(p_argv[0]);
            std::exit(0);
        }
        if (i + 1 >= p_argc) return false;

        const char* value = p_argv[++i];
        if (std::strcmp(arg, "--root-ini") == 0) {
            p_options.rootIni = value;
        } else if (std::strcmp(arg, "--new-head") == 0) {
            p_options.newHeadIndex = value;
        } else if (std::strcmp(arg, "--old-head") == 0) {
            p_options.oldHeadIndex = value;
        } else if (std::strcmp(arg, "--query") == 0) {
            p_options.queryFile = value;
        } else if (std::strcmp(arg, "--output") == 0) {
            p_options.outputFile = value;
        } else if (std::strcmp(arg, "--exact-heads-output") == 0) {
            p_options.exactHeadsOutputFile = value;
        } else if (std::strcmp(arg, "--offset") == 0) {
            if (!ParseSize(value, p_options.offset)) return false;
        } else if (std::strcmp(arg, "--count") == 0) {
            if (!ParseSize(value, p_options.count) || p_options.count == 0) return false;
        } else if (std::strcmp(arg, "--warmup-count") == 0) {
            if (!ParseSize(value, p_options.warmupCount)) return false;
        } else if (std::strcmp(arg, "--exact-threads") == 0) {
            if (!ParseSize(value, p_options.exactThreads) || p_options.exactThreads == 0) return false;
        } else if (std::strcmp(arg, "--max-k") == 0) {
            std::size_t maxK = 0;
            if (!ParseSize(value, maxK) ||
                maxK > static_cast<std::size_t>((std::numeric_limits<int>::max)())) {
                return false;
            }
            p_options.maxK = static_cast<int>(maxK);
        } else {
            return false;
        }
    }

    return !p_options.rootIni.empty() && !p_options.newHeadIndex.empty() &&
        !p_options.oldHeadIndex.empty() && !p_options.queryFile.empty() &&
        p_options.maxK >= kRecall50 && p_options.warmupCount <= p_options.offset;
}

bool IsUInt8L2SpannConfig(const Helper::IniReader& p_config)
{
    return p_config.GetParameter("Index", "IndexAlgoType", std::string()) == "SPANN" &&
        p_config.GetParameter("Index", "ValueType", std::string()) == "UInt8" &&
        p_config.GetParameter("Base", "ValueType", std::string()) == "UInt8" &&
        p_config.GetParameter("Base", "DistCalcMethod", std::string()) == "L2";
}

bool LoadRuntimeConfig(const Options& p_options, DimensionType& p_dimension, int& p_maxCheck)
{
    Helper::IniReader config;
    if (config.LoadIniFile(p_options.rootIni) != ErrorCode::Success) {
        std::cerr << "Cannot load root native INI: " << p_options.rootIni << "\n";
        return false;
    }
    if (!IsUInt8L2SpannConfig(config)) {
        std::cerr << "Root native INI must describe a UInt8 L2 SPANN index\n";
        return false;
    }

    const int dimension = config.GetParameter<int>("Base", "Dim", -1);
    p_maxCheck = config.GetParameter<int>("SearchSSDIndex", "MaxCheck", -1);
    if (dimension <= 0 || p_maxCheck <= 0) {
        std::cerr << "Root native INI needs positive [Base] Dim and [SearchSSDIndex] MaxCheck\n";
        return false;
    }
    p_dimension = static_cast<DimensionType>(dimension);
    return true;
}

bool ReadQuerySlice(const Options& p_options, DimensionType p_expectedDimension, QuerySlice& p_queries)
{
    std::ifstream input(p_options.queryFile, std::ios::binary | std::ios::ate);
    if (!input) {
        std::cerr << "Cannot open query file: " << p_options.queryFile << "\n";
        return false;
    }

    const std::streamoff fileSize = input.tellg();
    input.seekg(0);
    std::int32_t rawCount = 0;
    std::int32_t rawDimension = 0;
    if (!input.read(reinterpret_cast<char*>(&rawCount), sizeof(rawCount)) ||
        !input.read(reinterpret_cast<char*>(&rawDimension), sizeof(rawDimension)) ||
        rawCount <= 0 || rawDimension <= 0 || rawDimension != p_expectedDimension) {
        std::cerr << "Invalid UInt8 query header or dimension mismatch\n";
        return false;
    }

    p_queries.totalCount = static_cast<std::size_t>(rawCount);
    p_queries.dimension = static_cast<std::size_t>(rawDimension);
    if (p_queries.totalCount > (std::numeric_limits<std::size_t>::max)() / p_queries.dimension ||
        p_queries.totalCount * p_queries.dimension >
            (std::numeric_limits<std::size_t>::max)() - sizeof(rawCount) - sizeof(rawDimension) ||
        p_options.offset > p_queries.totalCount ||
        p_options.count > p_queries.totalCount - p_options.offset) {
        std::cerr << "Query slice is outside the UInt8 query dataset\n";
        return false;
    }

    const std::size_t expectedSize = sizeof(rawCount) + sizeof(rawDimension) +
        p_queries.totalCount * p_queries.dimension;
    if (fileSize < 0 || static_cast<std::uintmax_t>(fileSize) != expectedSize) {
        std::cerr << "Unexpected size for UInt8 query dataset\n";
        return false;
    }

    p_queries.warmup.resize(p_options.warmupCount * p_queries.dimension);
    p_queries.measured.resize(p_options.count * p_queries.dimension);
    if (!p_queries.warmup.empty() &&
        !input.read(reinterpret_cast<char*>(p_queries.warmup.data()),
                    static_cast<std::streamsize>(p_queries.warmup.size()))) {
        std::cerr << "Cannot read warmup queries\n";
        return false;
    }

    input.seekg(static_cast<std::streamoff>(sizeof(rawCount) + sizeof(rawDimension) +
        p_options.offset * p_queries.dimension));
    if (!input.read(reinterpret_cast<char*>(p_queries.measured.data()),
                    static_cast<std::streamsize>(p_queries.measured.size()))) {
        std::cerr << "Cannot read measured queries\n";
        return false;
    }
    return true;
}

bool SameVectorFile(const std::string& p_newHead, const std::string& p_oldHead)
{
    struct stat newFile {};
    struct stat oldFile {};
    const std::string newPath = p_newHead + "/vectors.bin";
    const std::string oldPath = p_oldHead + "/vectors.bin";
    if (stat(newPath.c_str(), &newFile) != 0 || stat(oldPath.c_str(), &oldFile) != 0) {
        std::cerr << "Cannot stat head vectors.bin files\n";
        return false;
    }
    if (newFile.st_dev != oldFile.st_dev || newFile.st_ino != oldFile.st_ino) {
        std::cerr << "Head vectors.bin files are not the same hard-linked file\n";
        return false;
    }
    return true;
}

using UInt8BKT = BKT::Index<std::uint8_t>;

bool LoadHeadIndex(const std::string& p_path,
                   DimensionType p_dimension,
                   int p_runtimeMaxCheck,
                   std::shared_ptr<UInt8BKT>& p_index)
{
    std::shared_ptr<VectorIndex> loaded;
    if (VectorIndex::LoadIndex(p_path, loaded) != ErrorCode::Success || loaded == nullptr) {
        std::cerr << "Cannot load BKT head index: " << p_path << "\n";
        return false;
    }
    p_index = std::dynamic_pointer_cast<UInt8BKT>(loaded);
    if (p_index == nullptr || p_index->GetIndexAlgoType() != IndexAlgoType::BKT ||
        p_index->GetVectorValueType() != VectorValueType::UInt8 ||
        p_index->GetDistCalcMethod() != DistCalcMethod::L2 ||
        p_index->GetFeatureDim() != p_dimension || p_index->GetNumSamples() <= 0) {
        std::cerr << "Head index is not a nonempty UInt8 L2 BKT graph matching the root INI\n";
        return false;
    }

    const std::string maxCheck = std::to_string(p_runtimeMaxCheck);
    if (p_index->SetParameter("MaxCheck", maxCheck.c_str()) != ErrorCode::Success ||
        p_index->GetCurrMaxCheck() != p_runtimeMaxCheck) {
        std::cerr << "Cannot apply runtime [SearchSSDIndex] MaxCheck to head graph\n";
        return false;
    }
    return true;
}

std::uint32_t UInt8L2(const std::uint8_t* p_left, const std::uint8_t* p_right, std::size_t p_dimension)
{
    std::uint32_t distance = 0;
    for (std::size_t d = 0; d < p_dimension; ++d) {
        const int difference = static_cast<int>(p_left[d]) - static_cast<int>(p_right[d]);
        distance += static_cast<std::uint32_t>(difference * difference);
    }
    return distance;
}

std::vector<Candidate> ExactTopK(const UInt8BKT& p_index,
                                 const std::uint8_t* p_query,
                                 std::size_t p_dimension,
                                 int p_maxK,
                                 std::size_t p_threadCount)
{
    const SizeType sampleCount = p_index.GetNumSamples();
    p_threadCount = (std::min)(p_threadCount, static_cast<std::size_t>(sampleCount));
    std::vector<std::vector<Candidate>> partials(p_threadCount);
    std::vector<std::thread> workers;
    workers.reserve(p_threadCount);

    for (std::size_t worker = 0; worker < p_threadCount; ++worker) {
        const SizeType begin = static_cast<SizeType>(
            (static_cast<std::uint64_t>(sampleCount) * worker) / p_threadCount);
        const SizeType end = static_cast<SizeType>(
            (static_cast<std::uint64_t>(sampleCount) * (worker + 1)) / p_threadCount);
        workers.emplace_back([&, worker, begin, end] {
            std::priority_queue<Candidate, std::vector<Candidate>, CandidateBetter> best;
            for (SizeType id = begin; id < end; ++id) {
                const Candidate candidate = {
                    id,
                    UInt8L2(p_query, static_cast<const std::uint8_t*>(p_index.GetSample(id)), p_dimension)
                };
                if (static_cast<int>(best.size()) < p_maxK) {
                    best.push(candidate);
                } else if (CandidateBetter{}(candidate, best.top())) {
                    best.pop();
                    best.push(candidate);
                }
            }
            auto& output = partials[worker];
            output.reserve(best.size());
            while (!best.empty()) {
                output.push_back(best.top());
                best.pop();
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    std::priority_queue<Candidate, std::vector<Candidate>, CandidateBetter> best;
    for (const auto& partial : partials) {
        for (const Candidate& candidate : partial) {
            if (static_cast<int>(best.size()) < p_maxK) {
                best.push(candidate);
            } else if (CandidateBetter{}(candidate, best.top())) {
                best.pop();
                best.push(candidate);
            }
        }
    }

    std::vector<Candidate> output;
    output.reserve(best.size());
    while (!best.empty()) {
        output.push_back(best.top());
        best.pop();
    }
    std::sort(output.begin(), output.end(), CandidateBetter{});
    return output;
}

bool WriteExactHeads(const std::string& p_path,
                     const Options& p_options,
                     const std::vector<std::vector<Candidate>>& p_exact)
{
    if (p_exact.size() != p_options.count) {
        std::cerr << "Exact-head query count does not match the requested slice\n";
        return false;
    }

    const std::uint32_t headK = static_cast<std::uint32_t>(p_options.maxK);
    ExactHeadsFileHeader header = {
        kExactHeadsMagic,
        kExactHeadsVersion,
        static_cast<std::uint32_t>(sizeof(ExactHeadsFileHeader)),
        static_cast<std::uint64_t>(p_options.offset),
        static_cast<std::uint64_t>(p_exact.size()),
        headK,
        static_cast<std::uint32_t>(sizeof(std::int32_t))
    };

    std::ofstream output(p_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        std::cerr << "Cannot open exact-head output file: " << p_path << "\n";
        return false;
    }
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (const auto& query : p_exact) {
        if (query.size() != headK) {
            std::cerr << "Exact-head result has an unexpected K\n";
            return false;
        }
        for (const Candidate& candidate : query) {
            if (candidate.id < 0) {
                std::cerr << "Exact-head result contains an invalid local ID\n";
                return false;
            }
            const std::int32_t id = static_cast<std::int32_t>(candidate.id);
            output.write(reinterpret_cast<const char*>(&id), sizeof(id));
        }
    }
    output.close();
    if (!output) {
        std::cerr << "Cannot write exact-head output file: " << p_path << "\n";
        return false;
    }
    return true;
}

void Warmup(const UInt8BKT& p_index, const QuerySlice& p_queries, int p_maxK)
{
    for (std::size_t query = 0; query * p_queries.dimension < p_queries.warmup.size(); ++query) {
        QueryResult result(
            p_queries.warmup.data() + query * p_queries.dimension,
            p_maxK,
            false);
        p_index.SearchIndex(result);
    }
}

std::uint64_t OverlapAt(const std::vector<Candidate>& p_exact,
                        const std::vector<SizeType>& p_approximate,
                        int p_k)
{
    std::unordered_set<SizeType> expected;
    expected.reserve(static_cast<std::size_t>(p_k));
    for (int i = 0; i < p_k; ++i) {
        expected.insert(p_exact[static_cast<std::size_t>(i)].id);
    }

    std::unordered_set<SizeType> seen;
    seen.reserve(static_cast<std::size_t>(p_k));
    std::uint64_t hits = 0;
    for (int i = 0; i < p_k && i < static_cast<int>(p_approximate.size()); ++i) {
        const SizeType id = p_approximate[static_cast<std::size_t>(i)];
        if (id >= 0 && seen.insert(id).second && expected.find(id) != expected.end()) {
            ++hits;
        }
    }
    return hits;
}

GraphMetrics EvaluateGraph(const std::string& p_name,
                           const UInt8BKT& p_index,
                           const QuerySlice& p_queries,
                           const std::vector<std::vector<Candidate>>& p_exact,
                           int p_maxK)
{
    GraphMetrics metrics;
    metrics.name = p_name;
    for (std::size_t query = 0; query < p_exact.size(); ++query) {
        QueryResult result(
            p_queries.measured.data() + query * p_queries.dimension,
            p_maxK,
            false);
        if (p_index.SearchIndex(result) != ErrorCode::Success) {
            ++metrics.failedQueries;
            continue;
        }

        metrics.checkedLeaves += static_cast<std::uint64_t>(result.GetScanned());
        std::vector<SizeType> approximate;
        approximate.reserve(static_cast<std::size_t>(p_maxK));
        for (int i = 0; i < p_maxK; ++i) {
            const BasicResult* candidate = result.GetResult(i);
            if (candidate != nullptr && candidate->VID >= 0) {
                approximate.push_back(candidate->VID);
            }
        }
        metrics.candidatesFound += approximate.size();
        metrics.overlap50 += OverlapAt(p_exact[query], approximate, kRecall50);
        metrics.overlapResultMaxK += OverlapAt(p_exact[query], approximate, p_maxK);
        if (p_maxK >= kRecall104) {
            metrics.overlap104 += OverlapAt(p_exact[query], approximate, kRecall104);
        }
    }
    return metrics;
}

void WriteResult(std::ostream& p_output,
                 const GraphMetrics& p_metrics,
                 const Options& p_options,
                 std::size_t p_queryCount,
                 int p_runtimeMaxCheck,
                 std::size_t p_headCount,
                 DimensionType p_dimension,
                 double p_exactSeconds)
{
    const double queryCount = static_cast<double>(p_queryCount);
    p_output << "{"
             << "\"type\":\"head_graph_recall\","
             << "\"graph\":\"" << p_metrics.name << "\","
             << "\"query_offset\":" << p_options.offset << ","
             << "\"query_count\":" << p_queryCount << ","
             << "\"warmup_count\":" << p_options.warmupCount << ","
             << "\"head_count\":" << p_headCount << ","
             << "\"dimension\":" << p_dimension << ","
             << "\"value_type\":\"UInt8\","
             << "\"distance\":\"L2\","
             << "\"max_check_source\":\"root_ini.SearchSSDIndex.MaxCheck\","
             << "\"max_check\":" << p_runtimeMaxCheck << ","
             << "\"result_max_k\":" << p_options.maxK << ","
             << "\"exact_threads\":" << p_options.exactThreads << ","
             << "\"exact_scan_seconds\":" << p_exactSeconds << ","
             << "\"mean_candidates_found\":" << p_metrics.candidatesFound / queryCount << ","
             << "\"mean_checked_leaves\":" << p_metrics.checkedLeaves / queryCount << ","
             << "\"mean_exact_overlap_at_50\":" << p_metrics.overlap50 / queryCount << ","
             << "\"mean_exact_recall_at_50\":"
             << p_metrics.overlap50 / (queryCount * kRecall50) << ","
             << "\"mean_exact_overlap_at_result_max_k\":"
             << p_metrics.overlapResultMaxK / queryCount << ","
             << "\"mean_exact_recall_at_result_max_k\":"
             << p_metrics.overlapResultMaxK / (queryCount * p_options.maxK) << ",";
    if (p_options.maxK >= kRecall104) {
        p_output << "\"mean_exact_overlap_at_104\":" << p_metrics.overlap104 / queryCount << ","
                 << "\"mean_exact_recall_at_104\":"
                 << p_metrics.overlap104 / (queryCount * kRecall104) << ",";
    } else {
        p_output << "\"mean_exact_overlap_at_104\":null,"
                 << "\"mean_exact_recall_at_104\":null,";
    }
    p_output << "\"failed_queries\":" << p_metrics.failedQueries << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!ParseArgs(argc, argv, options)) {
        Usage(argv[0]);
        return 2;
    }

    DimensionType dimension = 0;
    int runtimeMaxCheck = 0;
    if (!LoadRuntimeConfig(options, dimension, runtimeMaxCheck) ||
        !SameVectorFile(options.newHeadIndex, options.oldHeadIndex)) {
        return 1;
    }

    QuerySlice queries;
    if (!ReadQuerySlice(options, dimension, queries)) {
        return 1;
    }

    if (options.exactThreads == 0) {
        const unsigned int hardwareThreads = std::thread::hardware_concurrency();
        options.exactThreads = (std::min)(std::size_t{8},
            hardwareThreads == 0 ? std::size_t{1} : static_cast<std::size_t>(hardwareThreads));
    }

    std::shared_ptr<UInt8BKT> newIndex;
    if (!LoadHeadIndex(options.newHeadIndex, dimension, runtimeMaxCheck, newIndex)) {
        return 1;
    }
    Warmup(*newIndex, queries, options.maxK);

    std::vector<std::vector<Candidate>> exact;
    exact.reserve(options.count);
    const auto exactStart = std::chrono::steady_clock::now();
    for (std::size_t query = 0; query < options.count; ++query) {
        exact.push_back(ExactTopK(
            *newIndex,
            queries.measured.data() + query * queries.dimension,
            queries.dimension,
            options.maxK,
            options.exactThreads));
        if (static_cast<int>(exact.back().size()) < options.maxK) {
            std::cerr << "Head index has fewer samples than --max-k\n";
            return 1;
        }
    }
    const double exactSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - exactStart).count();
    if (!options.exactHeadsOutputFile.empty() &&
        !WriteExactHeads(options.exactHeadsOutputFile, options, exact)) {
        return 1;
    }
    const std::size_t headCount = static_cast<std::size_t>(newIndex->GetNumSamples());
    const GraphMetrics newMetrics = EvaluateGraph(
        "new", *newIndex, queries, exact, options.maxK);
    newIndex.reset();

    std::shared_ptr<UInt8BKT> oldIndex;
    if (!LoadHeadIndex(options.oldHeadIndex, dimension, runtimeMaxCheck, oldIndex) ||
        oldIndex->GetNumSamples() != static_cast<SizeType>(headCount)) {
        std::cerr << "Retained head graph does not match the new head vector count\n";
        return 1;
    }
    Warmup(*oldIndex, queries, options.maxK);
    const GraphMetrics oldMetrics = EvaluateGraph(
        "old", *oldIndex, queries, exact, options.maxK);

    if (options.outputFile.empty()) {
        WriteResult(std::cout, newMetrics, options, options.count, runtimeMaxCheck, headCount, dimension, exactSeconds);
        WriteResult(std::cout, oldMetrics, options, options.count, runtimeMaxCheck, headCount, dimension, exactSeconds);
    } else {
        std::ofstream output(options.outputFile, std::ios::out | std::ios::trunc);
        if (!output) {
            std::cerr << "Cannot open output file: " << options.outputFile << "\n";
            return 1;
        }
        WriteResult(output, newMetrics, options, options.count, runtimeMaxCheck, headCount, dimension, exactSeconds);
        WriteResult(output, oldMetrics, options, options.count, runtimeMaxCheck, headCount, dimension, exactSeconds);
    }
    return 0;
}
