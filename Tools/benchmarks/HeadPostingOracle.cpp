// Native exact-head production-scan oracle for static SPANN indexes.

#include "inc/Core/VectorIndex.h"
#include "inc/Helper/SimpleIniReader.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace SPTAG;

namespace {

constexpr int kResultK = 10;
constexpr std::uint64_t kExactHeadsMagic = 0x3153444145485053ULL; // "SPHEADS1", little-endian.
constexpr std::uint32_t kExactHeadsVersion = 1;

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

static_assert(sizeof(ExactHeadsFileHeader) == 40, "Unexpected exact-head file header layout");

struct Options
{
    std::string indexPath;
    std::string queryFile;
    std::string groundTruthFile;
    std::string exactHeadsFile;
    std::string outputFile;
    std::size_t offset = 0;
    std::size_t count = 0;
    std::size_t headK = 0;
    bool hasOffset = false;
    bool hasCount = false;
    bool compareNormal = false;
};

struct RuntimeConfig
{
    std::string indexDirectory;
    DimensionType dimension = 0;
    int postingOffset = 0;
    int postingPageLimit = 0;
    int internalResultNum = 0;
    std::string postingPageLimitKey;
};

struct QuerySlice
{
    std::size_t totalCount = 0;
    std::size_t dimension = 0;
    std::vector<std::uint8_t> values;
};

struct ExactHeadSlice
{
    std::size_t offset = 0;
    std::size_t count = 0;
    std::size_t storedK = 0;
    std::vector<std::vector<SizeType>> ids;
};

struct WorkStats
{
    std::uint64_t readPostings = 0;
    std::uint64_t matchedPostings = 0;
    std::uint64_t prePSPostings = 0;
    std::uint64_t scannedVectors = 0;
    std::uint64_t matchedVectors = 0;
    std::uint64_t uniqueMatchedPostings = 0;
    std::uint64_t uniqueMatchedVectors = 0;
    std::uint64_t primaryHeadCandidates = 0;
    std::uint64_t postingPageReads = 0;
    std::uint64_t postingLogicalBytes = 0;
    std::uint64_t postingPhysicalBytes = 0;
    std::uint64_t adcScannedVectors = 0;
    std::uint64_t adcSurvivors = 0;
    std::uint64_t rerankCandidates = 0;
    std::uint64_t rerankReadRequests = 0;
    std::uint64_t rerankPhysicalBytes = 0;

    void Add(const VectorIndex::PostingScanStats& p_stats)
    {
        readPostings += p_stats.m_readPostings;
        matchedPostings += p_stats.m_matchedPostings;
        prePSPostings += p_stats.m_prePSPostings;
        scannedVectors += p_stats.m_scannedVectors;
        matchedVectors += p_stats.m_matchedVectors;
        uniqueMatchedPostings += p_stats.m_uniqueMatchedPostings;
        uniqueMatchedVectors += p_stats.m_uniqueMatchedVectors;
        primaryHeadCandidates += p_stats.m_primaryHeadCandidates;
        postingPageReads += p_stats.m_postingPageReads;
        postingLogicalBytes += p_stats.m_postingLogicalBytes;
        postingPhysicalBytes += p_stats.m_postingPhysicalBytes;
        adcScannedVectors += p_stats.m_adcScannedVectors;
        adcSurvivors += p_stats.m_adcSurvivors;
        rerankCandidates += p_stats.m_rerankCandidates;
        rerankReadRequests += p_stats.m_rerankReadRequests;
        rerankPhysicalBytes += p_stats.m_rerankPhysicalBytes;
    }
};

struct Metrics
{
    std::uint64_t resultCount = 0;
    std::uint64_t recallHits = 0;
    std::size_t failures = 0;
    WorkStats work;
};

void Usage(const char* p_program)
{
    std::cerr << "Usage: " << p_program
              << " --index <tenant-directory-or-indexloader.ini>"
              << " --query <query.u8bin>"
              << " --ground-truth <gt_unfilter.ibin>"
              << " --exact-heads <exact_heads.bin>"
              << " --head-k <K> --offset <query-offset> --count <query-count>"
              << " [--compare-normal] [--output result.jsonl]\n";
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
        if (std::strcmp(arg, "--compare-normal") == 0) {
            p_options.compareNormal = true;
            continue;
        }
        if (i + 1 >= p_argc) return false;

        const char* value = p_argv[++i];
        if (std::strcmp(arg, "--index") == 0) {
            p_options.indexPath = value;
        } else if (std::strcmp(arg, "--query") == 0) {
            p_options.queryFile = value;
        } else if (std::strcmp(arg, "--ground-truth") == 0) {
            p_options.groundTruthFile = value;
        } else if (std::strcmp(arg, "--exact-heads") == 0) {
            p_options.exactHeadsFile = value;
        } else if (std::strcmp(arg, "--output") == 0) {
            p_options.outputFile = value;
        } else if (std::strcmp(arg, "--offset") == 0) {
            if (!ParseSize(value, p_options.offset)) return false;
            p_options.hasOffset = true;
        } else if (std::strcmp(arg, "--count") == 0) {
            if (!ParseSize(value, p_options.count) || p_options.count == 0) return false;
            p_options.hasCount = true;
        } else if (std::strcmp(arg, "--head-k") == 0) {
            if (!ParseSize(value, p_options.headK) || p_options.headK == 0 ||
                p_options.headK > static_cast<std::size_t>((std::numeric_limits<int>::max)())) {
                return false;
            }
        } else {
            return false;
        }
    }

    return !p_options.indexPath.empty() && !p_options.queryFile.empty() &&
        !p_options.groundTruthFile.empty() && !p_options.exactHeadsFile.empty() &&
        p_options.headK > 0 && p_options.hasOffset && p_options.hasCount;
}

std::string Lower(std::string p_value)
{
    std::transform(p_value.begin(), p_value.end(), p_value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return p_value;
}

bool IsUInt8L2SpannConfig(const Helper::IniReader& p_config)
{
    return Lower(p_config.GetParameter("Index", "IndexAlgoType", std::string())) == "spann" &&
        Lower(p_config.GetParameter("Index", "ValueType", std::string())) == "uint8" &&
        Lower(p_config.GetParameter("Base", "ValueType", std::string())) == "uint8" &&
        Lower(p_config.GetParameter("Base", "DistCalcMethod", std::string())) == "l2";
}

bool ResolveRuntimeConfig(const Options& p_options, RuntimeConfig& p_runtime)
{
    namespace fs = std::filesystem;
    std::error_code error;
    const fs::path input(p_options.indexPath);
    fs::path configPath;
    if (fs::is_directory(input, error)) {
        configPath = input / "indexloader.ini";
    } else if (!error && fs::is_regular_file(input, error)) {
        configPath = input;
        if (configPath.filename() != "indexloader.ini") {
            std::cerr << "Native index loading requires an indexloader.ini config path\n";
            return false;
        }
    } else {
        std::cerr << "Cannot access index path: " << p_options.indexPath << "\n";
        return false;
    }

    Helper::IniReader config;
    if (config.LoadIniFile(configPath.string()) != ErrorCode::Success) {
        std::cerr << "Cannot load native index INI: " << configPath << "\n";
        return false;
    }
    if (!IsUInt8L2SpannConfig(config)) {
        std::cerr << "Index INI must describe a UInt8 L2 SPANN index\n";
        return false;
    }
    if (Lower(config.GetParameter("BuildSSDIndex", "Storage", std::string())) != "static") {
        std::cerr << "Index INI must describe STATIC postings\n";
        return false;
    }
    if (!config.DoesParameterExist("BuildSSDIndex", "PostingOffset")) {
        std::cerr << "Index INI must explicitly declare [BuildSSDIndex] PostingOffset\n";
        return false;
    }

    const int dimension = config.GetParameter<int>("Base", "Dim", -1);
    const int postingOffset = config.GetParameter<int>("BuildSSDIndex", "PostingOffset", -1);
    if (dimension <= 0 || postingOffset != 0) {
        if (postingOffset != 0) {
            std::cerr << "This oracle requires PostingOffset==0; index has PostingOffset="
                      << postingOffset << "\n";
        } else {
            std::cerr << "Index INI needs a positive [Base] Dim\n";
        }
        return false;
    }

    int postingPageLimit = -1;
    std::string postingPageLimitKey;
    if (config.DoesParameterExist("SearchSSDIndex", "PostingPageLimit")) {
        postingPageLimit = config.GetParameter<int>("SearchSSDIndex", "PostingPageLimit", -1);
        postingPageLimitKey = "PostingPageLimit";
    } else if (config.DoesParameterExist("SearchSSDIndex", "SearchPostingPageLimit")) {
        postingPageLimit = config.GetParameter<int>("SearchSSDIndex", "SearchPostingPageLimit", -1);
        postingPageLimitKey = "SearchPostingPageLimit";
    }
    const int internalResultNum = config.GetParameter<int>("SearchSSDIndex", "InternalResultNum", -1);
    if (postingPageLimit <= 0 || internalResultNum <= 0) {
        std::cerr << "Index INI needs positive [SearchSSDIndex] PostingPageLimit and InternalResultNum\n";
        return false;
    }

    p_runtime.indexDirectory = configPath.parent_path().empty()
        ? "."
        : configPath.parent_path().string();
    p_runtime.dimension = static_cast<DimensionType>(dimension);
    p_runtime.postingOffset = postingOffset;
    p_runtime.postingPageLimit = postingPageLimit;
    p_runtime.internalResultNum = internalResultNum;
    p_runtime.postingPageLimitKey = std::move(postingPageLimitKey);
    return true;
}

bool ReadQuerySlice(const Options& p_options, DimensionType p_dimension, QuerySlice& p_queries)
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
        rawCount <= 0 || rawDimension <= 0 || rawDimension != p_dimension) {
        std::cerr << "Invalid UInt8 query header or dimension mismatch\n";
        return false;
    }

    p_queries.totalCount = static_cast<std::size_t>(rawCount);
    p_queries.dimension = static_cast<std::size_t>(rawDimension);
    if (p_queries.totalCount > (std::numeric_limits<std::size_t>::max)() / p_queries.dimension ||
        p_options.offset > p_queries.totalCount ||
        p_options.count > p_queries.totalCount - p_options.offset) {
        std::cerr << "Query slice is outside the UInt8 query dataset\n";
        return false;
    }
    const std::size_t payloadBytes = p_queries.totalCount * p_queries.dimension;
    if (payloadBytes > (std::numeric_limits<std::size_t>::max)() - 2 * sizeof(std::int32_t) ||
        fileSize < 0 || static_cast<std::uintmax_t>(fileSize) != 2 * sizeof(std::int32_t) + payloadBytes) {
        std::cerr << "Unexpected size for UInt8 query dataset\n";
        return false;
    }

    p_queries.values.resize(p_options.count * p_queries.dimension);
    input.seekg(static_cast<std::streamoff>(2 * sizeof(std::int32_t) +
        p_options.offset * p_queries.dimension));
    if (!input.read(reinterpret_cast<char*>(p_queries.values.data()),
                    static_cast<std::streamsize>(p_queries.values.size()))) {
        std::cerr << "Cannot read query slice\n";
        return false;
    }
    return true;
}

bool ReadExactHeads(const Options& p_options, ExactHeadSlice& p_exactHeads)
{
    std::ifstream input(p_options.exactHeadsFile, std::ios::binary | std::ios::ate);
    if (!input) {
        std::cerr << "Cannot open exact-head file: " << p_options.exactHeadsFile << "\n";
        return false;
    }
    const std::streamoff fileSize = input.tellg();
    input.seekg(0);

    ExactHeadsFileHeader header{};
    if (!input.read(reinterpret_cast<char*>(&header), sizeof(header)) ||
        header.magic != kExactHeadsMagic || header.version != kExactHeadsVersion ||
        header.headerBytes != sizeof(header) || header.idBytes != sizeof(std::int32_t) ||
        header.queryCount == 0 || header.headK == 0) {
        std::cerr << "Invalid exact-head file header\n";
        return false;
    }
    if (header.queryOffset > (std::numeric_limits<std::size_t>::max)() ||
        header.queryCount > (std::numeric_limits<std::size_t>::max)() ||
        header.headK > (std::numeric_limits<std::size_t>::max)()) {
        std::cerr << "Exact-head file header exceeds this platform's size limits\n";
        return false;
    }
    p_exactHeads.offset = static_cast<std::size_t>(header.queryOffset);
    p_exactHeads.count = static_cast<std::size_t>(header.queryCount);
    p_exactHeads.storedK = static_cast<std::size_t>(header.headK);
    if (p_exactHeads.offset != p_options.offset || p_exactHeads.count != p_options.count) {
        std::cerr << "Exact-head file query offset/count does not match the requested query slice\n";
        return false;
    }
    if (p_options.headK > p_exactHeads.storedK) {
        std::cerr << "--head-k exceeds exact-head file K\n";
        return false;
    }
    if (p_exactHeads.count > (std::numeric_limits<std::size_t>::max)() / p_exactHeads.storedK ||
        p_exactHeads.count * p_exactHeads.storedK >
            ((std::numeric_limits<std::size_t>::max)() - sizeof(header)) / sizeof(std::int32_t)) {
        std::cerr << "Exact-head file size overflows\n";
        return false;
    }
    const std::size_t expectedSize = sizeof(header) +
        p_exactHeads.count * p_exactHeads.storedK * sizeof(std::int32_t);
    if (fileSize < 0 || static_cast<std::uintmax_t>(fileSize) != expectedSize) {
        std::cerr << "Unexpected exact-head file size\n";
        return false;
    }

    p_exactHeads.ids.assign(p_exactHeads.count, std::vector<SizeType>(p_options.headK));
    for (std::size_t query = 0; query < p_exactHeads.count; ++query) {
        for (std::size_t rank = 0; rank < p_exactHeads.storedK; ++rank) {
            std::int32_t id = -1;
            if (!input.read(reinterpret_cast<char*>(&id), sizeof(id)) || id < 0) {
                std::cerr << "Invalid exact-head local ID\n";
                return false;
            }
            if (rank < p_options.headK) {
                p_exactHeads.ids[query][rank] = static_cast<SizeType>(id);
            }
        }
    }
    return true;
}

bool ReadGroundTruth(const Options& p_options,
                     std::size_t p_queryCount,
                     std::vector<std::unordered_set<SizeType>>& p_groundTruth)
{
    std::ifstream input(p_options.groundTruthFile, std::ios::binary | std::ios::ate);
    if (!input) {
        std::cerr << "Cannot open ground-truth file: " << p_options.groundTruthFile << "\n";
        return false;
    }
    const std::streamoff fileSize = input.tellg();
    input.seekg(0);
    std::int32_t rawCount = 0;
    std::int32_t rawK = 0;
    if (!input.read(reinterpret_cast<char*>(&rawCount), sizeof(rawCount)) ||
        !input.read(reinterpret_cast<char*>(&rawK), sizeof(rawK)) ||
        rawCount <= 0 || rawK < kResultK) {
        std::cerr << "Invalid .ibin ground-truth header; it must contain at least 10 IDs per query\n";
        return false;
    }
    const std::size_t totalCount = static_cast<std::size_t>(rawCount);
    const std::size_t storedK = static_cast<std::size_t>(rawK);
    if (totalCount != p_queryCount || p_options.offset > totalCount ||
        p_options.count > totalCount - p_options.offset ||
        totalCount > (std::numeric_limits<std::size_t>::max)() / storedK ||
        totalCount * storedK >
            ((std::numeric_limits<std::size_t>::max)() - 2 * sizeof(std::int32_t)) / sizeof(std::int32_t)) {
        std::cerr << "Ground-truth count or size is incompatible with the query dataset\n";
        return false;
    }
    const std::size_t expectedSize = 2 * sizeof(std::int32_t) +
        totalCount * storedK * sizeof(std::int32_t);
    if (fileSize < 0 || static_cast<std::uintmax_t>(fileSize) != expectedSize) {
        std::cerr << "Unexpected .ibin ground-truth size\n";
        return false;
    }

    p_groundTruth.assign(p_options.count, {});
    input.seekg(static_cast<std::streamoff>(2 * sizeof(std::int32_t) +
        p_options.offset * storedK * sizeof(std::int32_t)));
    for (std::size_t query = 0; query < p_options.count; ++query) {
        auto& ids = p_groundTruth[query];
        ids.reserve(kResultK);
        for (std::size_t rank = 0; rank < storedK; ++rank) {
            std::int32_t id = -1;
            if (!input.read(reinterpret_cast<char*>(&id), sizeof(id))) {
                std::cerr << "Cannot read .ibin ground-truth IDs\n";
                return false;
            }
            if (rank < kResultK && id >= 0) ids.insert(static_cast<SizeType>(id));
        }
        if (ids.size() != kResultK) {
            std::cerr << "Ground truth has duplicate or invalid IDs in its first 10 results\n";
            return false;
        }
    }
    return true;
}

bool ValidateExactHeadIDs(const ExactHeadSlice& p_exactHeads, const VectorIndex& p_index)
{
    const SizeType headCount = p_index.GetNumSamples();
    if (headCount <= 0) {
        std::cerr << "Native index has no head samples\n";
        return false;
    }
    for (const auto& query : p_exactHeads.ids) {
        std::unordered_set<SizeType> seen;
        seen.reserve(query.size());
        for (SizeType id : query) {
            if (id < 0 || id >= headCount) {
                std::cerr << "Exact-head local ID is outside the native head index range\n";
                return false;
            }
            if (!seen.insert(id).second) {
                std::cerr << "Exact-head file contains duplicate local IDs for a query\n";
                return false;
            }
        }
    }
    return true;
}

void AddRecallMetrics(const QueryResult& p_results,
                      const std::unordered_set<SizeType>& p_groundTruth,
                      Metrics& p_metrics)
{
    std::unordered_set<SizeType> seen;
    seen.reserve(kResultK);
    for (int rank = 0; rank < kResultK; ++rank) {
        const BasicResult* result = p_results.GetResult(rank);
        if (result == nullptr || result->VID < 0 || !seen.insert(result->VID).second) continue;
        ++p_metrics.resultCount;
        if (p_groundTruth.find(result->VID) != p_groundTruth.end()) {
            ++p_metrics.recallHits;
        }
    }
}

Metrics RunExactHeadProductionScan(const Options& p_options,
                                   const QuerySlice& p_queries,
                                   const ExactHeadSlice& p_exactHeads,
                                   const std::vector<std::unordered_set<SizeType>>& p_groundTruth,
                                   const VectorIndex& p_index)
{
    Metrics metrics;
    for (std::size_t query = 0; query < p_options.count; ++query) {
        VectorIndex::ThreadLocalSearchContext context;
        context.m_active = true;
        context.m_directPostingIDs = p_exactHeads.ids[query];
        context.m_directHeadLocalIDs = p_exactHeads.ids[query];
        QueryResult results(p_queries.values.data() + query * p_queries.dimension, kResultK, false);
        ErrorCode status;
        {
            VectorIndex::ThreadLocalSearchContextGuard guard(std::move(context));
            status = p_index.SearchIndex(results);
            metrics.work.Add(VectorIndex::GetThreadLocalPostingScanStats());
        }
        if (status != ErrorCode::Success) {
            ++metrics.failures;
            continue;
        }
        AddRecallMetrics(results, p_groundTruth[query], metrics);
    }
    return metrics;
}

Metrics RunNormalProductionRouting(const Options& p_options,
                                   const QuerySlice& p_queries,
                                   const std::vector<std::unordered_set<SizeType>>& p_groundTruth,
                                   const VectorIndex& p_index)
{
    Metrics metrics;
    for (std::size_t query = 0; query < p_options.count; ++query) {
        QueryResult results(p_queries.values.data() + query * p_queries.dimension, kResultK, false);
        const ErrorCode status = p_index.SearchIndex(results);
        metrics.work.Add(VectorIndex::GetThreadLocalPostingScanStats());
        if (status != ErrorCode::Success) {
            ++metrics.failures;
            continue;
        }
        AddRecallMetrics(results, p_groundTruth[query], metrics);
    }
    return metrics;
}

void WriteResult(std::ostream& p_output,
                 const Options& p_options,
                 const RuntimeConfig& p_runtime,
                 const Metrics& p_metrics,
                 bool p_isExactHeadScan)
{
    const double denominator = static_cast<double>(p_options.count) * kResultK;
    p_output << std::setprecision(10)
             << "{"
             << "\"type\":\""
             << (p_isExactHeadScan ? "exact_head_production_scan_oracle" : "normal_production_routing_comparison")
             << "\","
             << "\"oracle\":\""
             << (p_isExactHeadScan ? "exact-head production-scan" : "normal production routing")
             << "\","
             << "\"interpretation\":\""
             << (p_isExactHeadScan
                     ? "production PostingPageLimit scan; not an unlimited posting-membership bound"
                     : "native BKT routing with unchanged loaded INI settings and no direct posting IDs")
             << "\","
             << "\"head_selection\":\""
             << (p_isExactHeadScan ? "exact_head_local_ids" : "native_bkt_routing")
             << "\","
             << "\"direct_posting_ids\":" << (p_isExactHeadScan ? "true" : "false") << ","
             << "\"comparison_head_k\":" << p_options.headK << ","
             << "\"comparison_head_k_matches_native_internal_result_num\":"
             << (p_options.headK == static_cast<std::size_t>(p_runtime.internalResultNum) ? "true" : "false")
             << ","
             << "\"query_offset\":" << p_options.offset << ","
             << "\"query_count\":" << p_options.count << ","
             << "\"result_count\":" << p_metrics.resultCount << ","
             << "\"recall_at_10\":" << p_metrics.recallHits / denominator << ","
             << "\"failures\":" << p_metrics.failures << ","
             << "\"posting_offset\":" << p_runtime.postingOffset << ","
             << "\"posting_page_limit_source\":\"indexloader.ini.SearchSSDIndex."
             << p_runtime.postingPageLimitKey << "\","
             << "\"posting_page_limit\":" << p_runtime.postingPageLimit << ","
             << "\"internal_result_num_source\":\"indexloader.ini.SearchSSDIndex.InternalResultNum\","
             << "\"internal_result_num\":" << p_runtime.internalResultNum << ","
             << "\"" << (p_isExactHeadScan ? "direct_posting_scan" : "posting_scan") << "\":{"
             << "\"read_postings\":" << p_metrics.work.readPostings << ","
             << "\"matched_postings\":" << p_metrics.work.matchedPostings << ","
             << "\"pre_ps_postings\":" << p_metrics.work.prePSPostings << ","
             << "\"scanned_vectors\":" << p_metrics.work.scannedVectors << ","
             << "\"matched_vectors\":" << p_metrics.work.matchedVectors << ","
             << "\"unique_matched_postings\":" << p_metrics.work.uniqueMatchedPostings << ","
             << "\"unique_matched_vectors\":" << p_metrics.work.uniqueMatchedVectors << ","
             << "\"primary_head_candidates\":" << p_metrics.work.primaryHeadCandidates << ","
             << "\"posting_page_reads\":" << p_metrics.work.postingPageReads << ","
             << "\"posting_logical_bytes\":" << p_metrics.work.postingLogicalBytes << ","
             << "\"posting_physical_bytes\":" << p_metrics.work.postingPhysicalBytes << ","
             << "\"adc_scanned_vectors\":" << p_metrics.work.adcScannedVectors << ","
             << "\"adc_survivors\":" << p_metrics.work.adcSurvivors << ","
             << "\"rerank_candidates\":" << p_metrics.work.rerankCandidates << ","
             << "\"rerank_read_requests\":" << p_metrics.work.rerankReadRequests << ","
             << "\"rerank_physical_bytes\":" << p_metrics.work.rerankPhysicalBytes
             << "}}\n";
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!ParseArgs(argc, argv, options)) {
        Usage(argv[0]);
        return 2;
    }

    RuntimeConfig runtime;
    if (!ResolveRuntimeConfig(options, runtime)) return 1;

    QuerySlice queries;
    if (!ReadQuerySlice(options, runtime.dimension, queries)) return 1;

    ExactHeadSlice exactHeads;
    if (!ReadExactHeads(options, exactHeads)) return 1;

    std::vector<std::unordered_set<SizeType>> groundTruth;
    if (!ReadGroundTruth(options, queries.totalCount, groundTruth)) return 1;

    std::shared_ptr<VectorIndex> index;
    if (VectorIndex::LoadIndex(runtime.indexDirectory, index) != ErrorCode::Success || index == nullptr) {
        std::cerr << "Cannot load native STATIC index: " << runtime.indexDirectory << "\n";
        return 1;
    }
    if (index->GetIndexAlgoType() != IndexAlgoType::SPANN ||
        index->GetVectorValueType() != VectorValueType::UInt8 ||
        index->GetDistCalcMethod() != DistCalcMethod::L2 ||
        index->GetFeatureDim() != runtime.dimension) {
        std::cerr << "Native index does not match the UInt8 L2 SPANN configuration\n";
        return 1;
    }
    if (!ValidateExactHeadIDs(exactHeads, *index)) return 1;

    const Metrics exactMetrics = RunExactHeadProductionScan(
        options, queries, exactHeads, groundTruth, *index);
    Metrics normalMetrics;
    if (options.compareNormal) {
        normalMetrics = RunNormalProductionRouting(options, queries, groundTruth, *index);
    }

    if (options.outputFile.empty()) {
        WriteResult(std::cout, options, runtime, exactMetrics, true);
        if (options.compareNormal) {
            WriteResult(std::cout, options, runtime, normalMetrics, false);
        }
        return std::cout ? 0 : 1;
    }

    std::ofstream output(options.outputFile, std::ios::out | std::ios::trunc);
    if (!output) {
        std::cerr << "Cannot open output file: " << options.outputFile << "\n";
        return 1;
    }
    WriteResult(output, options, runtime, exactMetrics, true);
    if (options.compareNormal) {
        WriteResult(output, options, runtime, normalMetrics, false);
    }
    output.close();
    if (!output) {
        std::cerr << "Cannot write output file: " << options.outputFile << "\n";
        return 1;
    }
    return 0;
}
