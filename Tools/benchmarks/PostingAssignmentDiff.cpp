// Native small-sample comparison of STATIC posting assignments.  This does not
// write, alter, or inspect the existing posting files.
#include "inc/Core/BKT/Index.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/VectorSet.h"
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
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace SPTAG;

namespace {

constexpr std::uint64_t kExactHeadsMagic = 0x3153444145485053ULL; // "SPHEADS1", little-endian.
constexpr std::uint32_t kExactHeadsVersion = 1;

struct ExactHeadsFileHeader {
    std::uint64_t magic;
    std::uint32_t version;
    std::uint32_t headerBytes;
    std::uint64_t queryOffset;
    std::uint64_t queryCount;
    std::uint32_t headK;
    std::uint32_t idBytes;
};
static_assert(sizeof(ExactHeadsFileHeader) == 40, "Unexpected exact-head header layout");

struct Options {
    std::string rootIni;
    std::string newHead;
    std::string oldHead;
    std::string base;
    std::string headIDs;
    std::string query;
    std::string groundTruth;
    std::string exactHeads;
    std::string output;
    bool gtMode = false;
    bool uniformMode = false;
    std::size_t rankBegin = 0;
    std::size_t rankEnd = 0;
    std::size_t queryOffset = 0;
    std::size_t queryCount = 0;
    std::size_t uniformCount = 0;
    std::uint64_t uniformSeed = 0;
};

struct Config {
    DimensionType dimension = 0;
    int buildMaxCheck = 0;
    int candidateNum = 0;
    int replicaCount = 0;
    float rngFactor = 0;
    bool excludeHead = false;
    int assignmentThreads = 1;
    int gpuSSDNumTrees = 0;
    int gpuSSDLeafSize = 0;
    int numGPUs = 0;
};

struct BaseInfo {
    std::size_t count = 0;
    std::size_t dimension = 0;
};

struct Assignment {
    std::uint64_t vid = 0;
    bool isHead = false;
    SizeType localHead = -1;
    std::vector<SizeType> oldHeads;
    std::vector<SizeType> newHeads;
};

struct Pair {
    std::size_t query = 0;
    std::size_t rank = 0;
    std::uint64_t vid = 0;
};

struct ExactHeads {
    std::vector<std::vector<SizeType>> ids;
};

std::string Lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

void Usage(const char* program)
{
    std::cerr
        << "Usage (GT pairs): " << program
        << " --root-ini INDEX/indexloader.ini --new-head INDEX/HeadIndex"
        << " --old-head INDEX/HeadIndex.headrebuild.stage.N --base sift1b_base.u8bin"
        << " --head-ids INDEX/SPTAGHeadVectorIDs.bin --query query.u8bin"
        << " --ground-truth gt.ibin --exact-heads exact_heads.bin"
        << " --gt-rank-range begin:end --query-offset N --query-count N [--output out.jsonl]\n"
        << "Usage (uniform): " << program
        << " --root-ini INDEX/indexloader.ini --new-head INDEX/HeadIndex"
        << " --old-head INDEX/HeadIndex.headrebuild.stage.N --base sift1b_base.u8bin"
        << " --head-ids INDEX/SPTAGHeadVectorIDs.bin"
        << " --uniform-count N --uniform-seed S [--output out.jsonl]\n";
}

bool ParseSize(const char* text, std::size_t& value)
{
    if (text == nullptr || *text == '\0') return false;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' ||
        parsed > static_cast<unsigned long long>((std::numeric_limits<std::size_t>::max)())) {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return true;
}

bool ParseU64(const char* text, std::uint64_t& value)
{
    std::size_t parsed = 0;
    if (!ParseSize(text, parsed)) return false;
    value = static_cast<std::uint64_t>(parsed);
    return true;
}

bool ParseRange(const char* text, std::size_t& begin, std::size_t& end)
{
    const char* colon = std::strchr(text, ':');
    if (colon == nullptr || std::strchr(colon + 1, ':') != nullptr) return false;
    const std::string left(text, static_cast<std::size_t>(colon - text));
    const std::string right(colon + 1);
    return ParseSize(left.c_str(), begin) && ParseSize(right.c_str(), end) && begin < end;
}

bool ParseArgs(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            Usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) return false;
        const char* value = argv[++i];
        if (std::strcmp(arg, "--root-ini") == 0) options.rootIni = value;
        else if (std::strcmp(arg, "--new-head") == 0) options.newHead = value;
        else if (std::strcmp(arg, "--old-head") == 0) options.oldHead = value;
        else if (std::strcmp(arg, "--base") == 0) options.base = value;
        else if (std::strcmp(arg, "--head-ids") == 0) options.headIDs = value;
        else if (std::strcmp(arg, "--query") == 0) options.query = value;
        else if (std::strcmp(arg, "--ground-truth") == 0) options.groundTruth = value;
        else if (std::strcmp(arg, "--exact-heads") == 0) options.exactHeads = value;
        else if (std::strcmp(arg, "--output") == 0) options.output = value;
        else if (std::strcmp(arg, "--gt-rank-range") == 0) {
            if (!ParseRange(value, options.rankBegin, options.rankEnd)) return false;
            options.gtMode = true;
        } else if (std::strcmp(arg, "--query-offset") == 0) {
            if (!ParseSize(value, options.queryOffset)) return false;
        } else if (std::strcmp(arg, "--query-count") == 0) {
            if (!ParseSize(value, options.queryCount) || options.queryCount == 0) return false;
        } else if (std::strcmp(arg, "--uniform-count") == 0) {
            if (!ParseSize(value, options.uniformCount) || options.uniformCount == 0) return false;
            options.uniformMode = true;
        } else if (std::strcmp(arg, "--uniform-seed") == 0) {
            if (!ParseU64(value, options.uniformSeed)) return false;
        } else return false;
    }
    const bool common = !options.rootIni.empty() && !options.newHead.empty() && !options.oldHead.empty() &&
        !options.base.empty() && !options.headIDs.empty();
    const bool gt = options.gtMode && !options.uniformMode && options.queryCount > 0 &&
        !options.query.empty() && !options.groundTruth.empty() && !options.exactHeads.empty();
    return common && (gt || (options.uniformMode && !options.gtMode));
}

bool IsUInt8L2Static(const Helper::IniReader& ini)
{
    return Lower(ini.GetParameter("Index", "IndexAlgoType", std::string())) == "spann" &&
        Lower(ini.GetParameter("Index", "ValueType", std::string())) == "uint8" &&
        Lower(ini.GetParameter("Base", "ValueType", std::string())) == "uint8" &&
        Lower(ini.GetParameter("Base", "DistCalcMethod", std::string())) == "l2" &&
        Lower(ini.GetParameter("BuildSSDIndex", "Storage", std::string())) == "static";
}

bool ReadConfig(const Options& options, Config& config)
{
    Helper::IniReader ini;
    if (ini.LoadIniFile(options.rootIni) != ErrorCode::Success || !IsUInt8L2Static(ini)) {
        std::cerr << "Root INI must be a native UInt8/L2 global STATIC SPANN index\n";
        return false;
    }
    if (!ini.DoesParameterExist("BuildSSDIndex", "InternalResultNum") ||
        !ini.DoesParameterExist("BuildSSDIndex", "ReplicaCount") ||
        !ini.DoesParameterExist("BuildSSDIndex", "RNGFactor") ||
        !ini.DoesParameterExist("BuildSSDIndex", "ExcludeHead") ||
        !ini.DoesParameterExist("BuildSSDIndex", "TailReplicaCount") ||
        !ini.DoesParameterExist("BuildSSDIndex", "MaxCheck") ||
        !ini.DoesParameterExist("BuildSSDIndex", "GPUSSDNumTrees") ||
        !ini.DoesParameterExist("BuildSSDIndex", "GPUSSDLeafSize") ||
        !ini.DoesParameterExist("BuildSSDIndex", "NumGPUs")) {
        std::cerr << "Root INI lacks a required [BuildSSDIndex] assignment parameter\n";
        return false;
    }
    const int dimension = ini.GetParameter<int>("Base", "Dim", -1);
    config.buildMaxCheck = ini.GetParameter<int>("BuildSSDIndex", "MaxCheck", -1);
    config.candidateNum = ini.GetParameter<int>("BuildSSDIndex", "InternalResultNum", -1);
    config.replicaCount = ini.GetParameter<int>("BuildSSDIndex", "ReplicaCount", -1);
    config.rngFactor = ini.GetParameter<float>("BuildSSDIndex", "RNGFactor", -1);
    config.excludeHead = ini.GetParameter<bool>("BuildSSDIndex", "ExcludeHead", false);
    config.gpuSSDNumTrees = ini.GetParameter<int>("BuildSSDIndex", "GPUSSDNumTrees", -1);
    config.gpuSSDLeafSize = ini.GetParameter<int>("BuildSSDIndex", "GPUSSDLeafSize", -1);
    config.numGPUs = ini.GetParameter<int>("BuildSSDIndex", "NumGPUs", -1);
    const int tailReplicaCount = ini.GetParameter<int>("BuildSSDIndex", "TailReplicaCount", -1);
    if (dimension <= 0 || config.buildMaxCheck <= 0 || config.candidateNum <= 0 || config.replicaCount <= 0 ||
        config.rngFactor < 0 || config.gpuSSDNumTrees < 0 || config.gpuSSDLeafSize < 0 ||
        config.numGPUs < 0 || tailReplicaCount != 0) {
        std::cerr << "Invalid build assignment settings (this simulator requires TailReplicaCount=0)\n";
        return false;
    }
    config.dimension = static_cast<DimensionType>(dimension);
    // ApproximateRNG's build worker count is BuildSSDIndex.NumberOfThreads.  Do not
    // silently use the library default (16) when the INI did not explicitly set it.
    if (ini.DoesParameterExist("BuildSSDIndex", "NumberOfThreads")) {
        config.assignmentThreads = ini.GetParameter<int>("BuildSSDIndex", "NumberOfThreads", -1);
        if (config.assignmentThreads <= 0) {
            std::cerr << "Invalid [BuildSSDIndex] NumberOfThreads\n";
            return false;
        }
    }
    return true;
}

bool SameVectorFile(const std::string& newHead, const std::string& oldHead)
{
    struct stat current {};
    struct stat previous {};
    if (stat((newHead + "/vectors.bin").c_str(), &current) != 0 ||
        stat((oldHead + "/vectors.bin").c_str(), &previous) != 0 ||
        current.st_dev != previous.st_dev || current.st_ino != previous.st_ino) {
        std::cerr << "Old and new HeadIndex vectors.bin must be the same hard-linked file\n";
        return false;
    }
    return true;
}

using UInt8BKT = BKT::Index<std::uint8_t>;

bool LoadHead(const std::string& path, DimensionType dimension, int maxCheck, std::shared_ptr<UInt8BKT>& index)
{
    std::shared_ptr<VectorIndex> loaded;
    if (VectorIndex::LoadIndex(path, loaded) != ErrorCode::Success || loaded == nullptr) {
        std::cerr << "Cannot load BKT HeadIndex: " << path << "\n";
        return false;
    }
    index = std::dynamic_pointer_cast<UInt8BKT>(loaded);
    if (index == nullptr || index->GetIndexAlgoType() != IndexAlgoType::BKT ||
        index->GetVectorValueType() != VectorValueType::UInt8 ||
        index->GetDistCalcMethod() != DistCalcMethod::L2 ||
        index->GetFeatureDim() != dimension || index->GetNumSamples() <= 0) {
        std::cerr << "HeadIndex is not a nonempty UInt8/L2 BKT index of the configured dimension\n";
        return false;
    }
    const std::string maxCheckValue = std::to_string(maxCheck);
    if (index->SetParameter("MaxCheck", maxCheckValue.c_str()) != ErrorCode::Success ||
        index->GetCurrMaxCheck() != maxCheck) {
        std::cerr << "Cannot apply root [BuildSSDIndex] MaxCheck to HeadIndex\n";
        return false;
    }
    return true;
}

bool ReadBaseInfo(const std::string& path, DimensionType expectedDimension, BaseInfo& info)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        std::cerr << "Cannot open base file\n";
        return false;
    }
    const std::streamoff fileSize = input.tellg();
    input.seekg(0);
    std::int32_t count = 0;
    std::int32_t dimension = 0;
    if (!input.read(reinterpret_cast<char*>(&count), sizeof(count)) ||
        !input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension)) ||
        count <= 0 || dimension <= 0 || dimension != expectedDimension) {
        std::cerr << "Invalid base .u8bin header or dimension mismatch\n";
        return false;
    }
    const std::uint64_t payload = static_cast<std::uint64_t>(count) * static_cast<std::uint64_t>(dimension);
    if (fileSize < 0 || payload > static_cast<std::uint64_t>((std::numeric_limits<std::streamoff>::max)()) -
            2 * sizeof(std::int32_t) ||
        static_cast<std::uint64_t>(fileSize) != 2 * sizeof(std::int32_t) + payload) {
        std::cerr << "Unexpected base .u8bin size\n";
        return false;
    }
    info.count = static_cast<std::size_t>(count);
    info.dimension = static_cast<std::size_t>(dimension);
    return true;
}

bool ReadSelectedBase(const std::string& path, const BaseInfo& info,
                      const std::vector<std::uint64_t>& vids, std::vector<std::uint8_t>& values)
{
    values.resize(vids.size() * info.dimension);
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    for (std::size_t i = 0; i < vids.size(); ++i) {
        if (vids[i] >= info.count) {
            std::cerr << "Selected VID is outside the base dataset\n";
            return false;
        }
        const std::uint64_t offset = 2 * sizeof(std::int32_t) + vids[i] * info.dimension;
        if (offset > static_cast<std::uint64_t>((std::numeric_limits<std::streamoff>::max)())) {
            std::cerr << "Base seek offset overflows stream position\n";
            return false;
        }
        input.seekg(static_cast<std::streamoff>(offset));
        if (!input.read(reinterpret_cast<char*>(values.data() + i * info.dimension),
                        static_cast<std::streamsize>(info.dimension))) {
            std::cerr << "Cannot random-read selected base vector\n";
            return false;
        }
    }
    return true;
}

bool ReverseMapSelectedHeads(const std::string& path, SizeType expectedHeads,
                             const std::vector<std::uint64_t>& selected,
                             std::unordered_map<std::uint64_t, SizeType>& selectedHeads)
{
    std::unordered_set<std::uint64_t> wanted(selected.begin(), selected.end());
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        std::cerr << "Cannot open SPTAGHeadVectorIDs.bin\n";
        return false;
    }
    const std::streamoff fileSize = input.tellg();
    input.seekg(0);
    std::int32_t rows = 0;
    std::int32_t columns = 0;
    if (!input.read(reinterpret_cast<char*>(&rows), sizeof(rows)) ||
        !input.read(reinterpret_cast<char*>(&columns), sizeof(columns)) ||
        rows <= 0 || columns != 1 || rows != expectedHeads ||
        fileSize < 0 || static_cast<std::uint64_t>(fileSize) !=
            2 * sizeof(std::int32_t) + static_cast<std::uint64_t>(rows) * sizeof(std::uint64_t)) {
        std::cerr << "Malformed or incompatible SPTAGHeadVectorIDs.bin header\n";
        return false;
    }
    for (SizeType local = 0; local < rows; ++local) {
        std::uint64_t external = 0;
        if (!input.read(reinterpret_cast<char*>(&external), sizeof(external))) {
            std::cerr << "Cannot read SPTAGHeadVectorIDs.bin\n";
            return false;
        }
        if (wanted.find(external) != wanted.end()) {
            if (!selectedHeads.emplace(external, local).second) {
                std::cerr << "A selected external VID maps to multiple local heads\n";
                return false;
            }
        }
    }
    return true;
}

bool ReadQueryInfo(const std::string& path, DimensionType expectedDimension, std::size_t& count)
{
    BaseInfo info;
    if (!ReadBaseInfo(path, expectedDimension, info)) return false;
    count = info.count;
    return true;
}

bool ReadGTPairs(const Options& options, std::size_t totalQueries, std::vector<Pair>& pairs)
{
    std::ifstream input(options.groundTruth, std::ios::binary | std::ios::ate);
    if (!input) {
        std::cerr << "Cannot open ground-truth .ibin\n";
        return false;
    }
    const std::streamoff fileSize = input.tellg();
    input.seekg(0);
    std::int32_t count = 0;
    std::int32_t k = 0;
    if (!input.read(reinterpret_cast<char*>(&count), sizeof(count)) ||
        !input.read(reinterpret_cast<char*>(&k), sizeof(k)) || count <= 0 || k <= 0 ||
        static_cast<std::size_t>(count) != totalQueries ||
        options.queryOffset > totalQueries || options.queryCount > totalQueries - options.queryOffset ||
        options.rankEnd > static_cast<std::size_t>(k)) {
        std::cerr << "Ground truth header/rank range is incompatible with the query slice\n";
        return false;
    }
    const std::uint64_t expected = 2 * sizeof(std::int32_t) +
        static_cast<std::uint64_t>(count) * static_cast<std::uint64_t>(k) * sizeof(std::int32_t);
    if (fileSize < 0 || static_cast<std::uint64_t>(fileSize) != expected) {
        std::cerr << "Unexpected ground-truth .ibin size\n";
        return false;
    }
    pairs.reserve(options.queryCount * (options.rankEnd - options.rankBegin));
    for (std::size_t q = 0; q < options.queryCount; ++q) {
        const std::uint64_t offset = 2 * sizeof(std::int32_t) +
            (static_cast<std::uint64_t>(options.queryOffset + q) * k + options.rankBegin) * sizeof(std::int32_t);
        input.seekg(static_cast<std::streamoff>(offset));
        for (std::size_t rank = options.rankBegin; rank < options.rankEnd; ++rank) {
            std::int32_t vid = -1;
            if (!input.read(reinterpret_cast<char*>(&vid), sizeof(vid)) || vid < 0) {
                std::cerr << "Invalid ground-truth VID\n";
                return false;
            }
            pairs.push_back({q, rank, static_cast<std::uint64_t>(vid)});
        }
    }
    return true;
}

bool ReadExactHeads(const Options& options, SizeType headCount, ExactHeads& heads)
{
    std::ifstream input(options.exactHeads, std::ios::binary | std::ios::ate);
    if (!input) {
        std::cerr << "Cannot open SPHEADS1 file\n";
        return false;
    }
    const std::streamoff fileSize = input.tellg();
    input.seekg(0);
    ExactHeadsFileHeader header{};
    if (!input.read(reinterpret_cast<char*>(&header), sizeof(header)) ||
        header.magic != kExactHeadsMagic || header.version != kExactHeadsVersion ||
        header.headerBytes != sizeof(header) || header.idBytes != sizeof(std::int32_t) ||
        header.headK < 104 || header.queryOffset > options.queryOffset ||
        options.queryOffset - header.queryOffset > header.queryCount ||
        options.queryCount > header.queryCount - (options.queryOffset - header.queryOffset)) {
        std::cerr << "SPHEADS1 metadata does not cover the requested query slice or lacks K=104\n";
        return false;
    }
    const std::uint64_t expected = sizeof(header) +
        header.queryCount * static_cast<std::uint64_t>(header.headK) * sizeof(std::int32_t);
    if (fileSize < 0 || static_cast<std::uint64_t>(fileSize) != expected) {
        std::cerr << "Malformed SPHEADS1 file size\n";
        return false;
    }
    const std::uint64_t first = options.queryOffset - header.queryOffset;
    const std::uint64_t byteOffset = sizeof(header) +
        first * static_cast<std::uint64_t>(header.headK) * sizeof(std::int32_t);
    input.seekg(static_cast<std::streamoff>(byteOffset));
    if (!input) {
        std::cerr << "Cannot seek to requested SPHEADS1 query slice\n";
        return false;
    }
    heads.ids.assign(options.queryCount, std::vector<SizeType>(header.headK));
    for (auto& query : heads.ids) {
        std::unordered_set<SizeType> unique;
        for (SizeType& id : query) {
            std::int32_t raw = -1;
            if (!input.read(reinterpret_cast<char*>(&raw), sizeof(raw)) || raw < 0 ||
                raw >= headCount ||
                !unique.insert(static_cast<SizeType>(raw)).second) {
                std::cerr << "Invalid/duplicate SPHEADS1 local head ID\n";
                return false;
            }
            id = static_cast<SizeType>(raw);
        }
    }
    return true;
}

std::uint64_t SplitMix64(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::vector<std::uint64_t> UniformSpread(std::size_t count, std::size_t baseCount, std::uint64_t seed)
{
    std::vector<std::uint64_t> output;
    output.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t begin = (static_cast<std::uint64_t>(i) * baseCount) / count;
        const std::size_t end = (static_cast<std::uint64_t>(i + 1) * baseCount) / count;
        output.push_back(begin + SplitMix64(seed + i) % (end - begin));
    }
    return output;
}

bool SimulateAssignments(const Config& config, const std::vector<std::uint64_t>& vids,
                         const std::vector<std::uint8_t>& values,
                         const std::unordered_map<std::uint64_t, SizeType>& selectedHeads,
                         UInt8BKT& oldIndex, UInt8BKT& newIndex, std::vector<Assignment>& assignments)
{
    if (vids.empty() || values.size() != vids.size() * static_cast<std::size_t>(config.dimension)) return false;
    const auto build = [&](UInt8BKT& index, bool old) -> bool {
        if (config.candidateNum > index.GetNumSamples()) {
            std::cerr << "Build InternalResultNum exceeds the HeadIndex sample count\n";
            return false;
        }
        ByteArray bytes(const_cast<std::uint8_t*>(values.data()), values.size(), false);
        std::shared_ptr<VectorSet> vectors(new BasicVectorSet(
            bytes, VectorValueType::UInt8, config.dimension, static_cast<SizeType>(vids.size())));
        std::unordered_set<SizeType> excluded;
        for (std::size_t i = 0; i < vids.size(); ++i) {
            if (selectedHeads.find(vids[i]) != selectedHeads.end()) excluded.insert(static_cast<SizeType>(i));
        }
        std::vector<Edge> edges(vids.size() * static_cast<std::size_t>(config.replicaCount));
        // This is the production VectorIndex::ApproximateRNG call, preserving its
        // candidate-search order, strict RNGFactor*nnDist < candidateDist rejection,
        // and candidate tie behavior rather than reimplementing them here.
        index.ApproximateRNG(vectors, excluded, config.candidateNum, edges.data(), config.replicaCount,
                             config.assignmentThreads, config.gpuSSDNumTrees, config.gpuSSDLeafSize,
                             config.rngFactor, config.numGPUs);
        for (std::size_t i = 0; i < vids.size(); ++i) {
            auto& target = old ? assignments[i].oldHeads : assignments[i].newHeads;
            target.clear();
            const auto head = selectedHeads.find(vids[i]);
            if (head != selectedHeads.end()) {
                if (!config.excludeHead) target.push_back(head->second);
                continue;
            }
            for (int replica = 0; replica < config.replicaCount; ++replica) {
                const SizeType local = edges[i * static_cast<std::size_t>(config.replicaCount) + replica].node;
                if (local == MaxSize) break;
                if (local < 0 || local >= index.GetNumSamples()) {
                    std::cerr << "ApproximateRNG emitted a local head outside the HeadIndex\n";
                    return false;
                }
                target.push_back(local);
            }
        }
        return true;
    };
    assignments.resize(vids.size());
    for (std::size_t i = 0; i < vids.size(); ++i) {
        assignments[i].vid = vids[i];
        const auto found = selectedHeads.find(vids[i]);
        assignments[i].isHead = found != selectedHeads.end();
        assignments[i].localHead = assignments[i].isHead ? found->second : -1;
    }
    return build(oldIndex, true) && build(newIndex, false);
}

double Jaccard(const std::vector<SizeType>& left, const std::vector<SizeType>& right)
{
    std::unordered_set<SizeType> leftSet(left.begin(), left.end());
    std::unordered_set<SizeType> rightSet(right.begin(), right.end());
    std::size_t intersection = 0;
    for (SizeType id : rightSet) {
        if (leftSet.count(id) != 0) ++intersection;
    }
    const std::size_t unionCount = leftSet.size() + rightSet.size() - intersection;
    return unionCount == 0 ? 1.0 : static_cast<double>(intersection) / unionCount;
}

bool SameReplicaSet(const std::vector<SizeType>& left, const std::vector<SizeType>& right)
{
    return std::unordered_set<SizeType>(left.begin(), left.end()) ==
        std::unordered_set<SizeType>(right.begin(), right.end());
}

bool AnyHead(const std::vector<SizeType>& assigned, const std::vector<SizeType>& exact, std::size_t k)
{
    std::unordered_set<SizeType> wanted(exact.begin(), exact.begin() + k);
    return std::any_of(assigned.begin(), assigned.end(), [&](SizeType id) { return wanted.count(id) != 0; });
}

void WriteHeadArray(std::ostream& output, const std::vector<SizeType>& ids)
{
    output << "[";
    for (std::size_t i = 0; i < ids.size(); ++i) output << (i == 0 ? "" : ",") << ids[i];
    output << "]";
}

void WriteConfig(std::ostream& output, const Options& options, const Config& config,
                 std::size_t baseCount, std::size_t headCount)
{
    output << std::setprecision(10) << "{"
           << "\"type\":\"posting_assignment_simulation_config\","
           << "\"assignment_model\":\"global_STATIC_ApproximateRNG\","
           << "\"rewritten_postings\":false,"
           << "\"case\":\"" << (options.gtMode ? "gt_rank_pairs" : "uniform_spread") << "\","
           << "\"base_count\":" << baseCount << ",\"head_count\":" << headCount << ","
           << "\"dimension\":" << config.dimension << ","
           << "\"build_max_check_source\":\"root_ini.BuildSSDIndex.MaxCheck\","
           << "\"build_max_check\":" << config.buildMaxCheck << ","
           << "\"build_internal_result_num\":" << config.candidateNum << ","
           << "\"replica_count\":" << config.replicaCount << ","
           << "\"rng_factor\":" << config.rngFactor << ","
           << "\"exclude_head\":" << (config.excludeHead ? "true" : "false") << ","
           << "\"tail_replica_count\":0,"
           << "\"assignment_threads\":" << config.assignmentThreads << ","
           << "\"gpu_ssd_num_trees_source\":\"root_ini.BuildSSDIndex.GPUSSDNumTrees\","
           << "\"gpu_ssd_num_trees\":" << config.gpuSSDNumTrees << ","
           << "\"gpu_ssd_leaf_size_source\":\"root_ini.BuildSSDIndex.GPUSSDLeafSize\","
           << "\"gpu_ssd_leaf_size\":" << config.gpuSSDLeafSize << ","
           << "\"num_gpus_source\":\"root_ini.BuildSSDIndex.NumGPUs\","
           << "\"num_gpus\":" << config.numGPUs << "}\n";
}

void WriteAssignments(std::ostream& output, const Config& config, const std::vector<Assignment>& assignments)
{
    for (const Assignment& assignment : assignments) {
        output << "{\"type\":\"posting_assignment_simulation_assignment\","
               << "\"vid\":" << assignment.vid << ","
               << "\"is_head_vector\":" << (assignment.isHead ? "true" : "false") << ","
               << "\"excluded_from_postings\":" << (assignment.isHead && config.excludeHead ? "true" : "false") << ","
               << "\"direct_head_local_id\":" << assignment.localHead << ","
               << "\"old_heads\":";
        WriteHeadArray(output, assignment.oldHeads);
        output << ",\"new_heads\":";
        WriteHeadArray(output, assignment.newHeads);
        output << ",\"replica_jaccard\":" << Jaccard(assignment.oldHeads, assignment.newHeads) << "}\n";
    }
}

void WritePairsAndSummary(std::ostream& output, const Options& options, const std::vector<Pair>& pairs,
                          const ExactHeads& exact, const std::unordered_map<std::uint64_t, std::size_t>& byVid,
                          const Config& config, const std::vector<Assignment>& assignments)
{
    std::uint64_t excludedPairs = 0;
    std::uint64_t oldCovered[3] = {};
    std::uint64_t newCovered[3] = {};
    const std::size_t cutoffs[] = {50, 96, 104};
    for (const Pair& pair : pairs) {
        const Assignment& assignment = assignments.at(byVid.at(pair.vid));
        const std::vector<SizeType> effectiveOld = assignment.isHead && options.gtMode
            ? std::vector<SizeType>{assignment.localHead} : assignment.oldHeads;
        const std::vector<SizeType> effectiveNew = assignment.isHead && options.gtMode
            ? std::vector<SizeType>{assignment.localHead} : assignment.newHeads;
        if (assignment.isHead && config.excludeHead) ++excludedPairs;
        output << "{\"type\":\"posting_assignment_simulation_pair\",\"query_index\":"
               << options.queryOffset + pair.query << ",\"gt_rank\":" << pair.rank
               << ",\"vid\":" << pair.vid << ",\"excluded_head\":"
               << (assignment.isHead && config.excludeHead ? "true" : "false");
        for (std::size_t i = 0; i < 3; ++i) {
            const bool oldHit = AnyHead(effectiveOld, exact.ids[pair.query], cutoffs[i]);
            const bool newHit = AnyHead(effectiveNew, exact.ids[pair.query], cutoffs[i]);
            oldCovered[i] += oldHit;
            newCovered[i] += newHit;
            output << ",\"old_covered_at_" << cutoffs[i] << "\":" << (oldHit ? "true" : "false")
                   << ",\"new_covered_at_" << cutoffs[i] << "\":" << (newHit ? "true" : "false");
        }
        output << "}\n";
    }
    output << "{\"type\":\"posting_assignment_simulation_pair_summary\","
           << "\"assignment_model\":\"simulation_not_rewritten_postings\","
           << "\"pair_count\":" << pairs.size() << ",\"excluded_head_pair_count\":" << excludedPairs;
    for (std::size_t i = 0; i < 3; ++i) {
        output << ",\"old_covered_pairs_at_" << cutoffs[i] << "\":" << oldCovered[i]
               << ",\"new_covered_pairs_at_" << cutoffs[i] << "\":" << newCovered[i]
               << ",\"old_coverage_at_" << cutoffs[i] << "\":"
               << static_cast<double>(oldCovered[i]) / pairs.size()
               << ",\"new_coverage_at_" << cutoffs[i] << "\":"
               << static_cast<double>(newCovered[i]) / pairs.size();
    }
    output << "}\n";
}

void WriteAssignmentSummary(std::ostream& output, const Options& options,
                            const Config& config, const std::vector<Assignment>& assignments)
{
    std::uint64_t excluded = 0;
    std::uint64_t changedPrimary = 0;
    std::uint64_t anyChanged = 0;
    std::uint64_t compared = 0;
    double jaccard = 0;
    for (const Assignment& assignment : assignments) {
        if (assignment.isHead && config.excludeHead) {
            ++excluded;
            continue;
        }
        ++compared;
        jaccard += Jaccard(assignment.oldHeads, assignment.newHeads);
        const SizeType oldPrimary = assignment.oldHeads.empty() ? -1 : assignment.oldHeads.front();
        const SizeType newPrimary = assignment.newHeads.empty() ? -1 : assignment.newHeads.front();
        changedPrimary += oldPrimary != newPrimary;
        anyChanged += !SameReplicaSet(assignment.oldHeads, assignment.newHeads);
    }
    output << "{\"type\":\"posting_assignment_simulation_assignment_summary\","
           << "\"case\":\"" << (options.gtMode ? "gt_rank_pairs" : "uniform_spread") << "\","
           << "\"unique_selected_vid_count\":" << assignments.size() << ","
           << "\"excluded_head_vector_count\":" << excluded << ","
           << "\"posting_assigned_vector_count\":" << compared << ","
           << "\"changed_primary_count\":" << changedPrimary << ","
           << "\"any_replica_set_changed_count\":" << anyChanged << ","
           << "\"mean_replica_set_jaccard\":" << (compared == 0 ? 1.0 : jaccard / compared) << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!ParseArgs(argc, argv, options)) {
        Usage(argv[0]);
        return 2;
    }
    Config config;
    BaseInfo base;
    if (!ReadConfig(options, config) || !SameVectorFile(options.newHead, options.oldHead) ||
        !ReadBaseInfo(options.base, config.dimension, base)) return 1;
    if (options.uniformMode && options.uniformCount > base.count) {
        std::cerr << "--uniform-count exceeds base vector count\n";
        return 1;
    }

    std::shared_ptr<UInt8BKT> newIndex;
    std::shared_ptr<UInt8BKT> oldIndex;
    if (!LoadHead(options.newHead, config.dimension, config.buildMaxCheck, newIndex) ||
        !LoadHead(options.oldHead, config.dimension, config.buildMaxCheck, oldIndex) ||
        oldIndex->GetNumSamples() != newIndex->GetNumSamples()) {
        std::cerr << "Old/new HeadIndex sample counts do not match\n";
        return 1;
    }

    std::vector<Pair> pairs;
    ExactHeads exact;
    std::vector<std::uint64_t> uniqueVids;
    if (options.gtMode) {
        std::size_t queryTotal = 0;
        if (!ReadQueryInfo(options.query, config.dimension, queryTotal) ||
            options.queryOffset > queryTotal || options.queryCount > queryTotal - options.queryOffset ||
            !ReadGTPairs(options, queryTotal, pairs) ||
            !ReadExactHeads(options, newIndex->GetNumSamples(), exact)) return 1;
        std::unordered_set<std::uint64_t> seen;
        for (const Pair& pair : pairs) if (seen.insert(pair.vid).second) uniqueVids.push_back(pair.vid);
    } else {
        uniqueVids = UniformSpread(options.uniformCount, base.count, options.uniformSeed);
    }

    std::unordered_map<std::uint64_t, SizeType> selectedHeads;
    std::vector<std::uint8_t> values;
    if (!ReverseMapSelectedHeads(options.headIDs, newIndex->GetNumSamples(), uniqueVids, selectedHeads) ||
        !ReadSelectedBase(options.base, base, uniqueVids, values)) return 1;

    std::vector<Assignment> assignments;
    if (!SimulateAssignments(config, uniqueVids, values, selectedHeads, *oldIndex, *newIndex, assignments)) return 1;
    std::unordered_map<std::uint64_t, std::size_t> byVid;
    for (std::size_t i = 0; i < assignments.size(); ++i) byVid.emplace(assignments[i].vid, i);

    std::ofstream file;
    std::ostream* output = &std::cout;
    if (!options.output.empty()) {
        file.open(options.output, std::ios::out | std::ios::trunc);
        if (!file) {
            std::cerr << "Cannot open JSONL output\n";
            return 1;
        }
        output = &file;
    }
    WriteConfig(*output, options, config, base.count, static_cast<std::size_t>(newIndex->GetNumSamples()));
    WriteAssignments(*output, config, assignments);
    WriteAssignmentSummary(*output, options, config, assignments);
    if (options.gtMode) WritePairsAndSummary(*output, options, pairs, exact, byVid, config, assignments);
    if (!*output) {
        std::cerr << "Cannot write JSONL output\n";
        return 1;
    }
    return 0;
}
