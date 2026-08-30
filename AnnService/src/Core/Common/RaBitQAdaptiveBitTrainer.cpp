// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/RaBitQAdaptiveBitTrainer.h"

#include "inc/Core/Common/IQuantizer.h"
#include "inc/Core/Common/RaBitQQuantizer.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/CommonHelper.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace SPTAG
{
namespace COMMON
{
namespace
{

constexpr std::uint64_t kArtifactMagic = 0x3141545142524152ULL; // RARBQTA1
constexpr std::uint64_t kResultMagic = 0x3153455251424152ULL; // RARBQRES1
constexpr std::uint32_t kArtifactVersion = 1U;
constexpr std::uint32_t kResultVersion = 3U;
constexpr std::uint64_t kHashOffset = 14695981039346656037ULL;
constexpr std::uint64_t kHashPrime = 1099511628211ULL;

#pragma pack(push, 1)
struct ArtifactHeader
{
    std::uint64_t magic = kArtifactMagic;
    std::uint32_t version = kArtifactVersion;
    std::uint32_t headerBytes = sizeof(ArtifactHeader);
    std::int32_t dimension = 0;
    std::uint32_t metric = 0;
    std::uint64_t queryCount = 0;
    std::uint64_t truthDepth = 0;
    std::uint64_t baseCount = 0;
    std::uint64_t candidateCount = 0;
    std::uint64_t payloadBytes = 0;
    std::uint64_t fingerprint = 0;
};

struct ResultHeader
{
    std::uint64_t magic = kResultMagic;
    std::uint32_t version = kResultVersion;
    std::uint32_t headerBytes = sizeof(ResultHeader);
    std::int32_t dimension = 0;
    std::uint32_t metric = 0;
    std::uint64_t queryCount = 0;
    std::uint64_t truthDepth = 0;
    std::uint64_t baseCount = 0;
    std::uint64_t artifactFingerprint = 0;
    std::int32_t recallAt = 0;
    float targetRecallError = 0.0F;
    std::int32_t selectedBits = 0;
    std::uint32_t metricCount = 0;
    std::uint64_t modelBytes = 0;
    std::uint64_t modelFingerprint = 0;
    std::uint64_t fingerprint = 0;
};

struct MetricRecord
{
    std::int32_t bits = 0;
    float measuredRecall = 0.0F;
    float certifiedRecallLowerBound = 0.0F;
};
#pragma pack(pop)

struct TrainingData
{
    DimensionType dimension = 0;
    DistCalcMethod metric = DistCalcMethod::Undefined;
    SizeType baseCount = 0;
    int queryCount = 0;
    int truthDepth = 0;
    std::vector<float> queries;
    std::vector<SizeType> truth;
    std::vector<SizeType> candidateIDs;
    std::vector<float> candidates;
    std::uint64_t fingerprint = 0;
};

template <typename T>
void HashValue(std::uint64_t& p_hash, const T& p_value)
{
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&p_value);
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        p_hash ^= bytes[i];
        p_hash *= kHashPrime;
    }
}

void HashBytes(std::uint64_t& p_hash, const void* p_data, std::size_t p_bytes)
{
    const auto* bytes = static_cast<const std::uint8_t*>(p_data);
    for (std::size_t i = 0; i < p_bytes; ++i) {
        p_hash ^= bytes[i];
        p_hash *= kHashPrime;
    }
}

bool CheckedMultiply(std::uint64_t p_left,
                     std::uint64_t p_right,
                     std::uint64_t& p_result)
{
    if (p_left != 0 && p_right > (std::numeric_limits<std::uint64_t>::max)() / p_left) {
        return false;
    }
    p_result = p_left * p_right;
    return true;
}

bool CheckedAdd(std::uint64_t p_left,
                std::uint64_t p_right,
                std::uint64_t& p_result)
{
    if (p_right > (std::numeric_limits<std::uint64_t>::max)() - p_left) {
        return false;
    }
    p_result = p_left + p_right;
    return true;
}

bool ReadExact(std::ifstream& p_input, void* p_data, std::uint64_t p_bytes)
{
    if (p_bytes > static_cast<std::uint64_t>((std::numeric_limits<std::streamsize>::max)())) {
        return false;
    }
    p_input.read(static_cast<char*>(p_data), static_cast<std::streamsize>(p_bytes));
    return p_input.good();
}

bool WriteExact(std::ofstream& p_output, const void* p_data, std::uint64_t p_bytes)
{
    if (p_bytes > static_cast<std::uint64_t>((std::numeric_limits<std::streamsize>::max)())) {
        return false;
    }
    p_output.write(static_cast<const char*>(p_data), static_cast<std::streamsize>(p_bytes));
    return p_output.good();
}

std::string PrimaryPath(const std::string& p_paths)
{
    const std::size_t separator = p_paths.find(',');
    return separator == std::string::npos ? p_paths : p_paths.substr(0, separator);
}

std::vector<std::string> SplitPaths(const std::string& p_paths)
{
    std::vector<std::string> paths;
    std::size_t begin = 0;
    while (begin <= p_paths.size()) {
        const std::size_t separator = p_paths.find(',', begin);
        const std::size_t end =
            separator == std::string::npos ? p_paths.size() : separator;
        if (end > begin) paths.push_back(p_paths.substr(begin, end - begin));
        if (separator == std::string::npos) break;
        begin = separator + 1U;
    }
    return paths;
}

bool GetFileSize(const std::string& p_path, std::uint64_t& p_size)
{
    std::error_code ec;
    p_size = std::filesystem::file_size(p_path, ec);
    return !ec;
}

bool EnsureParentDirectory(const std::string& p_path)
{
    const std::filesystem::path parent = std::filesystem::path(p_path).parent_path();
    if (parent.empty()) {
        return true;
    }
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    return !ec;
}

template <typename Writer>
bool WriteAtomically(const std::string& p_path, Writer p_writer)
{
    if (!EnsureParentDirectory(p_path)) {
        return false;
    }
    const std::string temporary = p_path + ".writing";
    std::remove(temporary.c_str());
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output.good() || !p_writer(output)) {
            output.close();
            std::remove(temporary.c_str());
            return false;
        }
        output.flush();
        if (!output.good()) {
            output.close();
            std::remove(temporary.c_str());
            return false;
        }
    }

    std::error_code ec;
    std::filesystem::rename(temporary, p_path, ec);
    if (ec) {
        std::remove(temporary.c_str());
        return false;
    }
    return true;
}

std::uint64_t ComputeArtifactFingerprint(const TrainingData& p_data)
{
    std::uint64_t hash = kHashOffset;
    const std::int32_t dimension = p_data.dimension;
    const std::uint32_t metric = static_cast<std::uint32_t>(p_data.metric);
    const std::int64_t baseCount = p_data.baseCount;
    const std::int32_t queryCount = p_data.queryCount;
    const std::int32_t truthDepth = p_data.truthDepth;
    HashValue(hash, dimension);
    HashValue(hash, metric);
    HashValue(hash, baseCount);
    HashValue(hash, queryCount);
    HashValue(hash, truthDepth);
    const std::uint64_t candidateCount = p_data.candidateIDs.size();
    HashValue(hash, candidateCount);
    HashBytes(hash, p_data.queries.data(), p_data.queries.size() * sizeof(float));
    for (SizeType id : p_data.truth) {
        const std::int64_t storedID = id;
        HashValue(hash, storedID);
    }
    for (SizeType id : p_data.candidateIDs) {
        const std::int64_t storedID = id;
        HashValue(hash, storedID);
    }
    HashBytes(hash, p_data.candidates.data(), p_data.candidates.size() * sizeof(float));
    return hash;
}

std::uint64_t ComputeResultFingerprint(const ResultHeader& p_header,
                                       const std::vector<MetricRecord>& p_metrics)
{
    std::uint64_t hash = kHashOffset;
    HashValue(hash, p_header.dimension);
    HashValue(hash, p_header.metric);
    HashValue(hash, p_header.queryCount);
    HashValue(hash, p_header.truthDepth);
    HashValue(hash, p_header.baseCount);
    HashValue(hash, p_header.artifactFingerprint);
    HashValue(hash, p_header.recallAt);
    HashValue(hash, p_header.targetRecallError);
    HashValue(hash, p_header.selectedBits);
    HashValue(hash, p_header.metricCount);
    HashValue(hash, p_header.modelBytes);
    HashValue(hash, p_header.modelFingerprint);
    HashBytes(hash, p_metrics.data(), p_metrics.size() * sizeof(MetricRecord));
    return hash;
}

bool ComputeArtifactPayloadBytes(const ArtifactHeader& p_header,
                                 std::uint64_t& p_payloadBytes)
{
    std::uint64_t queryValues = 0;
    std::uint64_t truthValues = 0;
    std::uint64_t candidateValues = 0;
    std::uint64_t queryBytes = 0;
    std::uint64_t truthBytes = 0;
    std::uint64_t candidateIDBytes = 0;
    std::uint64_t candidateBytes = 0;
    std::uint64_t total = 0;
    return CheckedMultiply(p_header.queryCount, p_header.dimension, queryValues) &&
        CheckedMultiply(p_header.queryCount, p_header.truthDepth, truthValues) &&
        CheckedMultiply(p_header.candidateCount, p_header.dimension, candidateValues) &&
        CheckedMultiply(queryValues, sizeof(float), queryBytes) &&
        CheckedMultiply(truthValues, sizeof(std::int64_t), truthBytes) &&
        CheckedMultiply(p_header.candidateCount, sizeof(std::int64_t), candidateIDBytes) &&
        CheckedMultiply(candidateValues, sizeof(float), candidateBytes) &&
        CheckedAdd(queryBytes, truthBytes, total) &&
        CheckedAdd(total, candidateIDBytes, total) &&
        CheckedAdd(total, candidateBytes, p_payloadBytes);
}

bool LoadQueries(const RaBitQAdaptiveBitTrainer::Config& p_config,
                 std::vector<float>& p_queries)
{
    if (p_config.queryType == VectorFileType::TXT) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Adaptive RaBitQ calibration does not support TXT QueryType; use DEFAULT or XVEC.\n");
        return false;
    }
    const std::size_t valueCount =
        static_cast<std::size_t>(p_config.queryCount) *
        static_cast<std::size_t>(p_config.dimension);
    p_queries.assign(valueCount, 0.0F);
    if (p_config.queryType == VectorFileType::DEFAULT) {
        const std::string path = PrimaryPath(p_config.queryPath);
        std::ifstream input(path, std::ios::binary);
        SizeType rows = 0;
        DimensionType dimension = 0;
        std::uint64_t fileBytes = 0;
        std::uint64_t allValues = 0;
        std::uint64_t allValueBytes = 0;
        std::uint64_t expectedBytes = 0;
        if (!input.good() ||
            !ReadExact(input, &rows, sizeof(rows)) ||
            !ReadExact(input, &dimension, sizeof(dimension)) ||
            rows < p_config.queryCount || dimension != p_config.dimension ||
            !GetFileSize(path, fileBytes) ||
            !CheckedMultiply(
                static_cast<std::uint64_t>(rows),
                static_cast<std::uint64_t>(dimension),
                allValues) ||
            !CheckedMultiply(allValues, sizeof(float), allValueBytes) ||
            !CheckedAdd(
                sizeof(SizeType) + sizeof(DimensionType),
                allValueBytes,
                expectedBytes) ||
            fileBytes < expectedBytes ||
            !ReadExact(input, p_queries.data(), valueCount * sizeof(float))) {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "Invalid DEFAULT query file for adaptive RaBitQ calibration: %s.\n",
                path.c_str());
            return false;
        }
    } else if (p_config.queryType == VectorFileType::XVEC) {
        const std::uint64_t recordBytes =
            sizeof(std::int32_t) +
            static_cast<std::uint64_t>(p_config.dimension) * sizeof(float);
        int loaded = 0;
        for (const auto& path : SplitPaths(p_config.queryPath)) {
            std::uint64_t fileBytes = 0;
            if (!GetFileSize(path, fileBytes) || fileBytes % recordBytes != 0) {
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Error,
                    "Invalid XVEC query file for adaptive RaBitQ calibration: %s.\n",
                    path.c_str());
                return false;
            }
            std::ifstream input(path, std::ios::binary);
            const std::uint64_t rows = fileBytes / recordBytes;
            for (std::uint64_t row = 0;
                 row < rows && loaded < p_config.queryCount;
                 ++row, ++loaded) {
                std::int32_t dimension = 0;
                float* destination =
                    p_queries.data() +
                    static_cast<std::size_t>(loaded) *
                        static_cast<std::size_t>(p_config.dimension);
                if (!input.good() ||
                    !ReadExact(input, &dimension, sizeof(dimension)) ||
                    dimension != p_config.dimension ||
                    !ReadExact(
                        input,
                        destination,
                        static_cast<std::uint64_t>(p_config.dimension) *
                            sizeof(float))) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Invalid or truncated XVEC query file for adaptive RaBitQ calibration: %s.\n",
                        path.c_str());
                    return false;
                }
            }
            if (loaded == p_config.queryCount) break;
        }
        if (loaded != p_config.queryCount) {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "XVEC query files contain fewer than %d calibration queries.\n",
                p_config.queryCount);
            return false;
        }
    } else {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Adaptive RaBitQ calibration requires QueryType DEFAULT or XVEC.\n");
        return false;
    }

    for (float value : p_queries) {
        if (!std::isfinite(value)) {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "Adaptive RaBitQ query data contains non-finite values.\n");
            return false;
        }
    }
    return true;
}

bool LoadTruth(const RaBitQAdaptiveBitTrainer::Config& p_config,
               std::vector<SizeType>& p_truth)
{
    if (p_config.truthType == TruthFileType::TXT) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Adaptive RaBitQ calibration does not support TXT TruthType; use DEFAULT or XVEC.\n");
        return false;
    }
    if (p_config.truthPath.find(',') != std::string::npos) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Adaptive RaBitQ calibration requires a single truth file.\n");
        return false;
    }

    const std::string path = PrimaryPath(p_config.truthPath);
    std::ifstream input(path, std::ios::binary);
    if (!input.good()) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Failed to open adaptive RaBitQ truth file %s.\n",
            path.c_str());
        return false;
    }

    p_truth.assign(
        static_cast<std::size_t>(p_config.queryCount) *
            static_cast<std::size_t>(p_config.truthDepth),
        -1);
    if (p_config.truthType == TruthFileType::DEFAULT) {
        std::int32_t rows = 0;
        std::int32_t depth = 0;
        std::uint64_t fileBytes = 0;
        std::uint64_t idCount = 0;
        std::uint64_t idBytes = 0;
        std::uint64_t minimumBytes = 0;
        if (!ReadExact(input, &rows, sizeof(rows)) ||
            !ReadExact(input, &depth, sizeof(depth)) ||
            rows < p_config.queryCount || depth < p_config.truthDepth ||
            !GetFileSize(path, fileBytes) ||
            !CheckedMultiply(
                static_cast<std::uint64_t>(rows),
                static_cast<std::uint64_t>(depth),
                idCount) ||
            !CheckedMultiply(idCount, sizeof(std::int32_t), idBytes) ||
            !CheckedAdd(
                sizeof(std::int32_t) * 2U, idBytes, minimumBytes) ||
            fileBytes < minimumBytes) {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "DEFAULT truth file %s has fewer than %d queries or depth %d.\n",
                path.c_str(),
                p_config.queryCount,
                p_config.truthDepth);
            return false;
        }
        std::vector<std::int32_t> row(static_cast<std::size_t>(depth), -1);
        for (int query = 0; query < p_config.queryCount; ++query) {
            if (!ReadExact(
                    input,
                    row.data(),
                    static_cast<std::uint64_t>(row.size()) * sizeof(std::int32_t))) {
                return false;
            }
            for (int rank = 0; rank < p_config.truthDepth; ++rank) {
                p_truth[
                    static_cast<std::size_t>(query) * p_config.truthDepth + rank] =
                    static_cast<SizeType>(row[static_cast<std::size_t>(rank)]);
            }
        }
    } else if (p_config.truthType == TruthFileType::XVEC) {
        for (int query = 0; query < p_config.queryCount; ++query) {
            std::int32_t depth = 0;
            if (!ReadExact(input, &depth, sizeof(depth)) ||
                depth < p_config.truthDepth) {
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Error,
                    "XVEC truth file %s has fewer than %d neighbors for query %d.\n",
                    path.c_str(),
                    p_config.truthDepth,
                    query);
                return false;
            }
            const std::streamoff current = input.tellg();
            std::uint64_t fileBytes = 0;
            std::uint64_t rowBytes = 0;
            if (current < 0 || !GetFileSize(path, fileBytes) ||
                !CheckedMultiply(
                    static_cast<std::uint64_t>(depth),
                    sizeof(std::int32_t),
                    rowBytes) ||
                static_cast<std::uint64_t>(current) > fileBytes ||
                rowBytes > fileBytes - static_cast<std::uint64_t>(current)) {
                return false;
            }
            std::vector<std::int32_t> row(static_cast<std::size_t>(depth), -1);
            if (!ReadExact(
                    input,
                    row.data(),
                    static_cast<std::uint64_t>(row.size()) * sizeof(std::int32_t))) {
                return false;
            }
            for (int rank = 0; rank < p_config.truthDepth; ++rank) {
                p_truth[
                    static_cast<std::size_t>(query) * p_config.truthDepth + rank] =
                    static_cast<SizeType>(row[static_cast<std::size_t>(rank)]);
            }
        }
    } else {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Adaptive RaBitQ calibration requires TruthType DEFAULT or XVEC.\n");
        return false;
    }
    return true;
}

bool ReadBaseCandidates(const RaBitQAdaptiveBitTrainer::Config& p_config,
                        const std::shared_ptr<VectorSet>& p_memoryBase,
                        const std::vector<SizeType>& p_ids,
                        SizeType& p_baseCount,
                        std::vector<float>& p_vectors)
{
    const std::size_t dimension = static_cast<std::size_t>(p_config.dimension);
    p_vectors.assign(p_ids.size() * dimension, 0.0F);
    if (p_memoryBase != nullptr) {
        if (p_memoryBase->GetValueType() != VectorValueType::Float ||
            p_memoryBase->Dimension() != p_config.dimension ||
            p_memoryBase->Count() <= 0 ||
            (p_config.baseCount > 0 &&
             p_memoryBase->Count() != p_config.baseCount)) {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "Memory-backed adaptive RaBitQ calibration requires non-empty Float vectors with the configured dimension.\n");
            return false;
        }
        p_baseCount = p_memoryBase->Count();
        for (std::size_t i = 0; i < p_ids.size(); ++i) {
            if (p_ids[i] < 0 || p_ids[i] >= p_baseCount) {
                return false;
            }
            std::memcpy(
                p_vectors.data() + i * dimension,
                p_memoryBase->GetVector(p_ids[i]),
                dimension * sizeof(float));
        }
        return true;
    }

    if (p_config.baseType == VectorFileType::TXT) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Adaptive RaBitQ calibration does not support TXT VectorType; use DEFAULT or XVEC.\n");
        return false;
    }
    if (p_config.baseType == VectorFileType::DEFAULT) {
        const std::string path = PrimaryPath(p_config.basePath);
        std::ifstream input(path, std::ios::binary);
        SizeType rows = 0;
        DimensionType fileDimension = 0;
        std::uint64_t fileBytes = 0;
        std::uint64_t vectorBytes = 0;
        std::uint64_t expectedBytes = 0;
        if (!input.good() ||
            !ReadExact(input, &rows, sizeof(rows)) ||
            !ReadExact(input, &fileDimension, sizeof(fileDimension)) ||
            rows <= 0 || fileDimension != p_config.dimension ||
            !GetFileSize(path, fileBytes) ||
            !CheckedMultiply(
                static_cast<std::uint64_t>(rows),
                static_cast<std::uint64_t>(dimension) * sizeof(float),
                vectorBytes) ||
            !CheckedAdd(
                sizeof(SizeType) + sizeof(DimensionType),
                vectorBytes,
                expectedBytes) ||
            fileBytes < expectedBytes) {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "Invalid DEFAULT base-vector file for adaptive RaBitQ calibration: %s.\n",
                path.c_str());
            return false;
        }
        p_baseCount = rows;
        if (p_config.baseCount > 0 && p_config.baseCount != p_baseCount) {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "Adaptive RaBitQ base count mismatch: configured=%lld file=%lld.\n",
                static_cast<long long>(p_config.baseCount),
                static_cast<long long>(p_baseCount));
            return false;
        }
        const std::uint64_t recordBytes = dimension * sizeof(float);
        const std::uint64_t dataOffset =
            sizeof(SizeType) + sizeof(DimensionType);
        for (std::size_t i = 0; i < p_ids.size(); ++i) {
            const SizeType id = p_ids[i];
            if (id < 0 || id >= p_baseCount) {
                return false;
            }
            input.clear();
            input.seekg(
                static_cast<std::streamoff>(
                    dataOffset + static_cast<std::uint64_t>(id) * recordBytes),
                std::ios::beg);
            if (!input.good() ||
                !ReadExact(
                    input,
                    p_vectors.data() + i * dimension,
                    dimension * sizeof(float))) {
                return false;
            }
        }
        return true;
    }

    if (p_config.baseType != VectorFileType::XVEC) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Adaptive RaBitQ calibration requires VectorType DEFAULT or XVEC.\n");
        return false;
    }

    struct Shard
    {
        std::string path;
        std::uint64_t firstID;
        std::uint64_t count;
    };
    const std::uint64_t recordBytes =
        sizeof(std::int32_t) + dimension * sizeof(float);
    std::vector<Shard> shards;
    std::uint64_t totalCount = 0;
    for (const auto& path : SplitPaths(p_config.basePath)) {
        std::uint64_t fileBytes = 0;
        if (!GetFileSize(path, fileBytes) || fileBytes % recordBytes != 0) {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "Invalid XVEC base-vector file size for adaptive RaBitQ calibration: %s.\n",
                path.c_str());
            return false;
        }
        const std::uint64_t count = fileBytes / recordBytes;
        shards.push_back(Shard{path, totalCount, count});
        if (!CheckedAdd(totalCount, count, totalCount)) return false;
    }
    if (shards.empty() ||
        totalCount >
            static_cast<std::uint64_t>((std::numeric_limits<SizeType>::max)())) {
        return false;
    }
    p_baseCount = static_cast<SizeType>(totalCount);
    if (p_config.baseCount > 0 && p_config.baseCount != p_baseCount) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Adaptive RaBitQ base count mismatch: configured=%lld file=%lld.\n",
            static_cast<long long>(p_config.baseCount),
            static_cast<long long>(p_baseCount));
        return false;
    }

    std::size_t candidate = 0;
    for (const auto& shard : shards) {
        if (candidate >= p_ids.size()) break;
        const std::uint64_t shardEnd = shard.firstID + shard.count;
        if (static_cast<std::uint64_t>(p_ids[candidate]) >= shardEnd) continue;
        std::ifstream input(shard.path, std::ios::binary);
        if (!input.good()) return false;
        while (candidate < p_ids.size() &&
               static_cast<std::uint64_t>(p_ids[candidate]) < shardEnd) {
            const SizeType id = p_ids[candidate];
            if (id < 0 || static_cast<std::uint64_t>(id) < shard.firstID) {
                return false;
            }
            const std::uint64_t localID =
                static_cast<std::uint64_t>(id) - shard.firstID;
            input.clear();
            input.seekg(
                static_cast<std::streamoff>(localID * recordBytes),
                std::ios::beg);
            std::int32_t fileDimension = 0;
            if (!input.good() ||
                !ReadExact(input, &fileDimension, sizeof(fileDimension)) ||
                fileDimension != p_config.dimension ||
                !ReadExact(
                    input,
                    p_vectors.data() + candidate * dimension,
                    dimension * sizeof(float))) {
                return false;
            }
            ++candidate;
        }
    }
    return candidate == p_ids.size();
}

double RawDistance(const float* p_query,
                   const float* p_candidate,
                   DimensionType p_dimension,
                   DistCalcMethod p_metric)
{
    double sum = 0.0;
    double queryNorm = 0.0;
    double candidateNorm = 0.0;
    for (DimensionType dim = 0; dim < p_dimension; ++dim) {
        const double query = p_query[dim];
        const double candidate = p_candidate[dim];
        if (p_metric == DistCalcMethod::L2) {
            const double difference = query - candidate;
            sum += difference * difference;
        } else {
            sum += query * candidate;
            queryNorm += query * query;
            candidateNorm += candidate * candidate;
        }
    }
    if (p_metric == DistCalcMethod::L2 || p_metric == DistCalcMethod::InnerProduct) {
        return p_metric == DistCalcMethod::L2 ? sum : -sum;
    }
    if (queryNorm == 0.0 || candidateNorm == 0.0) {
        return 0.0;
    }
    return -sum / std::sqrt(queryNorm * candidateNorm);
}

bool ValidateTrainingData(const TrainingData& p_data)
{
    if (p_data.dimension <= 0 || p_data.baseCount <= 0 ||
        p_data.queryCount <= 0 || p_data.truthDepth <= 0 ||
        p_data.queries.size() !=
            static_cast<std::size_t>(p_data.queryCount) * p_data.dimension ||
        p_data.truth.size() !=
            static_cast<std::size_t>(p_data.queryCount) * p_data.truthDepth ||
        p_data.candidates.size() !=
            p_data.candidateIDs.size() * static_cast<std::size_t>(p_data.dimension)) {
        return false;
    }

    std::unordered_map<SizeType, std::size_t> candidateLookup;
    candidateLookup.reserve(p_data.candidateIDs.size());
    for (std::size_t i = 0; i < p_data.candidateIDs.size(); ++i) {
        const SizeType id = p_data.candidateIDs[i];
        if (id < 0 || id >= p_data.baseCount ||
            !candidateLookup.emplace(id, i).second) {
            return false;
        }
    }
    for (float value : p_data.queries) {
        if (!std::isfinite(value)) return false;
    }
    for (float value : p_data.candidates) {
        if (!std::isfinite(value)) return false;
    }

    const std::size_t dimension = static_cast<std::size_t>(p_data.dimension);
    for (int query = 0; query < p_data.queryCount; ++query) {
        std::unordered_set<SizeType> seen;
        seen.reserve(static_cast<std::size_t>(p_data.truthDepth));
        double previousDistance = -std::numeric_limits<double>::infinity();
        const float* queryVector =
            p_data.queries.data() + static_cast<std::size_t>(query) * dimension;
        for (int rank = 0; rank < p_data.truthDepth; ++rank) {
            const SizeType id =
                p_data.truth[
                    static_cast<std::size_t>(query) * p_data.truthDepth + rank];
            const auto candidate = candidateLookup.find(id);
            if (candidate == candidateLookup.end() || !seen.insert(id).second) {
                return false;
            }
            const float* candidateVector =
                p_data.candidates.data() + candidate->second * dimension;
            const double distance = RawDistance(
                queryVector, candidateVector, p_data.dimension, p_data.metric);
            const double tolerance =
                1e-5 * std::max(1.0, std::fabs(previousDistance));
            if (rank > 0 && distance + tolerance < previousDistance) {
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Error,
                    "Adaptive RaBitQ truth is not ordered by exact raw distance at query %d rank %d.\n",
                    query,
                    rank);
                return false;
            }
            previousDistance = distance;
        }
    }
    return true;
}

bool GenerateTrainingData(const RaBitQAdaptiveBitTrainer::Config& p_config,
                          const std::shared_ptr<VectorSet>& p_memoryBase,
                          TrainingData& p_data)
{
    p_data.dimension = p_config.dimension;
    p_data.metric = p_config.metric;
    p_data.queryCount = p_config.queryCount;
    p_data.truthDepth = p_config.truthDepth;
    if (!LoadQueries(p_config, p_data.queries) ||
        !LoadTruth(p_config, p_data.truth)) {
        return false;
    }

    std::set<SizeType> uniqueIDs;
    for (SizeType id : p_data.truth) {
        if (id < 0) {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "Adaptive RaBitQ truth contains a negative vector ID.\n");
            return false;
        }
        uniqueIDs.insert(id);
    }
    p_data.candidateIDs.assign(uniqueIDs.begin(), uniqueIDs.end());
    if (!ReadBaseCandidates(
            p_config,
            p_memoryBase,
            p_data.candidateIDs,
            p_data.baseCount,
            p_data.candidates) ||
        !ValidateTrainingData(p_data)) {
        return false;
    }
    p_data.fingerprint = ComputeArtifactFingerprint(p_data);
    return true;
}

bool SaveTrainingData(const std::string& p_path, const TrainingData& p_data)
{
    ArtifactHeader header;
    header.dimension = p_data.dimension;
    header.metric = static_cast<std::uint32_t>(p_data.metric);
    header.queryCount = static_cast<std::uint64_t>(p_data.queryCount);
    header.truthDepth = static_cast<std::uint64_t>(p_data.truthDepth);
    header.baseCount = static_cast<std::uint64_t>(p_data.baseCount);
    header.candidateCount = p_data.candidateIDs.size();
    header.fingerprint = p_data.fingerprint;
    if (!ComputeArtifactPayloadBytes(header, header.payloadBytes)) {
        return false;
    }

    std::vector<std::int64_t> truth(p_data.truth.begin(), p_data.truth.end());
    std::vector<std::int64_t> candidateIDs(
        p_data.candidateIDs.begin(), p_data.candidateIDs.end());
    return WriteAtomically(
        p_path,
        [&](std::ofstream& output) {
            return WriteExact(output, &header, sizeof(header)) &&
                WriteExact(
                    output,
                    p_data.queries.data(),
                    p_data.queries.size() * sizeof(float)) &&
                WriteExact(output, truth.data(), truth.size() * sizeof(std::int64_t)) &&
                WriteExact(
                    output,
                    candidateIDs.data(),
                    candidateIDs.size() * sizeof(std::int64_t)) &&
                WriteExact(
                    output,
                    p_data.candidates.data(),
                    p_data.candidates.size() * sizeof(float));
        });
}

bool ValidateArtifactHeader(
    const RaBitQAdaptiveBitTrainer::Config& p_config,
    const ArtifactHeader& p_header,
    std::uint64_t p_fileBytes)
{
    std::uint64_t payloadBytes = 0;
    std::uint64_t maximumCandidateCount = 0;
    return p_header.magic == kArtifactMagic &&
        p_header.version == kArtifactVersion &&
        p_header.headerBytes == sizeof(ArtifactHeader) &&
        p_header.dimension == p_config.dimension &&
        p_header.metric == static_cast<std::uint32_t>(p_config.metric) &&
        p_header.queryCount == static_cast<std::uint64_t>(p_config.queryCount) &&
        p_header.truthDepth == static_cast<std::uint64_t>(p_config.truthDepth) &&
        p_header.baseCount > 0 &&
        (p_config.baseCount <= 0 ||
         p_header.baseCount == static_cast<std::uint64_t>(p_config.baseCount)) &&
        p_header.candidateCount > 0 &&
        ComputeArtifactPayloadBytes(p_header, payloadBytes) &&
        CheckedMultiply(
            p_header.queryCount,
            p_header.truthDepth,
            maximumCandidateCount) &&
        p_header.candidateCount <= maximumCandidateCount &&
        payloadBytes == p_header.payloadBytes &&
        p_fileBytes == sizeof(ArtifactHeader) + payloadBytes &&
        p_header.queryCount <=
            static_cast<std::uint64_t>((std::numeric_limits<int>::max)()) &&
        p_header.truthDepth <=
            static_cast<std::uint64_t>((std::numeric_limits<int>::max)()) &&
        p_header.baseCount <=
            static_cast<std::uint64_t>((std::numeric_limits<SizeType>::max)()) &&
        p_header.candidateCount <=
            static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)());
}

bool LoadArtifactHeader(
    const RaBitQAdaptiveBitTrainer::Config& p_config,
    ArtifactHeader& p_header)
{
    std::uint64_t fileBytes = 0;
    if (!GetFileSize(p_config.trainingDataFile, fileBytes) ||
        fileBytes < sizeof(ArtifactHeader)) {
        return false;
    }
    std::ifstream input(p_config.trainingDataFile, std::ios::binary);
    return input.good() &&
        ReadExact(input, &p_header, sizeof(p_header)) &&
        ValidateArtifactHeader(p_config, p_header, fileBytes);
}

bool LoadTrainingData(const RaBitQAdaptiveBitTrainer::Config& p_config,
                      TrainingData& p_data)
{
    ArtifactHeader header;
    if (!LoadArtifactHeader(p_config, header)) {
        return false;
    }

    std::ifstream input(p_config.trainingDataFile, std::ios::binary);
    if (!input.good()) {
        return false;
    }
    input.seekg(static_cast<std::streamoff>(sizeof(ArtifactHeader)));
    if (!input.good()) return false;

    p_data.dimension = header.dimension;
    p_data.metric = static_cast<DistCalcMethod>(header.metric);
    p_data.baseCount = static_cast<SizeType>(header.baseCount);
    p_data.queryCount = static_cast<int>(header.queryCount);
    p_data.truthDepth = static_cast<int>(header.truthDepth);
    p_data.queries.resize(
        static_cast<std::size_t>(p_data.queryCount) * p_data.dimension);
    std::vector<std::int64_t> truth(
        static_cast<std::size_t>(p_data.queryCount) * p_data.truthDepth);
    std::vector<std::int64_t> candidateIDs(
        static_cast<std::size_t>(header.candidateCount));
    p_data.candidates.resize(
        static_cast<std::size_t>(header.candidateCount) * p_data.dimension);
    if (!ReadExact(
            input,
            p_data.queries.data(),
            p_data.queries.size() * sizeof(float)) ||
        !ReadExact(input, truth.data(), truth.size() * sizeof(std::int64_t)) ||
        !ReadExact(
            input,
            candidateIDs.data(),
            candidateIDs.size() * sizeof(std::int64_t)) ||
        !ReadExact(
            input,
            p_data.candidates.data(),
            p_data.candidates.size() * sizeof(float))) {
        return false;
    }

    p_data.truth.resize(truth.size());
    for (std::size_t i = 0; i < truth.size(); ++i) {
        p_data.truth[i] = static_cast<SizeType>(truth[i]);
        if (static_cast<std::int64_t>(p_data.truth[i]) != truth[i]) return false;
    }
    p_data.candidateIDs.resize(candidateIDs.size());
    for (std::size_t i = 0; i < candidateIDs.size(); ++i) {
        p_data.candidateIDs[i] = static_cast<SizeType>(candidateIDs[i]);
        if (static_cast<std::int64_t>(p_data.candidateIDs[i]) != candidateIDs[i]) {
            return false;
        }
    }
    p_data.fingerprint = ComputeArtifactFingerprint(p_data);
    return p_data.fingerprint == header.fingerprint && ValidateTrainingData(p_data);
}

bool ReadFileFingerprint(const std::string& p_path,
                         std::uint64_t& p_fileBytes,
                         std::uint64_t& p_fingerprint)
{
    if (!GetFileSize(p_path, p_fileBytes) || p_fileBytes == 0 ||
        p_fileBytes >
            static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)())) {
        return false;
    }
    std::ifstream input(p_path, std::ios::binary);
    if (!input.good()) return false;
    std::vector<char> buffer(1U << 20U);
    std::uint64_t remaining = p_fileBytes;
    p_fingerprint = kHashOffset;
    while (remaining > 0) {
        const std::size_t block = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, buffer.size()));
        if (!ReadExact(input, buffer.data(), block)) return false;
        HashBytes(p_fingerprint, buffer.data(), block);
        remaining -= block;
    }
    return true;
}

bool SaveModelAtomically(const std::string& p_path,
                         const std::shared_ptr<RaBitQQuantizer>& p_quantizer)
{
    if (!EnsureParentDirectory(p_path)) return false;
    const std::string temporary = p_path + ".writing";
    std::remove(temporary.c_str());
    auto output = f_createIO();
    if (output == nullptr ||
        !output->Initialize(temporary.c_str(), std::ios::binary | std::ios::out) ||
        p_quantizer->SaveQuantizer(output) != ErrorCode::Success) {
        if (output != nullptr) output->ShutDown();
        std::remove(temporary.c_str());
        return false;
    }
    output->ShutDown();
    std::error_code ec;
    if (std::filesystem::exists(p_path, ec)) {
        if (ec || !std::filesystem::remove(p_path, ec) || ec) {
            std::remove(temporary.c_str());
            return false;
        }
    }
    std::filesystem::rename(temporary, p_path, ec);
    if (ec) {
        std::remove(temporary.c_str());
        return false;
    }
    return true;
}

bool LoadAndValidateModel(const RaBitQAdaptiveBitTrainer::Config& p_config,
                          const ResultHeader& p_header)
{
    std::uint64_t modelBytes = 0;
    std::uint64_t modelFingerprint = 0;
    if (!ReadFileFingerprint(
            p_config.modelFile, modelBytes, modelFingerprint) ||
        modelBytes != p_header.modelBytes ||
        modelFingerprint != p_header.modelFingerprint) {
        return false;
    }

    auto input = f_createIO();
    if (input == nullptr ||
        !input->Initialize(
            p_config.modelFile.c_str(), std::ios::binary | std::ios::in)) {
        return false;
    }
    auto quantizer = IQuantizer::LoadIQuantizer(input);
    input->ShutDown();
    auto rabitq = std::dynamic_pointer_cast<RaBitQQuantizer>(quantizer);
    return rabitq != nullptr && rabitq->Ready() &&
        rabitq->Dimension() == p_config.dimension &&
        rabitq->GetMetric() == p_config.metric &&
        rabitq->IsNormalizationEnabled() ==
            (p_config.metric == DistCalcMethod::Cosine) &&
        rabitq->GetQuantizationMode() ==
            RaBitQQuantizer::QuantizationMode::Exact &&
        rabitq->Bits() == p_header.selectedBits;
}

bool SaveResult(const RaBitQAdaptiveBitTrainer::Config& p_config,
                const TrainingData& p_data,
                std::uint64_t p_modelBytes,
                const RaBitQAdaptiveBitTrainer::Result& p_result)
{
    ResultHeader header;
    header.dimension = p_data.dimension;
    header.metric = static_cast<std::uint32_t>(p_data.metric);
    header.queryCount = static_cast<std::uint64_t>(p_data.queryCount);
    header.truthDepth = static_cast<std::uint64_t>(p_data.truthDepth);
    header.baseCount = static_cast<std::uint64_t>(p_data.baseCount);
    header.artifactFingerprint = p_data.fingerprint;
    header.recallAt = p_config.recallAt;
    header.targetRecallError = p_config.targetRecallError;
    header.selectedBits = p_result.selectedBits;
    header.metricCount = static_cast<std::uint32_t>(p_result.metrics.size());
    header.modelBytes = p_modelBytes;
    header.modelFingerprint = p_result.modelFingerprint;

    std::vector<MetricRecord> metrics;
    metrics.reserve(p_result.metrics.size());
    for (const auto& metric : p_result.metrics) {
        metrics.push_back(MetricRecord{
            metric.bits,
            metric.measuredRecall,
            metric.certifiedRecallLowerBound,
        });
    }
    header.fingerprint = ComputeResultFingerprint(header, metrics);
    return WriteAtomically(
        p_config.trainingResultFile,
        [&](std::ofstream& output) {
            return WriteExact(output, &header, sizeof(header)) &&
                WriteExact(
                    output, metrics.data(), metrics.size() * sizeof(MetricRecord));
        });
}

enum class TargetDecision
{
    BelowTarget,
    MeetsTarget,
};

TargetDecision DecideTarget(
    const RaBitQAdaptiveBitTrainer::BitMetrics& p_metric,
    float p_targetRecall)
{
    if (p_metric.certifiedRecallLowerBound >= p_targetRecall) {
        return TargetDecision::MeetsTarget;
    }
    return p_metric.measuredRecall >= p_targetRecall
        ? TargetDecision::MeetsTarget
        : TargetDecision::BelowTarget;
}

bool ValidateSparseMetrics(
    const std::vector<RaBitQAdaptiveBitTrainer::BitMetrics>& p_metrics,
    int p_selectedBits,
    float p_targetRecall)
{
    std::map<int, TargetDecision> decisions;
    for (const auto& metric : p_metrics) {
        decisions.emplace(metric.bits, DecideTarget(metric, p_targetRecall));
    }

    std::set<int> expectedBits;
    const auto decisionAt = [&](int p_bits, TargetDecision& p_decision) {
        const auto found = decisions.find(p_bits);
        if (found == decisions.end()) return false;
        expectedBits.insert(p_bits);
        p_decision = found->second;
        return true;
    };

    TargetDecision decision;
    if (!decisionAt(1, decision)) return false;
    if (decision == TargetDecision::MeetsTarget) {
        return p_selectedBits == 1 &&
            expectedBits.size() == p_metrics.size();
    }

    if (!decisionAt(8, decision) ||
        decision != TargetDecision::MeetsTarget) {
        return false;
    }

    int lowerFailingBits = 1;
    int upperMeetingBits = 8;
    while (upperMeetingBits - lowerFailingBits > 1) {
        const int bits =
            lowerFailingBits + (upperMeetingBits - lowerFailingBits) / 2;
        if (!decisionAt(bits, decision)) return false;
        if (decision == TargetDecision::MeetsTarget) {
            upperMeetingBits = bits;
        } else {
            lowerFailingBits = bits;
        }
    }
    return p_selectedBits == upperMeetingBits &&
        expectedBits.size() == p_metrics.size();
}

bool LoadResult(const RaBitQAdaptiveBitTrainer::Config& p_config,
                bool p_artifactExists,
                RaBitQAdaptiveBitTrainer::Result& p_result)
{
    std::uint64_t fileBytes = 0;
    if (!GetFileSize(p_config.trainingResultFile, fileBytes) ||
        fileBytes < sizeof(ResultHeader)) {
        return false;
    }
    std::ifstream input(p_config.trainingResultFile, std::ios::binary);
    ResultHeader header;
    if (!input.good() || !ReadExact(input, &header, sizeof(header)) ||
        header.magic != kResultMagic ||
        header.version != kResultVersion ||
        header.headerBytes != sizeof(ResultHeader) ||
        header.dimension != p_config.dimension ||
        header.metric != static_cast<std::uint32_t>(p_config.metric) ||
        header.queryCount != static_cast<std::uint64_t>(p_config.queryCount) ||
        header.truthDepth != static_cast<std::uint64_t>(p_config.truthDepth) ||
        header.baseCount == 0 ||
        (p_config.baseCount > 0 &&
         header.baseCount != static_cast<std::uint64_t>(p_config.baseCount)) ||
        header.recallAt != p_config.recallAt ||
        header.targetRecallError != p_config.targetRecallError ||
        header.selectedBits < 1 || header.selectedBits > 8 ||
        header.metricCount == 0 || header.metricCount > 8 ||
        fileBytes !=
            sizeof(ResultHeader) +
                static_cast<std::uint64_t>(header.metricCount) *
                    sizeof(MetricRecord)) {
        return false;
    }

    std::vector<MetricRecord> metrics(header.metricCount);
    if (!ReadExact(
            input, metrics.data(), metrics.size() * sizeof(MetricRecord)) ||
        ComputeResultFingerprint(header, metrics) != header.fingerprint ||
        !LoadAndValidateModel(p_config, header)) {
        return false;
    }
    if (p_artifactExists) {
        ArtifactHeader artifactHeader;
        if (!LoadArtifactHeader(p_config, artifactHeader) ||
            artifactHeader.dimension != header.dimension ||
            artifactHeader.metric != header.metric ||
            artifactHeader.queryCount != header.queryCount ||
            artifactHeader.truthDepth != header.truthDepth ||
            artifactHeader.baseCount != header.baseCount ||
            artifactHeader.fingerprint != header.artifactFingerprint) {
            return false;
        }
    }

    p_result = RaBitQAdaptiveBitTrainer::Result();
    p_result.selectedBits = header.selectedBits;
    p_result.artifactFingerprint = header.artifactFingerprint;
    p_result.modelFingerprint = header.modelFingerprint;
    p_result.reused = true;
    const float targetRecall = 1.0F - p_config.targetRecallError;
    for (std::size_t i = 0; i < metrics.size(); ++i) {
        const auto& metric = metrics[i];
        if (metric.bits < 1 || metric.bits > 8 ||
            (i > 0 && metric.bits <= metrics[i - 1U].bits) ||
            !std::isfinite(metric.measuredRecall) ||
            !std::isfinite(metric.certifiedRecallLowerBound) ||
            metric.measuredRecall < 0.0F || metric.measuredRecall > 1.0F ||
            metric.certifiedRecallLowerBound < 0.0F ||
            metric.certifiedRecallLowerBound > metric.measuredRecall) {
            return false;
        }
        const RaBitQAdaptiveBitTrainer::BitMetrics loadedMetric{
            metric.bits,
            metric.measuredRecall,
            metric.certifiedRecallLowerBound,
        };
        p_result.metrics.push_back(loadedMetric);
    }
    return ValidateSparseMetrics(
        p_result.metrics, header.selectedBits, targetRecall);
}

RaBitQAdaptiveBitTrainer::BitMetrics EvaluateBits(
    const TrainingData& p_data,
    const std::shared_ptr<RaBitQQuantizer>& p_quantizer,
    int p_recallAt)
{
    const std::size_t candidateCount = p_data.candidateIDs.size();
    const std::size_t codeBytes =
        static_cast<std::size_t>(p_quantizer->GetNumSubvectors());
    std::vector<std::uint8_t> codes(candidateCount * codeBytes, 0);
    for (std::size_t candidate = 0; candidate < candidateCount; ++candidate) {
        p_quantizer->QuantizeVector(
            p_data.candidates.data() +
                candidate * static_cast<std::size_t>(p_data.dimension),
            codes.data() + candidate * codeBytes,
            false);
    }

    std::unordered_map<SizeType, std::size_t> candidateLookup;
    candidateLookup.reserve(candidateCount);
    for (std::size_t i = 0; i < candidateCount; ++i) {
        candidateLookup.emplace(p_data.candidateIDs[i], i);
    }

    p_quantizer->SetEnableADC(true);
    std::vector<std::uint8_t> queryCode(
        static_cast<std::size_t>(p_quantizer->QuantizeSize()), 0);
    double recallSum = 0.0;
    double lowerSum = 0.0;
    const int recallAt = p_recallAt;
    for (int query = 0; query < p_data.queryCount; ++query) {
        const float* queryVector =
            p_data.queries.data() +
            static_cast<std::size_t>(query) * p_data.dimension;
        p_quantizer->QuantizeVector(queryVector, queryCode.data(), true);

        struct Estimate
        {
            SizeType id;
            float distance;
            float lower;
            float upper;
        };
        std::vector<Estimate> estimates;
        estimates.reserve(static_cast<std::size_t>(p_data.truthDepth));
        for (int rank = 0; rank < p_data.truthDepth; ++rank) {
            const SizeType id =
                p_data.truth[
                    static_cast<std::size_t>(query) * p_data.truthDepth + rank];
            const std::size_t candidate = candidateLookup.find(id)->second;
            const auto estimate = p_quantizer->DistanceWithError(
                queryCode.data(), codes.data() + candidate * codeBytes);
            if (!std::isfinite(estimate.distance) ||
                !std::isfinite(estimate.errorBound)) {
                return RaBitQAdaptiveBitTrainer::BitMetrics{
                    p_quantizer->Bits(),
                    std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN(),
                };
            }
            const float error = std::max(0.0F, estimate.errorBound);
            estimates.push_back(Estimate{
                id,
                estimate.distance,
                estimate.distance - error,
                estimate.distance + error,
            });
        }

        std::vector<std::size_t> order(estimates.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(
            order.begin(),
            order.end(),
            [&](std::size_t left, std::size_t right) {
                if (estimates[left].distance != estimates[right].distance) {
                    return estimates[left].distance < estimates[right].distance;
                }
                return estimates[left].id < estimates[right].id;
            });
        std::unordered_set<SizeType> exactTop;
        exactTop.reserve(static_cast<std::size_t>(recallAt));
        for (int rank = 0; rank < recallAt; ++rank) {
            exactTop.insert(estimates[static_cast<std::size_t>(rank)].id);
        }
        int matches = 0;
        for (int rank = 0; rank < recallAt; ++rank) {
            if (exactTop.find(estimates[order[static_cast<std::size_t>(rank)]].id) !=
                exactTop.end()) {
                ++matches;
            }
        }
        recallSum += static_cast<double>(matches) / recallAt;

        std::vector<float> lowerEndpoints;
        lowerEndpoints.reserve(estimates.size());
        for (const auto& estimate : estimates) {
            lowerEndpoints.push_back(estimate.lower);
        }
        std::sort(lowerEndpoints.begin(), lowerEndpoints.end());
        int certified = 0;
        for (int rank = 0; rank < recallAt; ++rank) {
            const auto& estimate = estimates[static_cast<std::size_t>(rank)];
            const std::size_t endpointsAtOrBelowUpper =
                static_cast<std::size_t>(
                std::upper_bound(
                    lowerEndpoints.begin(), lowerEndpoints.end(), estimate.upper) -
                lowerEndpoints.begin());
            // The candidate's own lower endpoint is included exactly once.
            const std::size_t potentialCompetitors =
                endpointsAtOrBelowUpper - 1U;
            const std::size_t worstRank = 1U + potentialCompetitors;
            if (worstRank <= static_cast<std::size_t>(recallAt)) ++certified;
        }
        lowerSum += static_cast<double>(certified) / recallAt;
    }

    const double inverseQueries = 1.0 / p_data.queryCount;
    return RaBitQAdaptiveBitTrainer::BitMetrics{
        p_quantizer->Bits(),
        static_cast<float>(recallSum * inverseQueries),
        static_cast<float>(lowerSum * inverseQueries),
    };
}

} // namespace

ErrorCode RaBitQAdaptiveBitTrainer::Run(
    const Config& p_config,
    const std::shared_ptr<VectorSet>& p_memoryBase,
    Result& p_result)
{
    p_result = Result();
    if (p_config.dimension < 64 || p_config.dimension >= 4096 ||
        (p_config.metric != DistCalcMethod::L2 &&
         p_config.metric != DistCalcMethod::Cosine &&
         p_config.metric != DistCalcMethod::InnerProduct) ||
        p_config.queryCount <= 0 || p_config.truthDepth <= 0 ||
        p_config.recallAt <= 0 || p_config.recallAt > p_config.truthDepth ||
        !std::isfinite(p_config.targetRecallError) ||
        !(p_config.targetRecallError > 0.0F &&
          p_config.targetRecallError < 1.0F) ||
        p_config.trainingDataFile.empty() ||
        p_config.trainingResultFile.empty() || p_config.modelFile.empty() ||
        p_config.trainingDataFile == p_config.trainingResultFile ||
        p_config.trainingDataFile == p_config.modelFile ||
        p_config.trainingResultFile == p_config.modelFile) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Invalid adaptive RaBitQ calibration configuration.\n");
        return ErrorCode::FailedParseValue;
    }

    std::error_code existsError;
    const bool artifactExists =
        std::filesystem::exists(p_config.trainingDataFile, existsError);
    if (existsError) return ErrorCode::Fail;
    const bool resultExists =
        std::filesystem::exists(p_config.trainingResultFile, existsError);
    if (existsError) return ErrorCode::Fail;

    if (resultExists) {
        if (!LoadResult(p_config, artifactExists, p_result)) {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "Adaptive RaBitQ result or selected model is stale or corrupt.\n");
            return ErrorCode::Fail;
        }
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Info,
            "Reusing adaptive RaBitQ calibration: selected %d bits.\n",
            p_result.selectedBits);
        return ErrorCode::Success;
    }

    TrainingData data;
    if (artifactExists) {
        if (!LoadTrainingData(p_config, data)) {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "Adaptive RaBitQ training artifact is stale or corrupt: %s.\n",
                p_config.trainingDataFile.c_str());
            return ErrorCode::Fail;
        }
    } else {
        if (!GenerateTrainingData(p_config, p_memoryBase, data) ||
            !SaveTrainingData(p_config.trainingDataFile, data)) {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "Failed to generate adaptive RaBitQ training artifact %s.\n",
                p_config.trainingDataFile.c_str());
            return ErrorCode::Fail;
        }
    }

    ByteArray candidateBytes(
        reinterpret_cast<std::uint8_t*>(data.candidates.data()),
        data.candidates.size() * sizeof(float),
        false);
    auto candidateSet = std::make_shared<BasicVectorSet>(
        candidateBytes,
        VectorValueType::Float,
        data.dimension,
        static_cast<SizeType>(data.candidateIDs.size()));

    std::shared_ptr<RaBitQQuantizer> baseQuantizer;
    try {
        baseQuantizer = std::make_shared<RaBitQQuantizer>(
            data.dimension,
            1,
            data.metric == DistCalcMethod::Cosine,
            data.metric,
            RaBitQQuantizer::QuantizationMode::Exact);
    } catch (const std::invalid_argument&) {
        return ErrorCode::FailedParseValue;
    }
    if (baseQuantizer->SetDeterministicRotation(data.fingerprint) !=
            ErrorCode::Success ||
        baseQuantizer->Train(candidateSet) != ErrorCode::Success) {
        return ErrorCode::Fail;
    }

    const float targetRecall = 1.0F - p_config.targetRecallError;
    std::map<int, BitMetrics> evaluatedMetrics;
    auto evaluate = [&](int bits, TargetDecision& p_decision) {
        const auto cached = evaluatedMetrics.find(bits);
        if (cached != evaluatedMetrics.end()) {
            p_decision = DecideTarget(cached->second, targetRecall);
            return true;
        }
        auto quantizer = baseQuantizer->CloneWithBits(bits);
        if (quantizer == nullptr) return false;
        BitMetrics metrics = EvaluateBits(data, quantizer, p_config.recallAt);
        if (!std::isfinite(metrics.measuredRecall) ||
            !std::isfinite(metrics.certifiedRecallLowerBound) ||
            metrics.certifiedRecallLowerBound > metrics.measuredRecall) {
            return false;
        }
        const bool meetsByCertifiedLowerBound =
            metrics.certifiedRecallLowerBound >= targetRecall;
        p_decision = DecideTarget(metrics, targetRecall);
        evaluatedMetrics.emplace(bits, metrics);
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Info,
            "Adaptive RaBitQ %d-bit measuredRecall@%d=%.6f certifiedRecallLowerBound=%.6f target=%.6f decision=%s (%s).\n",
            bits,
            p_config.recallAt,
            metrics.measuredRecall,
            metrics.certifiedRecallLowerBound,
            targetRecall,
            p_decision == TargetDecision::MeetsTarget ? "meets" : "below",
            meetsByCertifiedLowerBound
                ? "certified recall lower bound"
                : "measured recall");
        return true;
    };

    int lowerFailingBits = 0;
    int upperMeetingBits = 0;
    TargetDecision decision;
    // Official bit-width quality is expected to be monotonic, so each endpoint
    // decision narrows the interval that still needs probing.
    if (!evaluate(1, decision)) return ErrorCode::Fail;
    if (decision == TargetDecision::MeetsTarget) {
        upperMeetingBits = 1;
    } else {
        lowerFailingBits = 1;
        if (!evaluate(8, decision)) return ErrorCode::Fail;
        if (decision == TargetDecision::MeetsTarget) {
            upperMeetingBits = 8;
        }
    }

    while (upperMeetingBits > 0 &&
           upperMeetingBits - lowerFailingBits > 1) {
        const int bits =
            lowerFailingBits + (upperMeetingBits - lowerFailingBits) / 2;
        if (!evaluate(bits, decision)) return ErrorCode::Fail;
        if (decision == TargetDecision::MeetsTarget) {
            upperMeetingBits = bits;
        } else {
            lowerFailingBits = bits;
        }
    }

    p_result.metrics.reserve(evaluatedMetrics.size());
    for (const auto& entry : evaluatedMetrics) {
        p_result.metrics.push_back(entry.second);
    }
    p_result.selectedBits = upperMeetingBits;

    if (p_result.selectedBits == 0) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "No official RaBitQ bit width in [1, 8] meets target Recall@%d %.6f.\n",
            p_config.recallAt,
            targetRecall);
        return ErrorCode::Fail;
    }
    if (!ValidateSparseMetrics(
            p_result.metrics, p_result.selectedBits, targetRecall)) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Adaptive RaBitQ failed to verify the selected bit-width boundary.\n");
        return ErrorCode::Fail;
    }

    auto selectedQuantizer =
        baseQuantizer->CloneWithBits(p_result.selectedBits);
    if (selectedQuantizer == nullptr) return ErrorCode::Fail;
    if (!SaveModelAtomically(p_config.modelFile, selectedQuantizer)) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Failed to atomically save adaptive RaBitQ model %s.\n",
            p_config.modelFile.c_str());
        return ErrorCode::Fail;
    }
    std::uint64_t modelBytes = 0;
    if (!ReadFileFingerprint(
            p_config.modelFile, modelBytes, p_result.modelFingerprint)) {
        return ErrorCode::Fail;
    }
    p_result.artifactFingerprint = data.fingerprint;
    if (!SaveResult(p_config, data, modelBytes, p_result)) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Failed to atomically save adaptive RaBitQ result %s.\n",
            p_config.trainingResultFile.c_str());
        return ErrorCode::Fail;
    }
    SPTAGLIB_LOG(
        Helper::LogLevel::LL_Info,
        "Adaptive RaBitQ selected %d bits for Recall@%d target %.6f.\n",
        p_result.selectedBits,
        p_config.recallAt,
        targetRecall);
    return ErrorCode::Success;
}

} // namespace COMMON
} // namespace SPTAG
