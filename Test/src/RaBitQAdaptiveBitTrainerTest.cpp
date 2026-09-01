// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"

#include "inc/Core/Common/IQuantizer.h"
#include "inc/Core/Common/RaBitQAdaptiveBitTrainer.h"
#include "inc/Core/Common/RaBitQQuantizer.h"
#include "inc/Core/VectorIndex.h"

#include <algorithm>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace SPTAG;

namespace
{

constexpr DimensionType kDimension = 64;
constexpr SizeType kBaseCount = 96;
constexpr int kQueryCount = 6;
constexpr int kTruthDepth = 32;
constexpr int kRecallAt = 5;
constexpr std::uint64_t kHashOffset = 14695981039346656037ULL;
constexpr std::uint64_t kHashPrime = 1099511628211ULL;

#pragma pack(push, 1)
struct StoredResultHeader
{
    std::uint64_t magic;
    std::uint32_t version;
    std::uint32_t headerBytes;
    std::int32_t dimension;
    std::uint32_t metric;
    std::uint64_t queryCount;
    std::uint64_t truthDepth;
    std::uint64_t baseCount;
    std::uint64_t artifactFingerprint;
    std::int32_t recallAt;
    float targetRecallError;
    std::int32_t selectedBits;
    std::uint32_t metricCount;
    std::uint64_t modelBytes;
    std::uint64_t modelFingerprint;
    std::uint64_t fingerprint;
};

struct StoredMetricRecord
{
    std::int32_t bits;
    float measuredRecall;
    float certifiedRecallLowerBound;
};
#pragma pack(pop)

template <typename T>
void HashValue(std::uint64_t& p_hash, const T& p_value)
{
    const auto* bytes =
        reinterpret_cast<const std::uint8_t*>(&p_value);
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        p_hash ^= bytes[i];
        p_hash *= kHashPrime;
    }
}

void HashBytes(std::uint64_t& p_hash,
               const void* p_data,
               std::size_t p_bytes)
{
    const auto* bytes = static_cast<const std::uint8_t*>(p_data);
    for (std::size_t i = 0; i < p_bytes; ++i) {
        p_hash ^= bytes[i];
        p_hash *= kHashPrime;
    }
}

std::uint64_t ComputeStoredResultFingerprint(
    const StoredResultHeader& p_header,
    const std::vector<StoredMetricRecord>& p_metrics)
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
    HashBytes(
        hash,
        p_metrics.data(),
        p_metrics.size() * sizeof(StoredMetricRecord));
    return hash;
}

bool AddUnexpectedSparseMetric(const std::string& p_path)
{
    std::ifstream input(p_path, std::ios::binary);
    StoredResultHeader header;
    if (!input.read(
            reinterpret_cast<char*>(&header), sizeof(header)) ||
        header.version != 3U ||
        header.headerBytes != sizeof(header) ||
        header.metricCount == 0U || header.metricCount >= 8U) {
        return false;
    }
    std::vector<StoredMetricRecord> metrics(header.metricCount);
    if (!input.read(
            reinterpret_cast<char*>(metrics.data()),
            static_cast<std::streamsize>(
                metrics.size() * sizeof(StoredMetricRecord)))) {
        return false;
    }

    std::vector<bool> evaluated(9, false);
    for (const auto& metric : metrics) {
        evaluated[static_cast<std::size_t>(metric.bits)] = true;
    }

    int sourceBits = header.selectedBits;
    int unexpectedBits = 0;
    if (header.selectedBits < 8) {
        for (int bits = header.selectedBits + 1; bits <= 8; ++bits) {
            if (!evaluated[static_cast<std::size_t>(bits)]) {
                unexpectedBits = bits;
                break;
            }
        }
    } else {
        sourceBits = header.selectedBits - 1;
        for (int bits = 2; bits < header.selectedBits - 1; ++bits) {
            if (!evaluated[static_cast<std::size_t>(bits)]) {
                unexpectedBits = bits;
                break;
            }
        }
    }
    const auto source = std::find_if(
        metrics.begin(),
        metrics.end(),
        [&](const auto& metric) {
            return metric.bits == sourceBits;
        });
    if (source == metrics.end() || unexpectedBits == 0) return false;

    StoredMetricRecord unexpected = *source;
    unexpected.bits = unexpectedBits;
    metrics.push_back(unexpected);
    std::sort(
        metrics.begin(),
        metrics.end(),
        [](const auto& left, const auto& right) {
            return left.bits < right.bits;
        });
    header.metricCount = static_cast<std::uint32_t>(metrics.size());
    header.fingerprint = ComputeStoredResultFingerprint(header, metrics);

    std::ofstream output(
        p_path, std::ios::binary | std::ios::trunc);
    return output.write(
               reinterpret_cast<const char*>(&header), sizeof(header)) &&
        output.write(
            reinterpret_cast<const char*>(metrics.data()),
            static_cast<std::streamsize>(
                metrics.size() * sizeof(StoredMetricRecord)));
}

class CapturingLogger final : public Helper::Logger
{
public:
    void Logging(const char*,
                 Helper::LogLevel,
                 const char*,
                 int,
                 const char*,
                 const char* p_format,
                 ...) override
    {
        char buffer[2048];
        va_list args;
        va_start(args, p_format);
        const int length =
            std::vsnprintf(buffer, sizeof(buffer), p_format, args);
        va_end(args);
        if (length > 0) {
            m_messages.append(
                buffer,
                static_cast<std::size_t>(
                    std::min<int>(length, sizeof(buffer) - 1)));
        }
    }

    const std::string& Messages() const
    {
        return m_messages;
    }

private:
    std::string m_messages;
};

class ScopedLogger
{
public:
    explicit ScopedLogger(const std::shared_ptr<Helper::Logger>& p_logger)
        : m_previous(GetLogger())
    {
        SetLogger(p_logger);
    }

    ~ScopedLogger()
    {
        SetLogger(m_previous);
    }

private:
    std::shared_ptr<Helper::Logger> m_previous;
};

class ScopedFiles
{
public:
    explicit ScopedFiles(std::vector<std::string> p_paths)
        : m_paths(std::move(p_paths))
    {
    }

    ~ScopedFiles()
    {
        for (const auto& path : m_paths) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
            std::filesystem::remove(path + ".writing", ec);
        }
    }

private:
    std::vector<std::string> m_paths;
};

std::shared_ptr<VectorSet> MakeCalibrationBase(bool identical = false)
{
    ByteArray bytes = ByteArray::Alloc(
        sizeof(float) * static_cast<std::size_t>(kBaseCount) * kDimension);
    auto* values = reinterpret_cast<float*>(bytes.Data());
    for (SizeType vector = 0; vector < kBaseCount; ++vector) {
        for (DimensionType dim = 0; dim < kDimension; ++dim) {
            values[static_cast<std::size_t>(vector) * kDimension + dim] =
                identical
                ? 0.25F + static_cast<float>(dim) * 0.002F
                : std::sin(
                      static_cast<float>((vector + 3) * (dim + 5)) * 0.017F) +
                      static_cast<float>(vector % 11) * 0.013F;
        }
    }
    return std::make_shared<BasicVectorSet>(
        bytes, VectorValueType::Float, kDimension, kBaseCount);
}

std::vector<float> MakeQueries(const std::shared_ptr<VectorSet>& p_base,
                               int p_queryCount)
{
    std::vector<float> queries(
        static_cast<std::size_t>(p_queryCount) * kDimension, 0.0F);
    for (int query = 0; query < p_queryCount; ++query) {
        const auto* source =
            static_cast<const float*>(p_base->GetVector((query * 13 + 7) % kBaseCount));
        for (DimensionType dim = 0; dim < kDimension; ++dim) {
            queries[static_cast<std::size_t>(query) * kDimension + dim] =
                source[dim] + static_cast<float>((query + dim) % 5) * 0.0003F;
        }
    }
    return queries;
}

std::vector<SizeType> ComputeTruth(const std::shared_ptr<VectorSet>& p_base,
                                   const std::vector<float>& p_queries,
                                   int p_queryCount,
                                   int p_depth,
                                   DistCalcMethod p_metric = DistCalcMethod::L2)
{
    std::vector<SizeType> truth(
        static_cast<std::size_t>(p_queryCount) * p_depth, -1);
    for (int query = 0; query < p_queryCount; ++query) {
        const float* queryVector =
            p_queries.data() + static_cast<std::size_t>(query) * kDimension;
        std::vector<std::pair<float, SizeType>> distances;
        distances.reserve(static_cast<std::size_t>(p_base->Count()));
        for (SizeType id = 0; id < p_base->Count(); ++id) {
            const float* candidate =
                static_cast<const float*>(p_base->GetVector(id));
            float distance = 0.0F;
            if (p_metric == DistCalcMethod::Cosine) {
                double dot = 0.0;
                double queryNorm = 0.0;
                double candidateNorm = 0.0;
                for (DimensionType dim = 0; dim < kDimension; ++dim) {
                    dot += static_cast<double>(queryVector[dim]) * candidate[dim];
                    queryNorm +=
                        static_cast<double>(queryVector[dim]) * queryVector[dim];
                    candidateNorm +=
                        static_cast<double>(candidate[dim]) * candidate[dim];
                }
                distance = static_cast<float>(
                    1.0 - dot / std::sqrt(queryNorm * candidateNorm));
            } else {
                for (DimensionType dim = 0; dim < kDimension; ++dim) {
                    const float difference = queryVector[dim] - candidate[dim];
                    distance += difference * difference;
                }
            }
            distances.emplace_back(distance, id);
        }
        std::sort(distances.begin(), distances.end());
        for (int rank = 0; rank < p_depth; ++rank) {
            truth[static_cast<std::size_t>(query) * p_depth + rank] =
                distances[static_cast<std::size_t>(rank)].second;
        }
    }
    return truth;
}

void WriteDefaultQueries(const std::string& p_path,
                         const std::vector<float>& p_queries,
                         int p_queryCount)
{
    std::ofstream output(p_path, std::ios::binary | std::ios::trunc);
    BOOST_REQUIRE(output.good());
    const SizeType rows = p_queryCount;
    output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char*>(&kDimension), sizeof(kDimension));
    output.write(
        reinterpret_cast<const char*>(p_queries.data()),
        static_cast<std::streamsize>(p_queries.size() * sizeof(float)));
    BOOST_REQUIRE(output.good());
}

void WriteDefaultTruth(const std::string& p_path,
                       const std::vector<SizeType>& p_truth,
                       int p_queryCount,
                       int p_depth)
{
    std::ofstream output(p_path, std::ios::binary | std::ios::trunc);
    BOOST_REQUIRE(output.good());
    const std::int32_t rows = p_queryCount;
    const std::int32_t depth = p_depth;
    output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char*>(&depth), sizeof(depth));
    for (SizeType id : p_truth) {
        const std::int32_t stored = static_cast<std::int32_t>(id);
        output.write(reinterpret_cast<const char*>(&stored), sizeof(stored));
    }
    std::vector<float> distances(p_truth.size(), 0.0F);
    output.write(
        reinterpret_cast<const char*>(distances.data()),
        static_cast<std::streamsize>(distances.size() * sizeof(float)));
    BOOST_REQUIRE(output.good());
}

void WriteDefaultBase(const std::string& p_path,
                      const std::shared_ptr<VectorSet>& p_base)
{
    std::ofstream output(p_path, std::ios::binary | std::ios::trunc);
    BOOST_REQUIRE(output.good());
    const SizeType rows = p_base->Count();
    output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char*>(&kDimension), sizeof(kDimension));
    output.write(
        reinterpret_cast<const char*>(p_base->GetData()),
        static_cast<std::streamsize>(
            static_cast<std::size_t>(rows) * kDimension * sizeof(float)));
    BOOST_REQUIRE(output.good());
}

void WriteXvecVectors(const std::string& p_path,
                      const float* p_vectors,
                      int p_count)
{
    std::ofstream output(p_path, std::ios::binary | std::ios::trunc);
    BOOST_REQUIRE(output.good());
    const std::int32_t dimension = kDimension;
    for (int vector = 0; vector < p_count; ++vector) {
        output.write(
            reinterpret_cast<const char*>(&dimension), sizeof(dimension));
        output.write(
            reinterpret_cast<const char*>(
                p_vectors + static_cast<std::size_t>(vector) * kDimension),
            static_cast<std::streamsize>(kDimension * sizeof(float)));
    }
    BOOST_REQUIRE(output.good());
}

void WriteXvecTruth(const std::string& p_path,
                    const std::vector<SizeType>& p_truth,
                    int p_queryCount,
                    int p_depth)
{
    std::ofstream output(p_path, std::ios::binary | std::ios::trunc);
    BOOST_REQUIRE(output.good());
    const std::int32_t depth = p_depth;
    for (int query = 0; query < p_queryCount; ++query) {
        output.write(reinterpret_cast<const char*>(&depth), sizeof(depth));
        for (int rank = 0; rank < p_depth; ++rank) {
            const std::int32_t id = static_cast<std::int32_t>(
                p_truth[static_cast<std::size_t>(query) * p_depth + rank]);
            output.write(reinterpret_cast<const char*>(&id), sizeof(id));
        }
    }
    BOOST_REQUIRE(output.good());
}

COMMON::RaBitQAdaptiveBitTrainer::Config MakeConfig(
    const std::string& p_prefix)
{
    COMMON::RaBitQAdaptiveBitTrainer::Config config;
    config.dimension = kDimension;
    config.metric = DistCalcMethod::L2;
    config.baseCount = kBaseCount;
    config.queryCount = kQueryCount;
    config.truthDepth = kTruthDepth;
    config.recallAt = kRecallAt;
    config.targetRecallError = 0.8F;
    config.queryPath = p_prefix + ".queries";
    config.queryType = VectorFileType::DEFAULT;
    config.truthPath = p_prefix + ".truth";
    config.truthType = TruthFileType::DEFAULT;
    config.baseType = VectorFileType::DEFAULT;
    config.trainingDataFile = p_prefix + ".training";
    config.trainingResultFile = p_prefix + ".result";
    config.modelFile = p_prefix + ".model";
    return config;
}

} // namespace

BOOST_AUTO_TEST_SUITE(RaBitQAdaptiveBitTrainerTest)

BOOST_AUTO_TEST_CASE(CreatesAndReusesFingerprintBoundCalibrationArtifacts)
{
    const std::string prefix = "rabitq_adaptive_trainer";
    auto config = MakeConfig(prefix);
    config.targetRecallError = 0.05F;
    ScopedFiles cleanup({
        config.queryPath,
        config.truthPath,
        config.trainingDataFile,
        config.trainingResultFile,
        config.modelFile,
    });
    const auto base = MakeCalibrationBase();
    const auto queries = MakeQueries(base, config.queryCount);
    const auto truth =
        ComputeTruth(base, queries, config.queryCount, config.truthDepth);
    WriteDefaultQueries(config.queryPath, queries, config.queryCount);
    WriteDefaultTruth(
        config.truthPath, truth, config.queryCount, config.truthDepth);

    COMMON::RaBitQAdaptiveBitTrainer::Result created;
    BOOST_REQUIRE(
        COMMON::RaBitQAdaptiveBitTrainer::Run(config, base, created) ==
        ErrorCode::Success);
    BOOST_CHECK(!created.reused);
    BOOST_CHECK(created.selectedBits >= 1 && created.selectedBits <= 8);
    BOOST_CHECK_GT(created.metrics.size(), 1U);
    BOOST_CHECK_LT(created.metrics.size(), 8U);
    BOOST_CHECK_LE(created.metrics.size(), 5U);
    BOOST_CHECK(std::is_sorted(
        created.metrics.begin(),
        created.metrics.end(),
        [](const auto& left, const auto& right) {
            return left.bits < right.bits;
        }));
    for (const auto& metric : created.metrics) {
        BOOST_CHECK_LE(
            metric.certifiedRecallLowerBound, metric.measuredRecall);
    }
    const auto selected = std::find_if(
        created.metrics.begin(),
        created.metrics.end(),
        [&](const auto& metric) {
            return metric.bits == created.selectedBits;
        });
    BOOST_REQUIRE(selected != created.metrics.end());
    if (created.selectedBits > 1) {
        BOOST_CHECK(std::any_of(
            created.metrics.begin(),
            created.metrics.end(),
            [&](const auto& metric) {
                return metric.bits == created.selectedBits - 1;
            }));
    }
    BOOST_CHECK(std::filesystem::exists(config.trainingDataFile));
    BOOST_CHECK(std::filesystem::exists(config.trainingResultFile));
    BOOST_CHECK(std::filesystem::exists(config.modelFile));

    std::filesystem::remove(config.queryPath);
    std::filesystem::remove(config.truthPath);
    std::filesystem::remove(config.trainingResultFile);
    std::filesystem::remove(config.modelFile);
    COMMON::RaBitQAdaptiveBitTrainer::Result replayed;
    BOOST_REQUIRE(
        COMMON::RaBitQAdaptiveBitTrainer::Run(config, nullptr, replayed) ==
        ErrorCode::Success);
    BOOST_CHECK(!replayed.reused);
    BOOST_CHECK_EQUAL(replayed.selectedBits, created.selectedBits);
    BOOST_CHECK_EQUAL(
        replayed.artifactFingerprint, created.artifactFingerprint);
    BOOST_CHECK_EQUAL(replayed.modelFingerprint, created.modelFingerprint);

    std::filesystem::remove(config.trainingDataFile);
    COMMON::RaBitQAdaptiveBitTrainer::Result reused;
    BOOST_REQUIRE(
        COMMON::RaBitQAdaptiveBitTrainer::Run(config, nullptr, reused) ==
        ErrorCode::Success);
    BOOST_CHECK(reused.reused);
    BOOST_CHECK_EQUAL(reused.selectedBits, replayed.selectedBits);
    BOOST_CHECK_EQUAL(reused.artifactFingerprint, replayed.artifactFingerprint);
    BOOST_CHECK_EQUAL(reused.modelFingerprint, replayed.modelFingerprint);

    auto mismatched = config;
    mismatched.recallAt = config.recallAt - 1;
    COMMON::RaBitQAdaptiveBitTrainer::Result stale;
    BOOST_CHECK(
        COMMON::RaBitQAdaptiveBitTrainer::Run(mismatched, nullptr, stale) ==
        ErrorCode::Fail);

    std::fstream corrupt(
        config.trainingResultFile,
        std::ios::binary | std::ios::in | std::ios::out);
    BOOST_REQUIRE(corrupt.good());
    char byte = 0;
    corrupt.read(&byte, 1);
    byte ^= 0x5a;
    corrupt.seekp(0);
    corrupt.write(&byte, 1);
    corrupt.close();
    COMMON::RaBitQAdaptiveBitTrainer::Result rejected;
    BOOST_CHECK(
        COMMON::RaBitQAdaptiveBitTrainer::Run(config, nullptr, rejected) ==
        ErrorCode::Fail);
}

BOOST_AUTO_TEST_CASE(FailsWithoutCertifyingLowBitRejection)
{
    const std::string prefix = "rabitq_adaptive_no_bit";
    auto config = MakeConfig(prefix);
    config.queryCount = 1;
    config.truthDepth = 16;
    config.recallAt = 1;
    config.targetRecallError = 0.01F;
    ScopedFiles cleanup({
        config.queryPath,
        config.truthPath,
        config.trainingDataFile,
        config.trainingResultFile,
        config.modelFile,
    });

    const auto base = MakeCalibrationBase(true);
    const auto queries = MakeQueries(base, config.queryCount);
    std::vector<SizeType> truth(static_cast<std::size_t>(config.truthDepth));
    std::iota(truth.rbegin(), truth.rend(), 0);
    WriteDefaultQueries(config.queryPath, queries, config.queryCount);
    WriteDefaultTruth(
        config.truthPath, truth, config.queryCount, config.truthDepth);

    auto logger = std::make_shared<CapturingLogger>();
    COMMON::RaBitQAdaptiveBitTrainer::Result result;
    {
        ScopedLogger scopedLogger(logger);
        BOOST_CHECK(
            COMMON::RaBitQAdaptiveBitTrainer::Run(config, base, result) ==
            ErrorCode::Fail);
    }
    BOOST_CHECK_EQUAL(result.selectedBits, 0);
    BOOST_CHECK_LT(result.metrics.size(), 8U);
    BOOST_CHECK(!std::filesystem::exists(config.trainingResultFile));
    BOOST_CHECK(!std::filesystem::exists(config.modelFile));
    const std::size_t lowBitLog =
        logger->Messages().find("Adaptive RaBitQ 1-bit ");
    BOOST_REQUIRE(lowBitLog != std::string::npos);
    const std::size_t lowBitLogEnd =
        logger->Messages().find('\n', lowBitLog);
    const std::string lowBitDecision = logger->Messages().substr(
        lowBitLog, lowBitLogEnd - lowBitLog);
    BOOST_CHECK(
        lowBitDecision.find("decision=below (measured recall)") !=
        std::string::npos);
    BOOST_CHECK(
        lowBitDecision.find("decision=below (certified") ==
        std::string::npos);
}

BOOST_AUTO_TEST_CASE(RejectsUnexpectedSparseResultMetric)
{
    const std::string prefix = "rabitq_adaptive_sparse_result";
    auto config = MakeConfig(prefix);
    config.targetRecallError = 0.05F;
    ScopedFiles cleanup({
        config.queryPath,
        config.truthPath,
        config.trainingDataFile,
        config.trainingResultFile,
        config.modelFile,
    });
    const auto base = MakeCalibrationBase();
    const auto queries = MakeQueries(base, config.queryCount);
    const auto truth =
        ComputeTruth(base, queries, config.queryCount, config.truthDepth);
    WriteDefaultQueries(config.queryPath, queries, config.queryCount);
    WriteDefaultTruth(
        config.truthPath, truth, config.queryCount, config.truthDepth);

    COMMON::RaBitQAdaptiveBitTrainer::Result created;
    BOOST_REQUIRE(
        COMMON::RaBitQAdaptiveBitTrainer::Run(config, base, created) ==
        ErrorCode::Success);
    BOOST_REQUIRE_LT(created.metrics.size(), 8U);
    BOOST_REQUIRE(AddUnexpectedSparseMetric(config.trainingResultFile));

    COMMON::RaBitQAdaptiveBitTrainer::Result rejected;
    BOOST_CHECK(
        COMMON::RaBitQAdaptiveBitTrainer::Run(config, nullptr, rejected) ==
        ErrorCode::Fail);
}

BOOST_AUTO_TEST_CASE(CosineModelEnablesInternalNormalization)
{
    const std::string prefix = "rabitq_adaptive_cosine";
    auto config = MakeConfig(prefix);
    config.metric = DistCalcMethod::Cosine;
    config.targetRecallError = 0.8F;
    ScopedFiles cleanup({
        config.queryPath,
        config.truthPath,
        config.trainingDataFile,
        config.trainingResultFile,
        config.modelFile,
    });
    const auto base = MakeCalibrationBase();
    const auto queries = MakeQueries(base, config.queryCount);
    const auto truth = ComputeTruth(
        base,
        queries,
        config.queryCount,
        config.truthDepth,
        DistCalcMethod::Cosine);
    WriteDefaultQueries(config.queryPath, queries, config.queryCount);
    WriteDefaultTruth(
        config.truthPath, truth, config.queryCount, config.truthDepth);

    COMMON::RaBitQAdaptiveBitTrainer::Result result;
    BOOST_REQUIRE(
        COMMON::RaBitQAdaptiveBitTrainer::Run(config, base, result) ==
        ErrorCode::Success);
    for (const auto& metric : result.metrics) {
        BOOST_CHECK_LE(
            metric.certifiedRecallLowerBound, metric.measuredRecall);
    }

    auto input = f_createIO();
    BOOST_REQUIRE(input != nullptr);
    BOOST_REQUIRE(input->Initialize(
        config.modelFile.c_str(), std::ios::binary | std::ios::in));
    auto quantizer = COMMON::IQuantizer::LoadIQuantizer(input);
    input->ShutDown();
    auto rabitq =
        std::dynamic_pointer_cast<COMMON::RaBitQQuantizer>(quantizer);
    BOOST_REQUIRE(rabitq != nullptr);
    BOOST_CHECK(rabitq->IsNormalizationEnabled());
}

BOOST_AUTO_TEST_CASE(LoadsDefaultAndXvecSourcesWithoutMemoryBase)
{
    const auto base = MakeCalibrationBase();
    const auto queries = MakeQueries(base, 2);
    const auto truth = ComputeTruth(base, queries, 2, 16);

    for (const bool useXvec : {false, true}) {
        const std::string prefix =
            useXvec ? "rabitq_adaptive_xvec" : "rabitq_adaptive_default";
        auto config = MakeConfig(prefix);
        config.basePath = prefix + ".base";
        config.queryCount = 2;
        config.truthDepth = 16;
        config.recallAt = 3;
        config.targetRecallError = 0.8F;
        std::vector<std::string> cleanupPaths = {
            config.basePath,
            config.queryPath,
            config.truthPath,
            config.trainingDataFile,
            config.trainingResultFile,
            config.modelFile,
        };
        if (useXvec) {
            const std::string baseSecond = prefix + ".base.second";
            const std::string querySecond = prefix + ".queries.second";
            cleanupPaths.push_back(baseSecond);
            cleanupPaths.push_back(querySecond);
            config.basePath += "," + baseSecond;
            config.queryPath += "," + querySecond;
        }
        ScopedFiles cleanup(cleanupPaths);

        if (useXvec) {
            config.baseType = VectorFileType::XVEC;
            config.queryType = VectorFileType::XVEC;
            config.truthType = TruthFileType::XVEC;
            const auto basePaths = std::vector<std::string>{
                prefix + ".base", prefix + ".base.second"};
            const auto queryPaths = std::vector<std::string>{
                prefix + ".queries", prefix + ".queries.second"};
            WriteXvecVectors(
                basePaths[0],
                static_cast<const float*>(base->GetData()),
                40);
            WriteXvecVectors(
                basePaths[1],
                static_cast<const float*>(base->GetData()) +
                    static_cast<std::size_t>(40) * kDimension,
                base->Count() - 40);
            WriteXvecVectors(queryPaths[0], queries.data(), 1);
            WriteXvecVectors(
                queryPaths[1], queries.data() + kDimension, 1);
            WriteXvecTruth(
                config.truthPath,
                truth,
                config.queryCount,
                config.truthDepth);
        } else {
            config.baseType = VectorFileType::DEFAULT;
            WriteDefaultBase(config.basePath, base);
            WriteDefaultQueries(
                config.queryPath, queries, config.queryCount);
            WriteDefaultTruth(
                config.truthPath,
                truth,
                config.queryCount,
                config.truthDepth);
        }

        COMMON::RaBitQAdaptiveBitTrainer::Result result;
        BOOST_REQUIRE(
            COMMON::RaBitQAdaptiveBitTrainer::Run(config, nullptr, result) ==
            ErrorCode::Success);
        BOOST_CHECK(result.selectedBits >= 1 && result.selectedBits <= 8);
    }
}

BOOST_AUTO_TEST_SUITE_END()
