// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/RaBitQAutoTuner.h"

#include "inc/Core/VectorIndex.h"
#include "inc/Helper/StringConvert.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace SPTAG
{
namespace COMMON
{
namespace
{

constexpr SizeType kBatchSize = 64 * 1024;
constexpr const char* kSection = "RaBitQAutoTune";

template <typename T>
bool ReadRequired(const Helper::IniReader& p_config,
                  const char* p_section,
                  const char* p_name,
                  T& p_value,
                  std::string& p_error)
{
    if (!p_config.DoesParameterExist(p_section, p_name)) {
        p_error = std::string("[") + p_section + "] " + p_name + " is required";
        return false;
    }
    const std::string raw = p_config.GetParameter(
        p_section, p_name, std::string());
    if (!Helper::Convert::ConvertStringTo<T>(raw.c_str(), p_value)) {
        p_error = std::string("invalid [") + p_section + "] " + p_name + ": " + raw;
        return false;
    }
    return true;
}

bool ReadRequiredString(const Helper::IniReader& p_config,
                        const char* p_section,
                        const char* p_name,
                        std::string& p_value,
                        std::string& p_error)
{
    if (!p_config.DoesParameterExist(p_section, p_name) ||
        (p_value = p_config.GetParameter(p_section, p_name, std::string())).empty()) {
        p_error = std::string("[") + p_section + "] " + p_name + " is required";
        return false;
    }
    return true;
}

ErrorCode LoadTruth(const std::string& p_path,
                    TruthFileType p_type,
                    SizeType p_queryCount,
                    std::vector<std::vector<SizeType>>& p_truth,
                    std::string& p_error)
{
    auto input = f_createIO();
    if (!input || !input->Initialize(p_path.c_str(), std::ios::binary | std::ios::in)) {
        p_error = "cannot open [Base] TruthPath: " + p_path;
        return ErrorCode::FailedOpenFile;
    }

    p_truth.clear();
    p_truth.reserve(static_cast<std::size_t>(p_queryCount));
    if (p_type == TruthFileType::DEFAULT) {
        std::int32_t rows = 0;
        std::int32_t depth = 0;
        if (input->ReadBinary(sizeof(rows), reinterpret_cast<char*>(&rows)) != sizeof(rows) ||
            input->ReadBinary(sizeof(depth), reinterpret_cast<char*>(&depth)) != sizeof(depth) ||
            rows < p_queryCount || depth <= 0) {
            p_error = "invalid or insufficient DEFAULT truth header";
            return ErrorCode::FailedParseValue;
        }
        for (SizeType query = 0; query < p_queryCount; ++query) {
            std::vector<std::int32_t> ids(static_cast<std::size_t>(depth));
            const std::uint64_t bytes =
                sizeof(std::int32_t) * static_cast<std::uint64_t>(depth);
            if (input->ReadBinary(bytes, reinterpret_cast<char*>(ids.data())) != bytes) {
                p_error = "DEFAULT truth ended before QueryCountLimit rows";
                return ErrorCode::FailedParseValue;
            }
            p_truth.emplace_back(ids.begin(), ids.end());
        }
    } else if (p_type == TruthFileType::XVEC) {
        std::int32_t expectedDepth = -1;
        for (SizeType query = 0; query < p_queryCount; ++query) {
            std::int32_t depth = 0;
            if (input->ReadBinary(sizeof(depth), reinterpret_cast<char*>(&depth)) != sizeof(depth) ||
                depth <= 0 || (expectedDepth >= 0 && depth != expectedDepth)) {
                p_error = "XVEC truth has missing or inconsistent candidate depth";
                return ErrorCode::FailedParseValue;
            }
            expectedDepth = depth;
            std::vector<std::int32_t> ids(static_cast<std::size_t>(depth));
            const std::uint64_t bytes =
                sizeof(std::int32_t) * static_cast<std::uint64_t>(depth);
            if (input->ReadBinary(bytes, reinterpret_cast<char*>(ids.data())) != bytes) {
                p_error = "XVEC truth ended before QueryCountLimit rows";
                return ErrorCode::FailedParseValue;
            }
            p_truth.emplace_back(ids.begin(), ids.end());
        }
    } else if (p_type == TruthFileType::TXT) {
        std::size_t expectedDepth = 0;
        std::uint64_t bufferSize = 64 * 1024;
        std::unique_ptr<char[]> buffer(new char[bufferSize]);
        for (SizeType query = 0; query < p_queryCount; ++query) {
            if (input->ReadString(bufferSize, buffer, '\n') == 0) {
                p_error = "TXT truth ended before QueryCountLimit rows";
                return ErrorCode::FailedParseValue;
            }
            std::vector<SizeType> row;
            char* context = nullptr;
#ifdef _MSC_VER
            char* token = strtok_s(buffer.get(), " \t", &context);
#else
            char* token = strtok_r(buffer.get(), " \t", &context);
#endif
            while (token != nullptr) {
                SizeType id = -1;
                if (!Helper::Convert::ConvertStringTo<SizeType>(token, id)) {
                    p_error = "TXT truth contains a non-integer candidate ID";
                    return ErrorCode::FailedParseValue;
                }
                row.push_back(id);
#ifdef _MSC_VER
                token = strtok_s(nullptr, " \t", &context);
#else
                token = strtok_r(nullptr, " \t", &context);
#endif
            }
            if (row.empty() || (!p_truth.empty() && row.size() != expectedDepth)) {
                p_error = "TXT truth has empty or inconsistent candidate depth";
                return ErrorCode::FailedParseValue;
            }
            expectedDepth = row.size();
            p_truth.emplace_back(std::move(row));
        }
    } else {
        p_error = "[Base] TruthType must be DEFAULT, XVEC, or TXT";
        return ErrorCode::FailedParseValue;
    }
    return ErrorCode::Success;
}

ErrorCode TrainCentroid(const std::shared_ptr<Helper::VectorSetReader>& p_reader,
                        DimensionType p_dimension,
                        std::shared_ptr<RaBitQQuantizer>& p_model,
                        SizeType& p_count,
                        std::string& p_error)
{
    p_model = std::make_shared<RaBitQQuantizer>(p_dimension, 1, false);
    if (p_model->BeginTraining() != ErrorCode::Success) {
        p_error = "failed to initialize streaming RaBitQ centroid training";
        return ErrorCode::Fail;
    }

    p_count = 0;
    for (SizeType start = 0;; start += kBatchSize) {
        const auto batch = p_reader->GetVectorSet(start, start + kBatchSize);
        if (!batch || batch->Count() == 0) {
            break;
        }
        if (batch->GetValueType() != VectorValueType::Float ||
            batch->Dimension() != p_dimension ||
            p_model->AddTrainingBatch(batch) != ErrorCode::Success) {
            p_error = "base vector batch is incompatible with Float RaBitQ training";
            return ErrorCode::FailedParseValue;
        }
        p_count += batch->Count();
        if (batch->Count() < kBatchSize) {
            break;
        }
        if (start > (std::numeric_limits<SizeType>::max)() - 2 * kBatchSize) {
            p_error = "base vector count exceeds native SizeType";
            return ErrorCode::Fail;
        }
    }
    if (p_count <= 0 || p_model->FinishTraining() != ErrorCode::Success) {
        p_error = "base vector source is empty";
        return ErrorCode::EmptyData;
    }
    return ErrorCode::Success;
}

ErrorCode EvaluateBits(const std::shared_ptr<RaBitQQuantizer>& p_quantizer,
                       const std::shared_ptr<Helper::VectorSetReader>& p_baseReader,
                       const std::shared_ptr<VectorSet>& p_queries,
                       const std::vector<std::vector<SizeType>>& p_truth,
                       SizeType p_baseCount,
                       int p_resultCount,
                       int p_threads,
                       float& p_recall,
                       std::string& p_error)
{
    std::vector<SizeType> candidateIds;
    for (const auto& row : p_truth) {
        candidateIds.insert(candidateIds.end(), row.begin(), row.end());
    }
    std::sort(candidateIds.begin(), candidateIds.end());
    candidateIds.erase(std::unique(candidateIds.begin(), candidateIds.end()), candidateIds.end());

    const DimensionType codeDimension = p_quantizer->GetNumSubvectors();
    std::vector<std::uint8_t> codes(
        candidateIds.size() * static_cast<std::size_t>(codeDimension));
    p_quantizer->SetEnableADC(true);

    std::size_t candidate = 0;
    for (SizeType start = 0; start < p_baseCount; start += kBatchSize) {
        const SizeType end = std::min<SizeType>(p_baseCount, start + kBatchSize);
        const auto batch = p_baseReader->GetVectorSet(start, end);
        if (!batch || batch->Count() != end - start) {
            p_error = "base vector source changed while evaluating RaBitQ bits";
            return ErrorCode::Fail;
        }
        while (candidate < candidateIds.size() && candidateIds[candidate] < end) {
            const SizeType id = candidateIds[candidate];
            p_quantizer->QuantizeVector(
                batch->GetVector(id - start),
                codes.data() + candidate * static_cast<std::size_t>(codeDimension),
                false);
            ++candidate;
        }
    }
    if (candidate != candidateIds.size()) {
        p_error = "not all truth candidates could be encoded";
        return ErrorCode::Fail;
    }

    std::unordered_map<SizeType, std::size_t> codeOffsets;
    codeOffsets.reserve(candidateIds.size());
    for (std::size_t i = 0; i < candidateIds.size(); ++i) {
        codeOffsets.emplace(candidateIds[i], i);
    }

    const int queryBytes = p_quantizer->QuantizeSize();
    std::vector<std::uint8_t> queryCodes(
        static_cast<std::size_t>(p_queries->Count()) * static_cast<std::size_t>(queryBytes));
    for (SizeType query = 0; query < p_queries->Count(); ++query) {
        p_quantizer->QuantizeVector(
            p_queries->GetVector(query),
            queryCodes.data() + static_cast<std::size_t>(query) * queryBytes,
            true);
    }

    std::vector<float> queryRecalls(static_cast<std::size_t>(p_queries->Count()), 0.0F);
    std::atomic<SizeType> nextQuery(0);
    std::atomic<bool> evaluationFailed(false);
    const int workerCount = std::max(1, std::min<int>(p_threads, p_queries->Count()));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(workerCount));
    for (int worker = 0; worker < workerCount; ++worker) {
        workers.emplace_back([&]() {
            for (;;) {
                const SizeType query = nextQuery.fetch_add(1);
                if (query >= p_queries->Count()) {
                    return;
                }
                std::vector<std::pair<float, SizeType>> ranked;
                ranked.reserve(p_truth[static_cast<std::size_t>(query)].size());
                const std::uint8_t* queryCode =
                    queryCodes.data() + static_cast<std::size_t>(query) * queryBytes;
                for (SizeType id : p_truth[static_cast<std::size_t>(query)]) {
                    const auto offset = codeOffsets.find(id);
                    if (offset == codeOffsets.end()) {
                        evaluationFailed.store(true);
                        return;
                    }
                    const std::uint8_t* code =
                        codes.data() + offset->second * static_cast<std::size_t>(codeDimension);
                    ranked.emplace_back(p_quantizer->L2Distance(queryCode, code), id);
                }
                std::partial_sort(
                    ranked.begin(), ranked.begin() + p_resultCount, ranked.end(),
                    [](const auto& p_left, const auto& p_right) {
                        return p_left.first < p_right.first ||
                            (p_left.first == p_right.first && p_left.second < p_right.second);
                    });
                std::vector<SizeType> ids;
                ids.reserve(static_cast<std::size_t>(p_resultCount));
                for (int rank = 0; rank < p_resultCount; ++rank) {
                    ids.push_back(ranked[static_cast<std::size_t>(rank)].second);
                }
                queryRecalls[static_cast<std::size_t>(query)] =
                    RaBitQAutoTuner::RecallAtK(
                        p_truth[static_cast<std::size_t>(query)], ids, p_resultCount);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    if (evaluationFailed.load()) {
        p_error = "truth candidate code lookup failed during parallel evaluation";
        return ErrorCode::Fail;
    }
    double recallSum = 0.0;
    for (float recall : queryRecalls) {
        recallSum += recall;
    }
    p_recall = static_cast<float>(recallSum / p_queries->Count());
    return ErrorCode::Success;
}

ErrorCode SaveArtifacts(const std::shared_ptr<RaBitQQuantizer>& p_quantizer,
                        const std::shared_ptr<Helper::VectorSetReader>& p_reader,
                        SizeType p_count,
                        const std::string& p_outputFolder,
                        RaBitQAutoTuneResult& p_result,
                        std::string& p_error)
{
    namespace fs = std::filesystem;
    std::error_code filesystemError;
    fs::create_directories(p_outputFolder, filesystemError);
    if (filesystemError) {
        p_error = "cannot create RaBitQ artifact directory: " + filesystemError.message();
        return ErrorCode::FailedCreateFile;
    }

    const fs::path folder(p_outputFolder);
    const fs::path marker = folder / "rabitq_auto.incomplete";
    const fs::path modelTemporary = folder / "rabitq_auto_quantizer.bin.incomplete";
    const fs::path vectorTemporary = folder / "rabitq_auto_vectors.bin.incomplete";
    const fs::path modelFinal = folder / "rabitq_auto_quantizer.bin";
    const fs::path vectorFinal = folder / "rabitq_auto_vectors.bin";
    fs::remove(modelTemporary, filesystemError);
    fs::remove(vectorTemporary, filesystemError);
    {
        auto markerOutput = f_createIO();
        if (!markerOutput ||
            !markerOutput->Initialize(marker.string().c_str(), std::ios::out | std::ios::binary) ||
            markerOutput->WriteString("RaBitQ auto-tuning artifacts are incomplete\n") == 0) {
            p_error = "cannot create RaBitQ incomplete marker";
            return ErrorCode::FailedCreateFile;
        }
    }

    auto fail = [&](ErrorCode p_status, const std::string& p_message) {
        p_error = p_message;
        fs::remove(modelTemporary, filesystemError);
        fs::remove(vectorTemporary, filesystemError);
        return p_status;
    };

    {
        auto modelOutput = f_createIO();
        if (!modelOutput ||
            !modelOutput->Initialize(
                modelTemporary.string().c_str(), std::ios::out | std::ios::binary) ||
            p_quantizer->SaveQuantizer(modelOutput) != ErrorCode::Success) {
            return fail(ErrorCode::DiskIOFail, "failed to write RaBitQ quantizer");
        }
    }

    const DimensionType codeDimension = p_quantizer->GetNumSubvectors();
    {
        auto vectorOutput = f_createIO();
        if (!vectorOutput ||
            !vectorOutput->Initialize(
                vectorTemporary.string().c_str(), std::ios::out | std::ios::binary) ||
            vectorOutput->WriteBinary(sizeof(p_count), reinterpret_cast<const char*>(&p_count)) !=
                sizeof(p_count) ||
            vectorOutput->WriteBinary(
                sizeof(codeDimension), reinterpret_cast<const char*>(&codeDimension)) !=
                sizeof(codeDimension)) {
            return fail(ErrorCode::DiskIOFail, "failed to initialize encoded vector artifact");
        }
        p_quantizer->SetEnableADC(true);
        std::vector<std::uint8_t> code(static_cast<std::size_t>(codeDimension));
        SizeType written = 0;
        for (SizeType start = 0; start < p_count; start += kBatchSize) {
            const SizeType end = std::min<SizeType>(p_count, start + kBatchSize);
            const auto batch = p_reader->GetVectorSet(start, end);
            if (!batch || batch->Count() != end - start) {
                return fail(ErrorCode::Fail, "base vector source changed during final encoding");
            }
            for (SizeType i = 0; i < batch->Count(); ++i) {
                p_quantizer->QuantizeVector(batch->GetVector(i), code.data(), false);
                if (vectorOutput->WriteBinary(
                        code.size(), reinterpret_cast<const char*>(code.data())) != code.size()) {
                    return fail(ErrorCode::DiskIOFail, "failed to stream encoded base vectors");
                }
                ++written;
            }
        }
        if (written != p_count) {
            return fail(ErrorCode::Fail, "encoded vector count mismatch");
        }
    }

    const std::uintmax_t expectedVectorSize =
        sizeof(SizeType) + sizeof(DimensionType) +
        static_cast<std::uintmax_t>(p_count) * static_cast<std::uintmax_t>(codeDimension);
    if (fs::file_size(vectorTemporary, filesystemError) != expectedVectorSize ||
        filesystemError) {
        return fail(ErrorCode::Fail, "encoded vector artifact size mismatch");
    }

    auto modelInput = f_createIO();
    if (!modelInput ||
        !modelInput->Initialize(
            modelTemporary.string().c_str(), std::ios::in | std::ios::binary)) {
        return fail(ErrorCode::FailedOpenFile, "cannot reopen generated RaBitQ quantizer");
    }
    const auto loaded = IQuantizer::LoadIQuantizer(modelInput);
    const auto loadedRaBitQ = std::dynamic_pointer_cast<RaBitQQuantizer>(loaded);
    if (!loadedRaBitQ || loadedRaBitQ->Bits() != p_quantizer->Bits() ||
        loadedRaBitQ->Dimension() != p_quantizer->Dimension() ||
        loadedRaBitQ->GetNumSubvectors() != codeDimension) {
        return fail(ErrorCode::FailedParseValue, "generated model is incompatible with encoded vectors");
    }

    fs::remove(modelFinal, filesystemError);
    fs::rename(modelTemporary, modelFinal, filesystemError);
    if (filesystemError) {
        return fail(ErrorCode::DiskIOFail, "cannot publish RaBitQ quantizer: " + filesystemError.message());
    }
    fs::remove(vectorFinal, filesystemError);
    fs::rename(vectorTemporary, vectorFinal, filesystemError);
    if (filesystemError) {
        return fail(ErrorCode::DiskIOFail, "cannot publish encoded vectors: " + filesystemError.message());
    }
    fs::remove(marker, filesystemError);

    p_result.quantizerPath = modelFinal.string();
    p_result.vectorPath = vectorFinal.string();
    p_result.codeDimension = codeDimension;
    p_result.vectorCount = p_count;
    p_result.quantizer = p_quantizer;
    return ErrorCode::Success;
}

} // namespace

bool RaBitQAutoTuner::IsEnabled(const Helper::IniReader& p_config)
{
    return p_config.DoesSectionExist(kSection) &&
        p_config.GetParameter(kSection, "isExecute", false);
}

ErrorCode RaBitQAutoTuner::Run(Helper::IniReader& p_config,
                               const std::string& p_outputFolder,
                               RaBitQAutoTuneResult& p_result,
                               std::string& p_error)
{
    p_result = RaBitQAutoTuneResult();
    p_error.clear();
    for (const auto& parameter : p_config.GetParameters(kSection)) {
        if (!Helper::StrUtils::StrEqualIgnoreCase(parameter.first.c_str(), "isExecute") &&
            !Helper::StrUtils::StrEqualIgnoreCase(parameter.first.c_str(), "TargetRecall")) {
            p_error = "unsupported [RaBitQAutoTune] parameter: " + parameter.first;
            return ErrorCode::FailedParseValue;
        }
    }

    VectorValueType valueType = VectorValueType::Undefined;
    DistCalcMethod distance = DistCalcMethod::Undefined;
    DimensionType dimension = 0;
    VectorFileType vectorType = VectorFileType::Undefined;
    VectorFileType queryType = VectorFileType::Undefined;
    TruthFileType truthType = TruthFileType::Undefined;
    SizeType queryCount = 0;
    int resultCount = 0;
    int threads = 0;
    float targetRecall = 0.0F;
    std::string vectorPath;
    std::string queryPath;
    std::string truthPath;
    if (!ReadRequired(p_config, "Base", "ValueType", valueType, p_error) ||
        !ReadRequired(p_config, "Base", "DistCalcMethod", distance, p_error) ||
        !ReadRequired(p_config, "Base", "Dim", dimension, p_error) ||
        !ReadRequired(p_config, "Base", "VectorType", vectorType, p_error) ||
        !ReadRequired(p_config, "Base", "QueryType", queryType, p_error) ||
        !ReadRequired(p_config, "Base", "TruthType", truthType, p_error) ||
        !ReadRequiredString(p_config, "Base", "VectorPath", vectorPath, p_error) ||
        !ReadRequiredString(p_config, "Base", "QueryPath", queryPath, p_error) ||
        !ReadRequiredString(p_config, "Base", "TruthPath", truthPath, p_error) ||
        !ReadRequired(p_config, "SearchSSDIndex", "QueryCountLimit", queryCount, p_error) ||
        !ReadRequired(p_config, "SearchSSDIndex", "ResultNum", resultCount, p_error) ||
        !ReadRequired(p_config, "BuildSSDIndex", "NumberOfThreads", threads, p_error) ||
        !ReadRequired(p_config, kSection, "TargetRecall", targetRecall, p_error)) {
        return ErrorCode::FailedParseValue;
    }
    if (valueType != VectorValueType::Float || distance != DistCalcMethod::L2) {
        p_error = "global RaBitQ auto-tuning requires [Base] ValueType=Float and DistCalcMethod=L2";
        return ErrorCode::FailedParseValue;
    }
    if (dimension <= 0 || queryCount <= 0 || resultCount <= 0 || threads <= 0 ||
        !std::isfinite(targetRecall) || targetRecall < 0.0F || targetRecall > 1.0F ||
        vectorType == VectorFileType::Undefined || queryType == VectorFileType::Undefined ||
        truthType == TruthFileType::Undefined || p_outputFolder.empty()) {
        p_error = "invalid RaBitQ input dimension, counts, types, threads, target recall, or output folder";
        return ErrorCode::FailedParseValue;
    }

    const std::string vectorDelimiter =
        p_config.GetParameter("Base", "VectorDelimiter", std::string("|"));
    const std::string queryDelimiter =
        p_config.GetParameter("Base", "QueryDelimiter", std::string("|"));
    auto baseOptions = std::make_shared<Helper::ReaderOptions>(
        VectorValueType::Float, dimension, vectorType, vectorDelimiter, threads, false);
    auto baseReader = Helper::VectorSetReader::CreateInstance(baseOptions);
    if (!baseReader || baseReader->LoadFile(vectorPath) != ErrorCode::Success) {
        p_error = "failed to load [Base] VectorPath with its declared VectorType";
        return ErrorCode::FailedOpenFile;
    }

    std::shared_ptr<RaBitQQuantizer> centroidModel;
    SizeType baseCount = 0;
    ErrorCode status =
        TrainCentroid(baseReader, dimension, centroidModel, baseCount, p_error);
    if (status != ErrorCode::Success) {
        return status;
    }

    auto queryOptions = std::make_shared<Helper::ReaderOptions>(
        VectorValueType::Float, dimension, queryType, queryDelimiter, threads, false);
    auto queryReader = Helper::VectorSetReader::CreateInstance(queryOptions);
    if (!queryReader || queryReader->LoadFile(queryPath) != ErrorCode::Success) {
        p_error = "failed to load [Base] QueryPath with its declared QueryType";
        return ErrorCode::FailedOpenFile;
    }
    const auto queries = queryReader->GetVectorSet(0, queryCount);
    if (!queries || queries->Count() != queryCount ||
        queries->GetValueType() != VectorValueType::Float ||
        queries->Dimension() != dimension) {
        p_error = "QueryPath does not contain exactly QueryCountLimit usable Float queries";
        return ErrorCode::FailedParseValue;
    }

    std::vector<std::vector<SizeType>> truth;
    status = LoadTruth(truthPath, truthType, queryCount, truth, p_error);
    if (status != ErrorCode::Success ||
        (status = ValidateTruth(
             truth, baseCount, queryCount, resultCount, p_error)) != ErrorCode::Success) {
        return status;
    }

    std::shared_ptr<RaBitQQuantizer> selected;
    status = SelectMinimumBits(
        targetRecall,
        [&](int p_bits, float& p_recall) {
            auto candidate = centroidModel->CreateWithBits(p_bits);
            if (!candidate) {
                p_error = "failed to create RaBitQ candidate from shared centroid";
                return ErrorCode::Fail;
            }
            const ErrorCode evaluation = EvaluateBits(
                candidate, baseReader, queries, truth, baseCount,
                resultCount, threads, p_recall, p_error);
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Info,
                "RaBitQ auto-tuning bits=%d Recall@%d=%.6f target=%.6f\n",
                p_bits, resultCount, p_recall, targetRecall);
            if (evaluation == ErrorCode::Success && p_recall >= targetRecall) {
                selected = std::move(candidate);
            }
            return evaluation;
        },
        p_result.selectedBits, p_result.recall);
    if (status != ErrorCode::Success) {
        if (p_error.empty()) {
            p_error = "no RaBitQ bit width in the fixed range 1..8 meets TargetRecall";
        }
        return status;
    }
    if (!selected || selected->Bits() != p_result.selectedBits) {
        p_error = "RaBitQ selected model does not match selected bit width";
        return ErrorCode::Fail;
    }

    status = SaveArtifacts(
        selected, baseReader, baseCount, p_outputFolder, p_result, p_error);
    if (status == ErrorCode::Success) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Info,
            "RaBitQ auto-tuning selected %d bits (Recall@%d=%.6f); encoded %d vectors at width %d\n",
            p_result.selectedBits, resultCount, p_result.recall,
            p_result.vectorCount, p_result.codeDimension);
    }
    return status;
}

ErrorCode RaBitQAutoTuner::SelectMinimumBits(float p_targetRecall,
                                             const BitEvaluator& p_evaluator,
                                             int& p_selectedBits,
                                             float& p_selectedRecall)
{
    p_selectedBits = 0;
    p_selectedRecall = 0.0F;
    if (!p_evaluator || !std::isfinite(p_targetRecall) ||
        p_targetRecall < 0.0F || p_targetRecall > 1.0F) {
        return ErrorCode::FailedParseValue;
    }
    for (int bits = 1; bits <= 8; ++bits) {
        float recall = 0.0F;
        const ErrorCode status = p_evaluator(bits, recall);
        if (status != ErrorCode::Success || !std::isfinite(recall)) {
            return status == ErrorCode::Success ? ErrorCode::Fail : status;
        }
        if (recall >= p_targetRecall) {
            p_selectedBits = bits;
            p_selectedRecall = recall;
            return ErrorCode::Success;
        }
    }
    return ErrorCode::Fail;
}

ErrorCode RaBitQAutoTuner::ValidateTruth(
    const std::vector<std::vector<SizeType>>& p_truth,
    SizeType p_baseCount,
    SizeType p_queryCount,
    int p_resultCount,
    std::string& p_error)
{
    if (p_baseCount <= 0 || p_queryCount <= 0 || p_resultCount <= 0 ||
        p_truth.size() != static_cast<std::size_t>(p_queryCount)) {
        p_error = "truth query count does not match QueryCountLimit";
        return ErrorCode::FailedParseValue;
    }
    std::size_t depth = 0;
    for (std::size_t query = 0; query < p_truth.size(); ++query) {
        const auto& row = p_truth[query];
        if (query == 0) {
            depth = row.size();
            if (depth <= static_cast<std::size_t>(p_resultCount)) {
                p_error = "truth candidate depth must be greater than ResultNum";
                return ErrorCode::FailedParseValue;
            }
        } else if (row.size() != depth) {
            p_error = "truth candidate depth is inconsistent across queries";
            return ErrorCode::FailedParseValue;
        }
        std::unordered_set<SizeType> seen;
        seen.reserve(row.size());
        for (SizeType id : row) {
            if (id < 0 || id >= p_baseCount) {
                p_error = "truth contains a candidate ID outside the base vector source";
                return ErrorCode::FailedParseValue;
            }
            if (!seen.insert(id).second) {
                p_error = "truth contains a duplicate candidate ID";
                return ErrorCode::FailedParseValue;
            }
        }
    }
    return ErrorCode::Success;
}

float RaBitQAutoTuner::RecallAtK(const std::vector<SizeType>& p_exact,
                                 const std::vector<SizeType>& p_reranked,
                                 int p_resultCount)
{
    if (p_resultCount <= 0 ||
        p_exact.size() < static_cast<std::size_t>(p_resultCount) ||
        p_reranked.size() < static_cast<std::size_t>(p_resultCount)) {
        return 0.0F;
    }
    std::unordered_set<SizeType> exact(
        p_exact.begin(), p_exact.begin() + p_resultCount);
    int matches = 0;
    for (int i = 0; i < p_resultCount; ++i) {
        matches += exact.find(p_reranked[static_cast<std::size_t>(i)]) != exact.end();
    }
    return static_cast<float>(matches) / p_resultCount;
}

} // namespace COMMON
} // namespace SPTAG
