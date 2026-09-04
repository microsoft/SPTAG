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

ErrorCode EvaluateBits(const std::shared_ptr<RaBitQQuantizer>& p_quantizer,
                       const std::shared_ptr<VectorSet>& p_base,
                       const std::shared_ptr<VectorSet>& p_queries,
                       const std::vector<std::vector<SizeType>>& p_truth,
                       int p_resultCount,
                       int p_threads,
                       float& p_recall)
{
    const DimensionType codeDimension = p_quantizer->GetNumSubvectors();
    std::vector<std::uint8_t> codes(
        p_base->Count() * static_cast<std::size_t>(codeDimension));
    for (SizeType i = 0; i < p_base->Count(); ++i) {
        p_quantizer->QuantizeVector(
            p_base->GetVector(i),
            codes.data() + i * static_cast<std::size_t>(codeDimension),
            false);
    }

    p_quantizer->SetEnableADC(true);
    std::vector<float> queryRecalls(p_queries->Count(), 0.0F);
    std::atomic<SizeType> nextQuery(0);
    const int workerCount = std::max(1, std::min<int>(p_threads, (int)(p_queries->Count())));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(workerCount));
    for (int worker = 0; worker < workerCount; ++worker) {
        workers.emplace_back([&]() {
            std::vector<std::uint8_t> queryCode(static_cast<std::size_t>(p_quantizer->QuantizeSize()));
            for (;;) {
                const SizeType query = nextQuery.fetch_add(1);
                if (query >= p_queries->Count()) {
                    return;
                }
                p_quantizer->QuantizeVector(p_queries->GetVector(query), queryCode.data(), true);

                SPTAG::COMMON::QueryResultSet<void> queryResult((void *)queryCode.data(), p_resultCount);
                for (SPTAG::SizeType j = 0; j < p_base->Count(); j++)
                {
                    const std::uint8_t* code = codes.data() + j * static_cast<std::size_t>(codeDimension);
                    queryResult.AddPoint(j, p_quantizer->L2Distance(queryCode, code));
                }
                queryResult.SortResult();

                int matches = 0;
                for (int i = 0; i < p_resultCount; ++i) {
                    matches += p_truth[query].find(queryResult.GetResult(i).VID) != p_truth[query].end();
                }
                queryRecalls[query] = static_cast<float>(matches) / p_resultCount;
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    double recallSum = 0.0;
    for (float recall : queryRecalls) {
        recallSum += recall;
    }
    p_recall = static_cast<float>(recallSum / p_queries->Count());
    return ErrorCode::Success;
}

ErrorCode SelectMinimumBits(float p_targetRecall,
                            std::function<ErrorCode(int, float&)>& p_evaluator,
                            int& p_selectedBits,
                            float& p_selectedRecall)
{
    p_selectedBits = 0;
    p_selectedRecall = 0.0F;
    if (!p_evaluator || !std::isfinite(p_targetRecall) ||
        p_targetRecall < 0.0F || p_targetRecall > 1.0F) {
        return ErrorCode::FailedParseValue;
    }
    int low = 1;
    int high = 8;
    while (low <= high) {
        const int bits = low + (high - low) / 2;
        float recall = 0.0F;
        const ErrorCode status = p_evaluator(bits, recall);
        if (status != ErrorCode::Success || !std::isfinite(recall)) {
            return status == ErrorCode::Success ? ErrorCode::Fail : status;
        }
        if (recall >= p_targetRecall) {
            p_selectedBits = bits;
            p_selectedRecall = recall;
            high = bits - 1;
        }
        else {
            low = bits + 1;
        }
    }
    return p_selectedBits == 0 ? ErrorCode::Fail : ErrorCode::Success;
}

} // namespace

std::shared_ptr<IQuantizer> RaBitQAutoTuner::Run(std::shared_ptr<VectorSet>& p_base,
                               int p_queryNum, int p_resultCount, int p_threads, float p_targetRecall,
                               DistCalcMethod p_distance,
                               const std::string& p_outputFolder,
                               RaBitQAutoTuneResult& p_result)
{
    p_result = RaBitQAutoTuneResult();

    if (p_queryNum <= 0 || p_resultCount <= 0 || p_threads <= 0 || p_targetRecall < 0.0F || p_targetRecall > 1.0F || p_distance == DistCalcMethod::Undefined || p_outputFolder.empty()) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Invalid RaBitQ query number, result counts, threads, target recall, distance type or output folder!\n");
        return ErrorCode::FailedParseValue;
    }

    std::shared_ptr<RaBitQQuantizer> centroidModel = std::make_shared<RaBitQQuantizer>(p_base->Dimension(), 8, false);
    ErrorCode status = centroidModel->Train(p_base);
    if (status != ErrorCode::Success) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to train RaBitQ centroid model!\n");
        return ErrorCode::Fail;
    }
    
    if (p_queryNum > p_base->Count()) p_queryNum = p_base->Count();
    std::shared_ptr<VectorSet> queries = std::make_shared<BasicVectorSet>(ByteArray((std::uint8_t*)(p_base->GetData()), p_base->GetVector(p_queryNum) - p_base->GetData(), false), 
                                                                          p_base->GetValueType(), p_base->Dimension(), p_queryNum);
    std::vector<std::vector<SizeType>> truth;
    TruthSet::GenerateTruth(queries, p_base, "", p_distance, p_resultCount, TruthFileType::DEFAULT, nullptr, p_threads, &truth);

    std::shared_ptr<RaBitQQuantizer> selected = nullptr;
    status = SelectMinimumBits(
        p_targetRecall,
        [&](int p_bits, float& p_recall) {
            auto candidate = centroidModel->CloneWithBits(p_bits);
            if (!candidate) {
                p_error = "failed to create RaBitQ candidate from shared centroid";
                return ErrorCode::Fail;
            }
            const ErrorCode evaluation = EvaluateBits(
                candidate, p_base, queries, truth,
                p_resultCount, p_threads, p_recall);
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Info,
                "RaBitQ auto-tuning bits=%d Recall@%d=%.6f target=%.6f\n",
                p_bits, p_resultCount, p_recall, p_targetRecall);
            if (evaluation == ErrorCode::Success && p_recall >= p_targetRecall) {
                selected = std::move(candidate);
            }
            return evaluation;
        },
        p_result.selectedBits, p_result.recall);
    if (status != ErrorCode::Success || !selected) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "no RaBitQ bit width in the fixed range 1..8 meets TargetRecall\n");
        return ErrorCode::Fail;
    }
    
    const fs::path folder(p_outputFolder);
    const fs::path modelFinal = folder / "rabitq_auto_quantizer.bin";
    auto modelOutput = f_createIO();
    if (!modelOutput ||
        !modelOutput->Initialize(
            modelFinal.string().c_str(), std::ios::out | std::ios::binary) ||
        selected->SaveQuantizer(modelOutput) != ErrorCode::Success) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write RaBitQ quantizer!\n");
        return ErrorCode::DiskIOFail;
    }

    p_result.codeDimension = selected->GetNumSubvectors();
    p_result.quantizerPath = modelFinal.string();    
    p_result.quantizer = selected;
    SPTAGLIB_LOG(
        Helper::LogLevel::LL_Info,
        "RaBitQ auto-tuning selected %d bits (Recall@%d=%.6f); encoded vectors at width %d\n",
        p_result.selectedBits, resultCount, p_result.recall, p_result.codeDimension);
    return ErrorCode::Success;
}

} // namespace COMMON
} // namespace SPTAG
