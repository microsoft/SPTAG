// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "inc/Core/Common/RaBitQQuantizer.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/VectorSetReader.h"

#include <functional>
#include <string>
#include <vector>

namespace SPTAG
{
namespace COMMON
{

struct RaBitQAutoTuneResult
{
    int selectedBits = 0;
    float recall = 0.0F;
    SizeType vectorCount = 0;
    DimensionType codeDimension = 0;
    std::string quantizerPath;
    std::string vectorPath;
    std::shared_ptr<RaBitQQuantizer> quantizer;
};

class RaBitQAutoTuner
{
public:
    using BitEvaluator = std::function<ErrorCode(int, float&)>;

    static bool IsEnabled(const Helper::IniReader& p_config);
    static ErrorCode Run(Helper::IniReader& p_config,
                         const std::string& p_outputFolder,
                         RaBitQAutoTuneResult& p_result,
                         std::string& p_error);

    static ErrorCode SelectMinimumBits(float p_targetRecall,
                                       const BitEvaluator& p_evaluator,
                                       int& p_selectedBits,
                                       float& p_selectedRecall);
    static ErrorCode ValidateTruth(const std::vector<std::vector<SizeType>>& p_truth,
                                   SizeType p_baseCount,
                                   SizeType p_queryCount,
                                   int p_resultCount,
                                   std::string& p_error);
    static float RecallAtK(const std::vector<SizeType>& p_exact,
                           const std::vector<SizeType>& p_reranked,
                           int p_resultCount);
};

} // namespace COMMON
} // namespace SPTAG
