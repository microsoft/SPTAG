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
    DimensionType codeDimension = 0;
    std::string quantizerPath;
    std::shared_ptr<RaBitQQuantizer> quantizer;
};

template <typename T>
class RaBitQAutoTuner
{
public:
    static ErrorCode Run(std::shared_ptr<VectorSet>& p_base,
                        int p_queryNum, int p_resultCount, int p_threads, float p_targetRecall,
                        DistCalcMethod p_distance,
                        const std::string& p_outputFolder,
                        RaBitQAutoTuneResult& p_result);
    static ErrorCode SelectMinimumBits(float p_targetRecall,
                        const std::function<ErrorCode(int, float&)>& p_evaluator,
                        int& p_selectedBits,
                        float& p_selectedRecall);
};

} // namespace COMMON
} // namespace SPTAG
