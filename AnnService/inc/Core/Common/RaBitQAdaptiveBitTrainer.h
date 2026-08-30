// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "inc/Core/Common.h"
#include "inc/Core/VectorSet.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace SPTAG
{
namespace COMMON
{

class RaBitQAdaptiveBitTrainer
{
public:
    struct Config
    {
        DimensionType dimension = 0;
        DistCalcMethod metric = DistCalcMethod::Undefined;
        SizeType baseCount = -1;
        int queryCount = 1000;
        int truthDepth = 1000;
        int recallAt = 10;
        float targetRecallError = -1.0F;
        std::string queryPath;
        VectorFileType queryType = VectorFileType::Undefined;
        std::string truthPath;
        TruthFileType truthType = TruthFileType::Undefined;
        std::string basePath;
        VectorFileType baseType = VectorFileType::Undefined;
        std::string trainingDataFile;
        std::string trainingResultFile;
        std::string modelFile;
    };

    struct BitMetrics
    {
        int bits = 0;
        float measuredRecall = 0.0F;
        float certifiedRecallLowerBound = 0.0F;
    };

    struct Result
    {
        int selectedBits = 0;
        std::uint64_t artifactFingerprint = 0;
        std::uint64_t modelFingerprint = 0;
        bool reused = false;
        std::vector<BitMetrics> metrics;
    };

    static ErrorCode Run(const Config& p_config,
                         const std::shared_ptr<VectorSet>& p_memoryBase,
                         Result& p_result);
};

} // namespace COMMON
} // namespace SPTAG
