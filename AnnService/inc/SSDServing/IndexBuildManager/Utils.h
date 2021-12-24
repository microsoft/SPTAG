// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <vector>
#include "inc/Core/Common.h"
#include <algorithm>
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/VectorSet.h"
#include "inc/SSDServing/VectorSearch/Options.h"
#include "inc/Helper/SimpleIniReader.h"

namespace SPTAG {
    namespace SSDServing {

        template<typename T>
        float Std(T* nums, size_t size) {
            float var = 0;

            float mean = 0;
            for (size_t i = 0; i < size; i++)
            {
                mean += nums[i] / static_cast<float>(size);
            }

            for (size_t i = 0; i < size; i++)
            {
				float temp = nums[i] - mean;
                var += (float) pow(temp, 2.0);
            }
            var /= static_cast<float>(size);

            return sqrt(var);
        }

		bool readSearchSSDSec(const char* iniFile, VectorSearch::Options& opts);
		bool readSearchSSDSec(const Helper::IniReader& iniFileReader, VectorSearch::Options& opts);
    }
}