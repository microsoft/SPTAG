// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "inc/SSDServing/SelectHead_BKT/Options.h"

namespace SPTAG {
	namespace SSDServing {
		namespace SelectHead_BKT {
			inline int CalcHeadCnt(double p_ratio, int p_vectorCount)
			{
				return static_cast<int>(std::round(p_ratio * p_vectorCount));
			}

			ErrorCode SelectHead(std::shared_ptr<VectorSet> vs, std::shared_ptr<COMMON::BKTree> bkt, Options& opts, std::unordered_map<int, int>& counter, std::vector<int>& selected);
		}
	}
}
