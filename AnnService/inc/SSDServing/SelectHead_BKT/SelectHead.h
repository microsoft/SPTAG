#pragma once
#include "inc/SSDServing/SelectHead_BKT/Options.h"

namespace SPTAG {
	namespace SSDServing {
		namespace SelectHead_BKT {
			ErrorCode SelectHead(std::shared_ptr<VectorSet> vs, std::shared_ptr<COMMON::BKTree> bkt, Options& opts, std::unordered_map<int, int>& counter);
		}
	}
}
