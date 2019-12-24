#pragma once

#include "inc/SSDServing/Common/stdafx.h"

using namespace std;

namespace SPTAG {
	namespace SSDServing {
		namespace SelectHead_BKT {
			ErrorCode SelectHead(BasicVectorSet vs, shared_ptr<COMMON::BKTree> bkt, Options& opts, unordered_map<int, int>& counter);
		}
	}
}
