#pragma once
#include "inc/Core/Common.h"
#include "inc/SSDServing/BuildHead/Options.h"

namespace SPTAG {
	namespace SSDServing {
		namespace BuildHead {
			ErrorCode Bootstrap(Options& options, const SPTAG::Helper::IniReader::ParameterValueMap& params);
		}
	}
}
