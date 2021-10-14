#pragma once
#include <limits>

#include "inc/Core/Common.h"
#include "inc/Helper/StringConvert.h"

namespace SPTAG {
	namespace SSDServing {
		namespace VectorSearch {
			class Options {
			public:
				// Both Building and Searching
				bool m_execute;
				bool m_buildSsdIndex;
				int m_internalResultNum;
				int m_iNumberOfThreads;
				std::string m_headConfig;

				// Building
				int m_replicaCount;
				int m_postingPageLimit;
				bool m_outputEmptyReplicaID;

				// Searching
				std::string m_searchResult;
				std::string m_logFile;
				int m_qpsLimit;
				int m_resultNum;
				int m_queryCountLimit;
				int m_maxCheck;

				Options() {
#define DefineSSDParameter(VarName, VarType, DefaultValue, RepresentStr) \
                VarName = DefaultValue; \

#include "inc/SSDServing/VectorSearch/ParameterDefinitionList.h"
#undef DefineSSDParameter
				}

				~Options() {}

				ErrorCode SetParameter(const char* p_param, const char* p_value)
				{
					if (nullptr == p_param || nullptr == p_value) return ErrorCode::Fail;

#define DefineSSDParameter(VarName, VarType, DefaultValue, RepresentStr) \
    else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
    { \
        LOG(Helper::LogLevel::LL_Info, "Setting %s with value %s\n", RepresentStr, p_value); \
        VarType tmp; \
        if (SPTAG::Helper::Convert::ConvertStringTo<VarType>(p_value, tmp)) \
        { \
            VarName = tmp; \
        } \
    } \

#include "inc/SSDServing/VectorSearch/ParameterDefinitionList.h"
#undef DefineSSDParameter

					return ErrorCode::Success;
				}
			};
		}
	}
}