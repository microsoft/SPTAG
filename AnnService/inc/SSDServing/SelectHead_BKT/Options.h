// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "inc/Core/Common.h"
#include "inc/Helper/StringConvert.h"

namespace SPTAG {
	namespace SSDServing {
		namespace SelectHead_BKT {
			class Options {
			public:
				// Section 2: for building BKT
				bool m_execute;
				int m_iTreeNumber;
				int m_iBKTKmeansK;
				int m_iBKTLeafSize; 
				int m_iSamples;
				float m_fBalanceFactor;
				int m_iNumberOfThreads;
				bool m_saveBKT;
				// Section 3: for selecting head
				// analyze
				bool m_analyzeOnly;
				bool m_calcStd;
				bool m_selectDynamically;
				bool m_noOutput;
				// selection factors
				int m_selectThreshold;
				int m_splitFactor;
				int m_splitThreshold;
				int m_maxRandomTryCount;
				double m_ratio;
				int m_headVectorCount;
				bool m_recursiveCheckSmallCluster;
				bool m_printSizeCount;
				std::string m_selectType;

				Options() {
#define DefineSelectHeadParameter(VarName, VarType, DefaultValue, RepresentStr) \
                VarName = DefaultValue; \

#include "inc/SSDServing/SelectHead_BKT/ParameterDefinitionList.h"
#undef DefineSelectHeadParameter
				}

				~Options() {}

				ErrorCode SetParameter(const char* p_param, const char* p_value)
				{
					if (nullptr == p_param || nullptr == p_value) return ErrorCode::Fail;

#define DefineSelectHeadParameter(VarName, VarType, DefaultValue, RepresentStr) \
    else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
    { \
        LOG(Helper::LogLevel::LL_Info, "Setting %s with value %s\n", RepresentStr, p_value); \
        VarType tmp; \
        if (SPTAG::Helper::Convert::ConvertStringTo<VarType>(p_value, tmp)) \
        { \
            VarName = tmp; \
        } \
    } \

#include "inc/SSDServing/SelectHead_BKT/ParameterDefinitionList.h"
#undef DefineSelectHeadParameter

					return ErrorCode::Success;
				}
			};
		}
	}
}