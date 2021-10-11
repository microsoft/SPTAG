// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "inc/Core/Common.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/CommonHelper.h"
#include <string>

namespace SPTAG {
	namespace SSDServing {
            class BaseOptions
            {
            public:
                SPTAG::VectorValueType m_valueType;
                SPTAG::DistCalcMethod m_distCalcMethod;
                SPTAG::IndexAlgoType m_indexAlgoType;
                SPTAG::DimensionType m_dim; 
                std::string m_vectorPath;
                SPTAG::VectorFileType m_vectorType;
                SPTAG::SizeType m_vectorSize; //Optional on condition
                std::string m_vectorDelimiter; //Optional on condition
                std::string m_queryPath;
                SPTAG::VectorFileType m_queryType;
                SPTAG::SizeType m_querySize; //Optional on condition
                std::string m_queryDelimiter; //Optional on condition
                std::string m_warmupPath;
                SPTAG::VectorFileType m_warmupType;
                SPTAG::SizeType m_warmupSize; //Optional on condition
                std::string m_warmupDelimiter; //Optional on condition
                std::string m_truthPath;
                SPTAG::TruthFileType m_truthType;
                bool m_generateTruth;
                std::string m_indexDirectory;
                std::string m_headIDFile;
                std::string m_headVectorFile;
                std::string m_headIndexFolder;
                std::string m_ssdIndex;
                bool m_deleteHeadVectors;
                int m_ssdIndexFileNum;

                BaseOptions() {
#define DefineBasicParameter(VarName, VarType, DefaultValue, RepresentStr) \
                VarName = DefaultValue; \

#include "inc/SSDServing/IndexBuildManager/ParameterDefinitionList.h"
#undef DefineBasicParameter
                }

                ~BaseOptions() {}

                ErrorCode SetParameter(const char* p_param, const char* p_value)
                {
                    if (nullptr == p_param || nullptr == p_value) return ErrorCode::Fail;

#define DefineBasicParameter(VarName, VarType, DefaultValue, RepresentStr) \
    else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
    { \
        LOG(Helper::LogLevel::LL_Info, "Setting %s with value %s\n", RepresentStr, p_value); \
        VarType tmp; \
        if (SPTAG::Helper::Convert::ConvertStringTo<VarType>(p_value, tmp)) \
        { \
            VarName = tmp; \
        } \
    } \

#include "inc/SSDServing/IndexBuildManager/ParameterDefinitionList.h"
#undef DefineBasicParameter

                    return ErrorCode::Success;
                }
            };
	}
}