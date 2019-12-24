#pragma once
#include "inc/SSDServing/Common/stdafx.h"
#include "inc/Core/Common.h"
#include "inc/Helper/VectorSetReader.h"
#include <string>
#include <vector>
#include <memory>

namespace SPTAG {
    namespace SSDServing {
        namespace BuildHead {
            class Options
            {
            public:
                std::string m_inputFiles;
                std::string m_outputFolder;
                SPTAG::IndexAlgoType m_indexAlgoType;
                std::string m_builderConfigFile;
                VectorValueType m_inputValueType;
                std::uint32_t m_threadNum;

                Options() {
#define DefineBuildHeadParameter(VarName, VarType, DefaultValue, RepresentStr) \
                VarName = DefaultValue; \

#include "inc/SSDServing/BuildHead/ParameterDefinitionList.h"
#undef DefineBuildHeadParameter
                }

                ~Options() {}

                ErrorCode SetParameter(const char* p_param, const char* p_value)
                {
                    if (nullptr == p_param || nullptr == p_value) return ErrorCode::Fail;

#define DefineBuildHeadParameter(VarName, VarType, DefaultValue, RepresentStr) \
    else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
    { \
        fprintf(stderr, "Setting %s with value %s\n", RepresentStr, p_value); \
        VarType tmp; \
        if (SPTAG::Helper::Convert::ConvertStringTo<VarType>(p_value, tmp)) \
        { \
            VarName = tmp; \
        } \
    } \

#include "inc/SSDServing/BuildHead/ParameterDefinitionList.h"
#undef DefineBuildHeadParameter

                    return ErrorCode::Success;
                }
            };
        }
    }
} 