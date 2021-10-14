#pragma once
#include "inc/Core/Common.h"
#include "inc/Helper/VectorSetReader.h"
#include "omp.h"
#include <string>
#include <vector>
#include <memory>

namespace SPTAG {
    namespace SSDServing {
        namespace BuildHead {
            class Options
            {
            public:
                bool m_execute;

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
        LOG(Helper::LogLevel::LL_Info, "Setting %s with value %s\n", RepresentStr, p_value); \
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