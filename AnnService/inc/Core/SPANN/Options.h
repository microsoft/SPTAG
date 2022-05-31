// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_OPTIONS_H_
#define _SPTAG_SPANN_OPTIONS_H_

#include "inc/Core/Common.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/CommonHelper.h"
#include <string>

namespace SPTAG {
    namespace SPANN {

        class Options
        {
        public:
            VectorValueType m_valueType;
            DistCalcMethod m_distCalcMethod;
            IndexAlgoType m_indexAlgoType;
            DimensionType m_dim;
            std::string m_vectorPath;
            VectorFileType m_vectorType;
            SizeType m_vectorSize; //Optional on condition
            std::string m_vectorDelimiter; //Optional on condition
            std::string m_queryPath;
            VectorFileType m_queryType;
            SizeType m_querySize; //Optional on condition
            std::string m_queryDelimiter; //Optional on condition
            std::string m_warmupPath;
            VectorFileType m_warmupType;
            SizeType m_warmupSize; //Optional on condition
            std::string m_warmupDelimiter; //Optional on condition
            std::string m_truthPath;
            TruthFileType m_truthType;
            bool m_generateTruth;
            std::string m_indexDirectory;
            std::string m_headIDFile;
            std::string m_headVectorFile;
            std::string m_headIndexFolder;
            std::string m_deleteIDFile;
            std::string m_ssdIndex;
            bool m_deleteHeadVectors;
            int m_ssdIndexFileNum;
            std::string m_quantizerFilePath;

            // Section 2: for selecting head
            bool m_selectHead;
            int m_iTreeNumber;
            int m_iBKTKmeansK;
            int m_iBKTLeafSize;
            int m_iSamples;
            float m_fBalanceFactor;
            int m_iSelectHeadNumberOfThreads;
            bool m_saveBKT;
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
            // Dataset constructor args
            int m_datasetRowsInBlock;
            int m_datasetCapacity;

            // Section 3: for build head
            bool m_buildHead;

            // Section 4: for build ssd and search ssd
            bool m_enableSSD;
            bool m_buildSsdIndex;
            int m_iSSDNumberOfThreads;
            bool m_enableDeltaEncoding;
            bool m_enablePostingListRearrange;
            bool m_enableDataCompression;
            bool m_enableDictTraining;
            int m_minDictTraingBufferSize;
            int m_dictBufferCapacity;
            int m_zstdCompressLevel;

            // Building
            int m_replicaCount;
            int m_postingPageLimit;
            int m_internalResultNum;
            bool m_outputEmptyReplicaID;
            int m_batches;
            std::string m_tmpdir;
            float m_rngFactor;
            int m_samples;
            bool m_excludehead;

            // GPU building
            int m_gpuSSDNumTrees;
            int m_gpuSSDLeafSize;
            int m_numGPUs;

            // Searching
            std::string m_searchResult;
            std::string m_logFile;
            int m_qpsLimit;
            int m_resultNum;
            int m_truthResultNum;
            int m_queryCountLimit;
            int m_maxCheck;
            int m_hashExp;
            float m_maxDistRatio;
            int m_ioThreads;
            int m_searchPostingPageLimit;
            int m_searchInternalResultNum;
            int m_rerank;
            bool m_recall_analysis;
            int m_debugBuildInternalResultNum;
            bool m_enableADC;
            int m_iotimeout;

            Options() {
#define DefineBasicParameter(VarName, VarType, DefaultValue, RepresentStr) \
                VarName = DefaultValue; \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBasicParameter

#define DefineSelectHeadParameter(VarName, VarType, DefaultValue, RepresentStr) \
                VarName = DefaultValue; \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSelectHeadParameter

#define DefineBuildHeadParameter(VarName, VarType, DefaultValue, RepresentStr) \
                VarName = DefaultValue; \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBuildHeadParameter

#define DefineSSDParameter(VarName, VarType, DefaultValue, RepresentStr) \
                VarName = DefaultValue; \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSSDParameter
            }

            ~Options() {}

            ErrorCode SetParameter(const char* p_section, const char* p_param, const char* p_value)
            {
                if (nullptr == p_section || nullptr == p_param || nullptr == p_value) return ErrorCode::Fail;

                if (Helper::StrUtils::StrEqualIgnoreCase(p_section, "Base")) {
#define DefineBasicParameter(VarName, VarType, DefaultValue, RepresentStr) \
    if (Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
    { \
        LOG(Helper::LogLevel::LL_Info, "Setting %s with value %s\n", RepresentStr, p_value); \
        VarType tmp; \
        if (Helper::Convert::ConvertStringTo<VarType>(p_value, tmp)) \
        { \
            VarName = tmp; \
        } \
    } else \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBasicParameter

                    ;
                }
                else if (Helper::StrUtils::StrEqualIgnoreCase(p_section, "SelectHead")) {
#define DefineSelectHeadParameter(VarName, VarType, DefaultValue, RepresentStr) \
    if (Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
    { \
        LOG(Helper::LogLevel::LL_Info, "Setting %s with value %s\n", RepresentStr, p_value); \
        VarType tmp; \
        if (Helper::Convert::ConvertStringTo<VarType>(p_value, tmp)) \
        { \
            VarName = tmp; \
        } \
    } else \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSelectHeadParameter

                    ;
                }
                else if (Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildHead")) {
#define DefineBuildHeadParameter(VarName, VarType, DefaultValue, RepresentStr) \
    if (Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
    { \
        LOG(Helper::LogLevel::LL_Info, "Setting %s with value %s\n", RepresentStr, p_value); \
        VarType tmp; \
        if (Helper::Convert::ConvertStringTo<VarType>(p_value, tmp)) \
        { \
            VarName = tmp; \
        } \
    } else \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBuildHeadParameter

                    ;
                }
                else if (Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildSSDIndex")) {
#define DefineSSDParameter(VarName, VarType, DefaultValue, RepresentStr) \
    if (Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
    { \
        LOG(Helper::LogLevel::LL_Info, "Setting %s with value %s\n", RepresentStr, p_value); \
        VarType tmp; \
        if (Helper::Convert::ConvertStringTo<VarType>(p_value, tmp)) \
        { \
            VarName = tmp; \
        } \
    } else \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSSDParameter

                    ;
                }
                return ErrorCode::Success;
            }
            
            std::string GetParameter(const char* p_section, const char* p_param) const
            {
                if (nullptr == p_section || nullptr == p_param) return std::string();

                if (Helper::StrUtils::StrEqualIgnoreCase(p_section, "Base")) {
#define DefineBasicParameter(VarName, VarType, DefaultValue, RepresentStr) \
        if (Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
        { \
            return SPTAG::Helper::Convert::ConvertToString(VarName); \
        } else \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBasicParameter

                    ;
                }
                else if (Helper::StrUtils::StrEqualIgnoreCase(p_section, "SelectHead")) {
#define DefineSelectHeadParameter(VarName, VarType, DefaultValue, RepresentStr) \
        if (Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
        { \
            return SPTAG::Helper::Convert::ConvertToString(VarName); \
        } else \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSelectHeadParameter

                    ;
                }
                else if (Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildHead")) {
#define DefineBuildHeadParameter(VarName, VarType, DefaultValue, RepresentStr) \
        if (Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
        { \
            return SPTAG::Helper::Convert::ConvertToString(VarName); \
        } else \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBuildHeadParameter

                    ;
                }
                else if (Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildSSDIndex")) {
#define DefineSSDParameter(VarName, VarType, DefaultValue, RepresentStr) \
        if (Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
        { \
            return SPTAG::Helper::Convert::ConvertToString(VarName); \
        } else \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSSDParameter

                    ;
                }
                return std::string();
            }
        };
    }
}

#endif // _SPTAG_SPANN_OPTIONS_H_