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
            int m_datasetRowsInBlock;
            int m_datasetCapacity;

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
            int m_postingVectorLimit;
            std::string m_fullDeletedIDFile;
            Storage m_storage;
            std::string m_KVFile;
            std::string m_ssdMappingFile;
            std::string m_ssdInfoFile;
            std::string m_checksumFile;
            bool m_useDirectIO;
            bool m_preReassign;
            float m_preReassignRatio;
            bool m_enableWAL;
            bool m_disableCheckpoint;

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

            int m_searchThreadNum;

            // Calculating
            std::string m_truthFilePrefix;
            bool m_calTruth;
            bool m_calAllTruth;
            int m_searchTimes;
            int m_minInternalResultNum;
            int m_stepInternalResultNum;
            int m_maxInternalResultNum;
            bool m_onlySearchFinalBatch;

            // Updating
            bool m_disableReassign;
            bool m_searchDuringUpdate;
            int m_reassignK;
            bool m_recovery;

            // Updating(SPFresh Update Test)
            bool m_update;
            bool m_inPlace;
            bool m_outOfPlace;
            float m_latencyLimit;
            int m_step;
            int m_insertThreadNum;
            int m_endVectorNum;
            std::string m_persistentBufferPath;
            int m_appendThreadNum;
            int m_reassignThreadNum;
            int m_batch;
            std::string m_fullVectorPath;

            // Steady State Update
            std::string m_updateFilePrefix;
            std::string m_updateMappingPrefix;
            int m_days;
            int m_deleteQPS;
            int m_sampling;
            bool m_showUpdateProgress;
            int m_mergeThreshold;
            bool m_loadAllVectors;
            bool m_steadyState;
            int m_spdkBatchSize;
            bool m_stressTest;
            int m_bufferLength;
            int m_maxFileSize;
            int m_startFileSize;
            int m_growthFileSize;
            float m_growThreshold;
            float m_fDeletePercentageForRefine;
            bool m_oneClusterCutMax;

            // Iterative
            int m_headBatch;

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
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Setting %s with value %s\n", RepresentStr, p_value); \
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
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Setting %s with value %s\n", RepresentStr, p_value); \
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
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Setting %s with value %s\n", RepresentStr, p_value); \
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
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Setting %s with value %s\n", RepresentStr, p_value); \
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