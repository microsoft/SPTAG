// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_OPTIONS_H_
#define _SPTAG_SPANN_OPTIONS_H_

#include "inc/Core/Common.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/CommonHelper.h"
#include "inc/Helper/KeyValueIO.h"
#include <memory>
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
            bool m_ratioExplicitlySet;
            int m_headVectorCount;
            bool m_recursiveCheckSmallCluster;
            bool m_printSizeCount;
            std::string m_selectType;
            std::string m_perVectorTagsFile;
            bool m_dualPoolAugment;
            double m_dualPoolExtraRatio;
            std::string m_uExtraIDFile;
            bool m_parallelBKTBuild;

            // Section 3: for build head
            bool m_buildHead;

            // Section 4: for build ssd and search ssd
            bool m_enableSSD;
            bool m_buildSsdIndex;
            int m_iSSDNumberOfThreads;
            bool m_enableDeltaEncoding;
            bool m_enablePostingListRearrange;
            bool m_enableOrderedPageStart;
            std::string m_orderedPageStartAttrs;
            bool m_enableDataCompression;
            bool m_enableDictTraining;
            int m_minDictTraingBufferSize;
            int m_dictBufferCapacity;
            int m_zstdCompressLevel;

            // Building
            int m_replicaCount;
            int m_tailReplicaCount;
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
            std::string m_postingPureCountsFile;
            bool m_useDirectIO;
            bool m_preReassign;
            float m_preReassignRatio;
            bool m_enableWAL;
            bool m_disableCheckpoint;
            std::string m_headRoleFile;

            // Per-vector tags embedded in posting metadata
            int m_numTagsPerVec;
            // Number of leading tag columns used by static ACL filtering.
            // Zero preserves the legacy behavior of using every tag column.
            int m_staticACLTagCols;
            // Build-time cross-subgraph sidecar used by STATIC unfilter-tail
            // construction and retained for runtime unified traversal.
            bool m_buildCrossEdges;
            int m_crossExtraEdges;

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
            int m_seedMaxCheck;
            bool m_collectPostingContributionStats;
            bool m_forceDenseTagSearch;
            int m_directSparseMaxPostings;
            float m_filteredSearchNprobeSafety;
            float m_filteredSearchTargetRecall;
            float m_filteredSearchCoverageExponent;
            bool m_enableAdaptiveFilteredNprobe;
            bool m_logAdaptiveNprobe;
            bool m_logPhaseTime;
            bool m_unifiedNprobeBudget;
            double m_multiNodeBudgetKeepRatio;
            int m_crossSingleSeed;
            bool m_disableCrossEdges;
            bool m_filterKeepCross;
            int m_crossExpandLimit;
            bool m_disableCrossSubgraph;
            bool m_logUExtra;
            bool m_logCrossStats;
            bool m_logPathStats;
            int m_dumpHeads;
            bool m_filterKeepUExtra;
            bool m_enableUnfilterTail;
            bool m_ablateUExtra;
            bool m_ablateTail;
            bool m_unfilterPurePages;
            int m_unfilterExtraTailPages;
            bool m_enableHierPostingFilter;
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
            int m_postingOffset;
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
            std::string m_updateVectorFile;

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
            int m_unfilterTailBufferLength;
            int m_maxFileSize;
            int m_startFileSize;
            int m_growthFileSize;
            float m_growThreshold;
            float m_fDeletePercentageForRefine;
            bool m_oneClusterCutMax;
            bool m_consistencyCheck;
            bool m_checksumCheck;
            bool m_checksumInRead;
            int m_cacheSize;
            int m_cacheShards;
            bool m_asyncMergeInSearch;
            bool m_centeringToZero;

            // Iterative
            int m_headBatch;
            int m_asyncAppendQueueSize;
            bool m_allowZeroReplica;

            // ShareDB: when true, ExtraDynamicSearcher will use the externally-provided
            // m_externalDB (typically a Helper::TenantPrefixedKeyValueIO wrapping a
            // shared RocksDB instance) instead of allocating its own per-tenant
            // RocksDBIO. The flag is exposed via the parameter system; m_externalDB
            // is a runtime-only handle set programmatically (e.g., by
            // SPANN::Index::SetSharedDB or by TenantIndexManager).
            bool m_shareDB;
            std::shared_ptr<Helper::KeyValueIO> m_externalDB;

            // Primary-head CSR bypass for sparse categorical filters.
            bool m_buildPrimaryHeadCSR;
            std::string m_primaryHeadCSRFile;
            bool m_enablePrimaryHeadBypass;
            int m_primaryHeadBypassRerankL;

            // In-posting quantization (unified config interface). See ParameterDefinitionList.h.
            std::string m_postingQuantizer;   // None|RaBitQ|OPQ|PipePQ
            int m_postingQuantBits;           // RaBitQ bits per dim
            int m_postingQuantM;              // OPQ/PipePQ code bytes per vector
            bool m_requantizeFromPipePQ;      // one-time same-stride PipePQ->OPQ posting rewrite
            bool m_quantizeHead;              // quantize the head index too
            std::string m_postingQuantFile;   // code sidecar path
            std::string m_pipePQPivotsFile;   // PipeANN PQ pivot sidecar path
            std::string m_fullVectorFile;     // full-precision base for cold rerank
            int m_rerankL;                    // exact-rerank depth (0 = default)
            bool m_quantADCOnly;              // skip rerank, return ADC/estimate order

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
                m_ratioExplicitlySet = false;
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