// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef DefineBasicParameter

// DefineBasicParameter(VarName, VarType, DefaultValue, RepresentStr)
DefineBasicParameter(m_valueType, SPTAG::VectorValueType, SPTAG::VectorValueType::Undefined, "ValueType")
DefineBasicParameter(m_distCalcMethod, SPTAG::DistCalcMethod, SPTAG::DistCalcMethod::Undefined, "DistCalcMethod")
DefineBasicParameter(m_indexAlgoType, SPTAG::IndexAlgoType, SPTAG::IndexAlgoType::KDT, "IndexAlgoType")
DefineBasicParameter(m_dim, SPTAG::DimensionType, -1, "Dim")
DefineBasicParameter(m_vectorPath, std::string, std::string(""), "VectorPath")
DefineBasicParameter(m_vectorType, SPTAG::VectorFileType, SPTAG::VectorFileType::DEFAULT, "VectorType")
DefineBasicParameter(m_vectorSize, SPTAG::SizeType, -1, "VectorSize")
DefineBasicParameter(m_vectorDelimiter, std::string, std::string("|"), "VectorDelimiter")
DefineBasicParameter(m_queryPath, std::string, std::string(""), "QueryPath")
DefineBasicParameter(m_queryType, SPTAG::VectorFileType, SPTAG::VectorFileType::Undefined, "QueryType")
DefineBasicParameter(m_querySize, SPTAG::SizeType, -1, "QuerySize")
DefineBasicParameter(m_queryDelimiter, std::string, std::string("|"), "QueryDelimiter")
DefineBasicParameter(m_warmupPath, std::string, std::string(""), "WarmupPath")
DefineBasicParameter(m_warmupType, SPTAG::VectorFileType, SPTAG::VectorFileType::Undefined, "WarmupType")
DefineBasicParameter(m_warmupSize, SPTAG::SizeType, -1, "WarmupSize")
DefineBasicParameter(m_warmupDelimiter, std::string, std::string("|"), "WarmupDelimiter")
DefineBasicParameter(m_truthPath, std::string, std::string(""), "TruthPath")
DefineBasicParameter(m_truthType, SPTAG::TruthFileType, SPTAG::TruthFileType::Undefined, "TruthType")
DefineBasicParameter(m_generateTruth, bool, false, "GenerateTruth")
DefineBasicParameter(m_indexDirectory, std::string, std::string("SPANN"), "IndexDirectory")
DefineBasicParameter(m_headIDFile, std::string, std::string("SPTAGHeadVectorIDs.bin"), "HeadVectorIDs")
DefineBasicParameter(m_deleteIDFile, std::string, std::string("DeletedIDs.bin"), "DeletedIDs")
DefineBasicParameter(m_headVectorFile, std::string, std::string("SPTAGHeadVectors.bin"), "HeadVectors")
DefineBasicParameter(m_headIndexFolder, std::string, std::string("HeadIndex"), "HeadIndexFolder")
DefineBasicParameter(m_ssdIndex, std::string, std::string("SPTAGFullList.bin"), "SSDIndex")
DefineBasicParameter(m_deleteHeadVectors, bool, false, "DeleteHeadVectors")
DefineBasicParameter(m_ssdIndexFileNum, int, 1, "SSDIndexFileNum")
DefineBasicParameter(m_quantizerFilePath, std::string, std::string(), "QuantizerFilePath")
DefineBasicParameter(m_datasetRowsInBlock, int, 1024 * 1024, "DataBlockSize")
DefineBasicParameter(m_datasetCapacity, int, SPTAG::MaxSize, "DataCapacity")
#endif

#ifdef DefineSelectHeadParameter

DefineSelectHeadParameter(m_selectHead, bool, false, "isExecute")
DefineSelectHeadParameter(m_iTreeNumber, int, 1, "TreeNumber")
DefineSelectHeadParameter(m_iBKTKmeansK, int, 32, "BKTKmeansK")
DefineSelectHeadParameter(m_iBKTLeafSize, int, 8, "BKTLeafSize")
DefineSelectHeadParameter(m_iSamples, int, 1000, "SamplesNumber")
DefineSelectHeadParameter(m_fBalanceFactor, float, -1.0F, "BKTLambdaFactor")

DefineSelectHeadParameter(m_iSelectHeadNumberOfThreads, int, 4, "NumberOfThreads")
DefineSelectHeadParameter(m_saveBKT, bool, false, "SaveBKT")

DefineSelectHeadParameter(m_analyzeOnly, bool, false, "AnalyzeOnly")
DefineSelectHeadParameter(m_calcStd, bool, false, "CalcStd")
DefineSelectHeadParameter(m_selectDynamically, bool, true, "SelectDynamically")
DefineSelectHeadParameter(m_noOutput, bool, false, "NoOutput")

DefineSelectHeadParameter(m_selectThreshold, int, 6, "SelectThreshold")
DefineSelectHeadParameter(m_splitFactor, int, 5, "SplitFactor")
DefineSelectHeadParameter(m_splitThreshold, int, 25, "SplitThreshold")
DefineSelectHeadParameter(m_maxRandomTryCount, int, 8, "SplitMaxTry")
DefineSelectHeadParameter(m_ratio, double, 0.2, "Ratio")
DefineSelectHeadParameter(m_headVectorCount, int, 0, "Count")
DefineSelectHeadParameter(m_recursiveCheckSmallCluster, bool, true, "RecursiveCheckSmallCluster")
DefineSelectHeadParameter(m_printSizeCount, bool, true, "PrintSizeCount")
DefineSelectHeadParameter(m_selectType, std::string, "BKT", "SelectHeadType")
#endif

#ifdef DefineBuildHeadParameter

DefineBuildHeadParameter(m_buildHead, bool, false, "isExecute")

#endif

#ifdef DefineSSDParameter
DefineSSDParameter(m_enableSSD, bool, false, "isExecute")
DefineSSDParameter(m_buildSsdIndex, bool, false, "BuildSsdIndex")
DefineSSDParameter(m_iSSDNumberOfThreads, int, 16, "NumberOfThreads")
DefineSSDParameter(m_enableDeltaEncoding, bool, false, "EnableDeltaEncoding")
DefineSSDParameter(m_enablePostingListRearrange, bool, false, "EnablePostingListRearrange")
DefineSSDParameter(m_enableDataCompression, bool, false, "EnableDataCompression")
DefineSSDParameter(m_enableDictTraining, bool, true, "EnableDictTraining")
DefineSSDParameter(m_minDictTraingBufferSize, int, 10240000, "MinDictTrainingBufferSize")
DefineSSDParameter(m_dictBufferCapacity, int, 204800, "DictBufferCapacity")
DefineSSDParameter(m_zstdCompressLevel, int, 0, "ZstdCompressLevel")

// Building
DefineSSDParameter(m_internalResultNum, int, 64, "InternalResultNum")
DefineSSDParameter(m_postingPageLimit, int, 3, "PostingPageLimit")
DefineSSDParameter(m_replicaCount, int, 8, "ReplicaCount")
DefineSSDParameter(m_outputEmptyReplicaID, bool, false, "OutputEmptyReplicaID")
DefineSSDParameter(m_batches, int, 1, "Batches")
DefineSSDParameter(m_tmpdir, std::string, std::string("."), "TmpDir")
DefineSSDParameter(m_rngFactor, float, 1.0f, "RNGFactor")
DefineSSDParameter(m_samples, int, 100, "RecallTestSampleNumber")
DefineSSDParameter(m_excludehead, bool, true, "ExcludeHead")
DefineSSDParameter(m_postingVectorLimit, int, 118, "PostingVectorLimit")
DefineSSDParameter(m_fullDeletedIDFile, std::string, std::string("fulldeleted"), "FullDeletedIDFile")
DefineSSDParameter(m_useKV, bool, false, "UseKV")
DefineSSDParameter(m_useSPDK, bool, false, "UseSPDK")
DefineSSDParameter(m_useFileIO, bool, false, "UseFileIO")
DefineSSDParameter(m_spdkBatchSize, int, 64, "SpdkBatchSize")
DefineSSDParameter(m_KVPath, std::string, std::string(""), "KVPath")
DefineSSDParameter(m_spdkMappingPath, std::string, std::string(""), "SpdkMappingPath")
DefineSSDParameter(m_ssdInfoFile, std::string, std::string(""), "SsdInfoFile")
DefineSSDParameter(m_useDirectIO, bool, false, "UseDirectIO")
DefineSSDParameter(m_preReassign, bool, false, "PreReassign")
DefineSSDParameter(m_preReassignRatio, float, 0.7f, "PreReassignRatio")
DefineSSDParameter(m_bufferLength, int, 3, "BufferLength")
DefineSSDParameter(m_enableWAL, bool, false, "EnableWAL")

// GPU Building
DefineSSDParameter(m_gpuSSDNumTrees, int, 100, "GPUSSDNumTrees")
DefineSSDParameter(m_gpuSSDLeafSize, int, 200, "GPUSSDLeafSize")
DefineSSDParameter(m_numGPUs, int, 1, "NumGPUs")

// Searching
DefineSSDParameter(m_searchResult, std::string, std::string(""), "SearchResult")
DefineSSDParameter(m_logFile, std::string, std::string(""), "LogFile")
DefineSSDParameter(m_qpsLimit, int, 0, "QpsLimit")
DefineSSDParameter(m_resultNum, int, 5, "ResultNum")
DefineSSDParameter(m_truthResultNum, int, -1, "TruthResultNum")
DefineSSDParameter(m_maxCheck, int, 4096, "MaxCheck")
DefineSSDParameter(m_hashExp, int, 4, "HashTableExponent")
DefineSSDParameter(m_queryCountLimit, int, (std::numeric_limits<int>::max)(), "QueryCountLimit")
DefineSSDParameter(m_maxDistRatio, float, 10000, "MaxDistRatio")
DefineSSDParameter(m_ioThreads, int, 4, "IOThreadsPerHandler")
DefineSSDParameter(m_searchInternalResultNum, int, 64, "SearchInternalResultNum")
DefineSSDParameter(m_searchPostingPageLimit, int, 3, "SearchPostingPageLimit")
DefineSSDParameter(m_rerank, int, 0, "Rerank")
DefineSSDParameter(m_enableADC, bool, false, "EnableADC")
DefineSSDParameter(m_recall_analysis, bool, false, "RecallAnalysis")
DefineSSDParameter(m_debugBuildInternalResultNum, int, 64, "DebugBuildInternalResultNum")
DefineSSDParameter(m_iotimeout, int, 30, "IOTimeout")

// Calculating
// TruthFilePrefix
DefineSSDParameter(m_truthFilePrefix, std::string, std::string(""), "TruthFilePrefix")
// CalTruth
DefineSSDParameter(m_calTruth, bool, true, "CalTruth")
DefineSSDParameter(m_onlySearchFinalBatch, bool, false, "OnlySearchFinalBatch")
// Search multiple times for stable result
DefineSSDParameter(m_searchTimes, int, 1, "SearchTimes")
// Frontend search threadnum
DefineSSDParameter(m_searchThreadNum, int, 16, "SearchThreadNum")
// Show tradeoff of latency and acurracy
DefineSSDParameter(m_minInternalResultNum, int, -1, "MinInternalResultNum")
DefineSSDParameter(m_stepInternalResultNum, int, -1, "StepInternalResultNum")
DefineSSDParameter(m_maxInternalResultNum, int, -1, "MaxInternalResultNum")

// Updating(SPFresh Update Test)
// For update mode: current only update
DefineSSDParameter(m_update, bool, false, "Update")
// For Test Mode
DefineSSDParameter(m_inPlace, bool, false, "InPlace")
DefineSSDParameter(m_outOfPlace, bool, false, "OutOfPlace")
// latency limit
DefineSSDParameter(m_latencyLimit, float, 10.0, "LatencyLimit")
// Update batch size
DefineSSDParameter(m_step, int, 0, "Step")
// Frontend update threadnum
DefineSSDParameter(m_insertThreadNum, int, 16, "InsertThreadNum")
// Update limit
DefineSSDParameter(m_endVectorNum, int, -1, "EndVectorNum")
// Persistent buffer path
DefineSSDParameter(m_persistentBufferPath, std::string, std::string(""), "PersistentBufferPath")
// Background append threadnum
DefineSSDParameter(m_appendThreadNum, int, 16, "AppendThreadNum")
// Background reassign threadnum
DefineSSDParameter(m_reassignThreadNum, int, 16, "ReassignThreadNum")
// Background process batch size
DefineSSDParameter(m_batch, int, 1000, "Batch")
// Total Vector Path
DefineSSDParameter(m_fullVectorPath, std::string, std::string(""), "FullVectorPath")
// Steady State: update trace
DefineSSDParameter(m_updateFilePrefix, std::string, std::string(""), "UpdateFilePrefix")
// Steady State: update mapping
DefineSSDParameter(m_updateMappingPrefix, std::string, std::string(""), "UpdateMappingPrefix")
// Steady State: days
DefineSSDParameter(m_days, int, 0, "Days")
// Steady State: deleteQPS
DefineSSDParameter(m_deleteQPS, int, -1, "DeleteQPS")
// Steady State: sampling
DefineSSDParameter(m_sampling, int, -1, "Sampling")
// Steady State: showUpdateProgress
DefineSSDParameter(m_showUpdateProgress, bool, true, "ShowUpdateProgress")
// Steady State: Merge Threshold
DefineSSDParameter(m_mergeThreshold, int, 10, "MergeThreshold")
// Steady State: showUpdateProgress
DefineSSDParameter(m_loadAllVectors, bool, false, "LoadAllVectors")
// Steady State: steady state
DefineSSDParameter(m_steadyState, bool, false, "SteadyState")
// Steady State: stress test
DefineSSDParameter(m_stressTest, bool, false, "StressTest")

// SPANN
DefineSSDParameter(m_disableReassign, bool, false, "DisableReassign")
DefineSSDParameter(m_searchDuringUpdate, bool, false, "SearchDuringUpdate")
DefineSSDParameter(m_reassignK, int, 0, "ReassignK")
DefineSSDParameter(m_recovery, bool, false, "Recovery")

// Iterative
DefineSSDParameter(m_headBatch, int, 32, "IterativeSearchHeadBatch")

// Calculating
// TruthFilePrefix
DefineSSDParameter(m_truthFilePrefix, std::string, std::string(""), "TruthFilePrefix")
// CalTruth
DefineSSDParameter(m_calTruth, bool, true, "CalTruth")
DefineSSDParameter(m_onlySearchFinalBatch, bool, false, "OnlySearchFinalBatch")
// Search multiple times for stable result
DefineSSDParameter(m_searchTimes, int, 1, "SearchTimes")
// Frontend search threadnum
DefineSSDParameter(m_searchThreadNum, int, 16, "SearchThreadNum")
// Show tradeoff of latency and acurracy
DefineSSDParameter(m_minInternalResultNum, int, -1, "MinInternalResultNum")
DefineSSDParameter(m_stepInternalResultNum, int, -1, "StepInternalResultNum")
DefineSSDParameter(m_maxInternalResultNum, int, -1, "MaxInternalResultNum")

// Updating(SPFresh Update Test)
// For update mode: current only update
DefineSSDParameter(m_update, bool, false, "Update")
// For Test Mode
DefineSSDParameter(m_inPlace, bool, false, "InPlace")
DefineSSDParameter(m_outOfPlace, bool, false, "OutOfPlace")
// latency limit
DefineSSDParameter(m_latencyLimit, float, 10.0, "LatencyLimit")
// Update batch size
DefineSSDParameter(m_step, int, 0, "Step")
// Frontend update threadnum
DefineSSDParameter(m_insertThreadNum, int, 16, "InsertThreadNum")
// Update limit
DefineSSDParameter(m_endVectorNum, int, -1, "EndVectorNum")
// Persistent buffer path
DefineSSDParameter(m_persistentBufferPath, std::string, std::string(""), "PersistentBufferPath")
// Background append threadnum
DefineSSDParameter(m_appendThreadNum, int, 16, "AppendThreadNum")
// Background reassign threadnum
DefineSSDParameter(m_reassignThreadNum, int, 16, "ReassignThreadNum")
// Background process batch size
DefineSSDParameter(m_batch, int, 1000, "Batch")
// Total Vector Path
DefineSSDParameter(m_fullVectorPath, std::string, std::string(""), "FullVectorPath")
// Steady State: update trace
DefineSSDParameter(m_updateFilePrefix, std::string, std::string(""), "UpdateFilePrefix")
// Steady State: update mapping
DefineSSDParameter(m_updateMappingPrefix, std::string, std::string(""), "UpdateMappingPrefix")
// Steady State: days
DefineSSDParameter(m_days, int, 0, "Days")
// Steady State: deleteQPS
DefineSSDParameter(m_deleteQPS, int, -1, "DeleteQPS")
// Steady State: sampling
DefineSSDParameter(m_sampling, int, -1, "Sampling")
// Steady State: showUpdateProgress
DefineSSDParameter(m_showUpdateProgress, bool, true, "ShowUpdateProgress")
// Steady State: Merge Threshold
DefineSSDParameter(m_mergeThreshold, int, 10, "MergeThreshold")
// Steady State: showUpdateProgress
DefineSSDParameter(m_loadAllVectors, bool, false, "LoadAllVectors")
// Steady State: steady state
DefineSSDParameter(m_steadyState, bool, false, "SteadyState")
// Steady State: stress test
DefineSSDParameter(m_stressTest, bool, false, "StressTest")

// SPANN
DefineSSDParameter(m_disableReassign, bool, false, "DisableReassign")
DefineSSDParameter(m_searchDuringUpdate, bool, false, "SearchDuringUpdate")
DefineSSDParameter(m_reassignK, int, 0, "ReassignK")
DefineSSDParameter(m_recovery, bool, false, "Recovery")
#endif
