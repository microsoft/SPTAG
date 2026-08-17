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
DefineBasicParameter(m_generateTruth, bool, false, "GenerateTruth") // Mutable
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

DefineSelectHeadParameter(m_iSelectHeadNumberOfThreads, int, 4, "NumberOfThreads") // Mutable
DefineSelectHeadParameter(m_saveBKT, bool, false, "SaveBKT")
DefineSelectHeadParameter(m_parallelBKTBuild, bool, false, "ParallelBKTBuild")

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
DefineSelectHeadParameter(m_perVectorTagsFile, std::string, std::string(), "PerVectorTagsFile")
DefineSelectHeadParameter(m_dualPoolAugment, bool, false, "DualPoolAugment")
DefineSelectHeadParameter(m_dualPoolExtraRatio, double, 0.1, "DualPoolExtraRatio")
DefineSelectHeadParameter(m_uExtraIDFile, std::string, std::string(), "UExtraIDFile")
#endif

#ifdef DefineBuildHeadParameter

DefineBuildHeadParameter(m_buildHead, bool, false, "isExecute")

#endif

#ifdef DefineSSDParameter
DefineSSDParameter(m_enableSSD, bool, false, "isExecute")
DefineSSDParameter(m_buildSsdIndex, bool, false, "BuildSsdIndex")
DefineSSDParameter(m_iSSDNumberOfThreads, int, 16, "NumberOfThreads") // Mutable
DefineSSDParameter(m_enableDeltaEncoding, bool, false, "EnableDeltaEncoding")
DefineSSDParameter(m_enablePostingListRearrange, bool, false, "EnablePostingListRearrange")
DefineSSDParameter(m_enableOrderedPageStart, bool, false, "EnableOrderedPageStart")
DefineSSDParameter(m_orderedPageStartAttrs, std::string, std::string(""), "OrderedPageStartAttrs")
DefineSSDParameter(m_enableHybridDistance, bool, false, "EnableHybridDistance")
DefineSSDParameter(m_hybridGenerationFingerprint, std::string, std::string("0"), "HybridGenerationFingerprint")
DefineSSDParameter(m_enableLimitedTagPosting, bool, false, "EnableLimitedTagPosting")
DefineSSDParameter(m_limitedTagGenerationFingerprint, std::string, std::string("0"), "LimitedTagGenerationFingerprint")
DefineSSDParameter(m_limitedTagSupportFile, std::string, std::string("limited_tag_support.bin"), "LimitedTagSupportFile")
DefineSSDParameter(m_limitedTagSlotsPerHead, int, 2, "LimitedTagSlotsPerHead")
DefineSSDParameter(m_limitedTagVoteHeadCount, int, 2, "LimitedTagVoteHeadCount")
DefineSSDParameter(m_limitedTagMinHeadCount, int, 8, "LimitedTagMinHeadCount")
DefineSSDParameter(m_hybridVectorWeight, float, 1.0f, "HybridVectorWeight")
DefineSSDParameter(m_hybridCategoricalCols, std::string, std::string(""), "HybridCategoricalCols")
DefineSSDParameter(m_hybridCategoricalWeights, std::string, std::string(""), "HybridCategoricalWeights")
DefineSSDParameter(m_hybridNumericCols, std::string, std::string(""), "HybridNumericCols")
DefineSSDParameter(m_hybridNumericWeights, std::string, std::string(""), "HybridNumericWeights")
DefineSSDParameter(m_hybridGraphDegree, int, 16, "HybridGraphDegree")
DefineSSDParameter(m_hybridCandidateCount, int, 128, "HybridCandidateCount")
DefineSSDParameter(m_hybridRouteSampleCount, int, 64, "HybridRouteSampleCount")
DefineSSDParameter(m_hybridRouteSelectivityThreshold, float, 0.02f, "HybridRouteSelectivityThreshold")
DefineSSDParameter(m_hybridRouteDeformationThreshold, float, 1.0f, "HybridRouteDeformationThreshold")
DefineSSDParameter(m_logHybridRoute, bool, false, "LogHybridRoute")
DefineSSDParameter(m_enableDataCompression, bool, false, "EnableDataCompression")
DefineSSDParameter(m_enableDictTraining, bool, true, "EnableDictTraining")
DefineSSDParameter(m_minDictTraingBufferSize, int, 10240000, "MinDictTrainingBufferSize")
DefineSSDParameter(m_dictBufferCapacity, int, 204800, "DictBufferCapacity")
DefineSSDParameter(m_zstdCompressLevel, int, 0, "ZstdCompressLevel")

// Building
DefineSSDParameter(m_internalResultNum, int, 64, "InternalResultNum")
DefineSSDParameter(m_postingPageLimit, int, 3, "PostingPageLimit")
DefineSSDParameter(m_replicaCount, int, 8, "ReplicaCount")
// Independent tail replica Kmax for the unfilter-tail. Each base vector considers
// its top-m_tailReplicaCount nearest heads for the tag-agnostic tail region, then
// the build trims per-head tail by page budget (currently <=2 pages, sparse page-2
// dropped). Decoupled from m_replicaCount above, which governs the per-tag pure
// posting region. Default 0 = unfilter-tail disabled.
DefineSSDParameter(m_tailReplicaCount, int, 0, "TailReplicaCount")
DefineSSDParameter(m_outputEmptyReplicaID, bool, false, "OutputEmptyReplicaID")
DefineSSDParameter(m_batches, int, 1, "Batches")
DefineSSDParameter(m_tmpdir, std::string, std::string("."), "TmpDir")
DefineSSDParameter(m_rngFactor, float, 1.0f, "RNGFactor")
DefineSSDParameter(m_samples, int, 100, "RecallTestSampleNumber")
DefineSSDParameter(m_excludehead, bool, true, "ExcludeHead")
DefineSSDParameter(m_postingVectorLimit, int, 118, "PostingVectorLimit")
DefineSSDParameter(m_fullDeletedIDFile, std::string, std::string("fulldeleted"), "FullDeletedIDFile")
DefineSSDParameter(m_storage, SPTAG::Storage, SPTAG::Storage::STATIC, "Storage")
DefineSSDParameter(m_spdkBatchSize, int, 64, "SpdkBatchSize")
DefineSSDParameter(m_KVFile, std::string, std::string("rocksdb"), "KVFile")
DefineSSDParameter(m_ssdMappingFile, std::string, std::string("ssdmapping"), "SsdMappingFile")
DefineSSDParameter(m_ssdInfoFile, std::string, std::string("ssdinfo"), "SsdInfoFile")
DefineSSDParameter(m_checksumFile, std::string, std::string("checksum"), "ChecksumFile")
DefineSSDParameter(m_postingPureCountsFile, std::string, std::string("posting_pure_counts.bin"), "PostingPureCountsFile")
DefineSSDParameter(m_useDirectIO, bool, false, "UseDirectIO")
DefineSSDParameter(m_preReassign, bool, false, "PreReassign")
DefineSSDParameter(m_preReassignRatio, float, 0.7f, "PreReassignRatio")
DefineSSDParameter(m_bufferLength, int, 3, "BufferLength")
DefineSSDParameter(m_unfilterTailBufferLength, int, 0, "UnfilterTailBufferLength")
DefineSSDParameter(m_enableWAL, bool, false, "EnableWAL")
DefineSSDParameter(m_disableCheckpoint, bool, false, "DisableCheckpoint")
DefineSSDParameter(m_headRoleFile, std::string, std::string("head_role.bin"), "HeadRoleFile")
DefineSSDParameter(m_numTagsPerVec, int, 0, "NumTagsPerVec")
// 0 scans every STM1 tag column; a positive value restricts flat ACL matching
// to the categorical prefix, excluding trailing numeric attributes.
DefineSSDParameter(m_staticACLTagCols, int, 0, "StaticACLTagCols")
// Build the bundle cross-edge sidecar before STATIC tail construction. This is
// separate from the mutable runtime DisableCrossEdges diagnostic switch below.
DefineSSDParameter(m_buildCrossEdges, bool, false, "CrossEdges")
DefineSSDParameter(m_crossExtraEdges, int, 10, "CrossExtraEdges")

// GPU Building
DefineSSDParameter(m_gpuSSDNumTrees, int, 100, "GPUSSDNumTrees")
DefineSSDParameter(m_gpuSSDLeafSize, int, 200, "GPUSSDLeafSize")
DefineSSDParameter(m_numGPUs, int, 1, "NumGPUs")

// Searching
DefineSSDParameter(m_searchResult, std::string, std::string(""), "SearchResult")
DefineSSDParameter(m_logFile, std::string, std::string(""), "LogFile")
DefineSSDParameter(m_qpsLimit, int, 0, "QpsLimit")
DefineSSDParameter(m_resultNum, int, 5, "ResultNum") // Mutable
DefineSSDParameter(m_truthResultNum, int, -1, "TruthResultNum") // Mutable
DefineSSDParameter(m_maxCheck, int, 4096, "MaxCheck") // Mutable
DefineSSDParameter(m_hashExp, int, 4, "HashTableExponent")
DefineSSDParameter(m_queryCountLimit, int, (std::numeric_limits<int>::max)(), "QueryCountLimit")
DefineSSDParameter(m_maxDistRatio, float, 10000, "MaxDistRatio")
DefineSSDParameter(m_ioThreads, int, 4, "IOThreadsPerHandler") // Mutable
DefineSSDParameter(m_searchInternalResultNum, int, 64, "SearchInternalResultNum") // Mutable; [SearchSSDIndex] InternalResultNum aliases here
DefineSSDParameter(m_searchPostingPageLimit, int, 3, "SearchPostingPageLimit") // Mutable; STATIC reads are sized from posting metadata
DefineSSDParameter(m_collectPostingContributionStats, bool, false, "CollectPostingContributionStats") // Mutable; diagnostic only
DefineSSDParameter(m_forceDenseTagSearch, bool, false, "ForceDenseTagSearch") // Mutable
DefineSSDParameter(m_directSparseMaxPostings, int, 320, "DirectSparseMaxPostings") // Sparse-tag sidecar threshold
DefineSSDParameter(m_filteredSearchNprobeSafety, float, 1.0f, "FilteredSearchNprobeSafety") // Mutable
DefineSSDParameter(m_filteredSearchTargetRecall, float, 1.0f, "FilteredSearchTargetRecall") // Mutable
DefineSSDParameter(m_filteredSearchCoverageExponent, float, 0.0f, "FilteredSearchCoverageExponent") // Mutable; 0 disables coverage-driven over-probing
DefineSSDParameter(m_enableAdaptiveFilteredNprobe, bool, false, "EnableAdaptiveFilteredNprobe") // Mutable; opt-in override of SearchInternalResultNum
DefineSSDParameter(m_logAdaptiveNprobe, bool, false, "LogAdaptiveNprobe") // Mutable; per-query observability
DefineSSDParameter(m_logPhaseTime, bool, false, "LogPhaseTime") // Mutable; diagnostic timing only
DefineSSDParameter(m_unifiedNprobeBudget, bool, true, "UnifiedNprobeBudget") // Mutable
DefineSSDParameter(m_multiNodeBudgetKeepRatio, double, 0.60, "MultiNodeBudgetKeepRatio") // Mutable
DefineSSDParameter(m_disableCrossEdges, bool, false, "DisableCrossEdges") // Mutable
DefineSSDParameter(m_filterKeepCross, bool, false, "FilterKeepCross") // Mutable
DefineSSDParameter(m_disableCrossSubgraph, bool, false, "DisableCrossSubgraph") // Mutable
DefineSSDParameter(m_logUExtra, bool, false, "LogUExtra") // Mutable
DefineSSDParameter(m_logCrossStats, bool, false, "LogCrossStats") // Mutable
DefineSSDParameter(m_logPathStats, bool, false, "LogPathStats") // Mutable
DefineSSDParameter(m_dumpHeads, int, 0, "DumpHeads") // Mutable; number of queries to dump
DefineSSDParameter(m_filterKeepUExtra, bool, false, "FilterKeepUExtra") // Mutable
DefineSSDParameter(m_enableUnfilterTail, bool, true, "EnableUnfilterTail") // Mutable
DefineSSDParameter(m_ablateUExtra, bool, false, "AblateUExtra") // Mutable
DefineSSDParameter(m_ablateTail, bool, false, "AblateTail") // Mutable
DefineSSDParameter(m_unfilterPurePages, bool, false, "UnfilterPurePages") // Mutable
DefineSSDParameter(m_unfilterExtraTailPages, int, 0, "UnfilterExtraTailPages") // Mutable
// STATIC distance-order diagnostic: scan the nearest pure prefix while retaining
// the complete tail suffix. 100 preserves normal full-posting behavior.
DefineSSDParameter(m_unfilterPureDistanceScanPercent, int, 100, "UnfilterPureDistanceScanPercent") // Mutable
DefineSSDParameter(m_enableHierPostingFilter, bool, false, "EnableHierPostingFilter") // Mutable
DefineSSDParameter(m_rerank, int, 0, "Rerank")
DefineSSDParameter(m_enableADC, bool, false, "EnableADC")
DefineSSDParameter(m_recall_analysis, bool, false, "RecallAnalysis")
DefineSSDParameter(m_debugBuildInternalResultNum, int, 64, "DebugBuildInternalResultNum")
DefineSSDParameter(m_iotimeout, int, 30, "IOTimeout") // Mutable

// Calculating
// TruthFilePrefix
DefineSSDParameter(m_truthFilePrefix, std::string, std::string(""), "TruthFilePrefix")
// CalTruth
DefineSSDParameter(m_calTruth, bool, true, "CalTruth") // Mutable
DefineSSDParameter(m_onlySearchFinalBatch, bool, false, "OnlySearchFinalBatch")
// Search multiple times for stable result
DefineSSDParameter(m_searchTimes, int, 1, "SearchTimes")
// Frontend search threadnum
DefineSSDParameter(m_searchThreadNum, int, 2, "SearchThreadNum") // Mutable
// Show tradeoff of latency and acurracy
DefineSSDParameter(m_minInternalResultNum, int, -1, "MinInternalResultNum")
DefineSSDParameter(m_stepInternalResultNum, int, -1, "StepInternalResultNum")
DefineSSDParameter(m_maxInternalResultNum, int, -1, "MaxInternalResultNum")

// Updating(SPFresh Update Test)
// For update mode: current only update
DefineSSDParameter(m_update, bool, false, "Update")
// For Test Mode
DefineSSDParameter(m_inPlace, bool, true, "InPlace")
DefineSSDParameter(m_outOfPlace, bool, false, "OutOfPlace")
// latency limit
DefineSSDParameter(m_latencyLimit, float, 10.0, "LatencyLimit") // Mutable
// Update batch size
DefineSSDParameter(m_step, int, 0, "Step")
// Frontend update threadnum
DefineSSDParameter(m_insertThreadNum, int, 1, "InsertThreadNum") // Mutable
// Update limit
DefineSSDParameter(m_endVectorNum, int, -1, "EndVectorNum")
// Persistent buffer path
DefineSSDParameter(m_persistentBufferPath, std::string, std::string(""), "PersistentBufferPath")
// Background append threadnum
DefineSSDParameter(m_appendThreadNum, int, 1, "AppendThreadNum") // Mutable
// Background reassign threadnum
DefineSSDParameter(m_reassignThreadNum, int, 0, "ReassignThreadNum") // Mutable
// Background process batch size
DefineSSDParameter(m_batch, int, 1000, "Batch")
// Total Vector Path
DefineSSDParameter(m_fullVectorPath, std::string, std::string(""), "FullVectorPath")
// Appended full-precision vectors for dynamic in-posting PQ updates
DefineSSDParameter(m_updateVectorFile, std::string, std::string("update_vectors.bin"), "UpdateVectorFile")
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
DefineSSDParameter(m_postingOffset, int, 0, "PostingOffset")
DefineSSDParameter(m_disableReassign, bool, false, "DisableReassign")
DefineSSDParameter(m_searchDuringUpdate, bool, false, "SearchDuringUpdate")
DefineSSDParameter(m_reassignK, int, 0, "ReassignK")
DefineSSDParameter(m_recovery, bool, false, "Recovery")
DefineSSDParameter(m_maxFileSize, int, 300, "MaxFileSizeGB")
DefineSSDParameter(m_startFileSize, int, 10, "StartFileSizeGB")
DefineSSDParameter(m_growthFileSize, int, 10, "GrowthFileSizeGB")
DefineSSDParameter(m_growThreshold, float, 0.05, "GrowthThreshold")
DefineSSDParameter(m_fDeletePercentageForRefine, float, 0.4F, "DeletePercentageForRefine") // Mutable
DefineSSDParameter(m_oneClusterCutMax, bool, false, "OneClusterCutMax") // Mutable
DefineSSDParameter(m_asyncMergeInSearch, bool, true, "AsyncMergeInSearch") // Mutable
DefineSSDParameter(m_consistencyCheck, bool, false, "ConsistencyCheck") // Mutable
DefineSSDParameter(m_checksumCheck, bool, false, "ChecksumCheck") // Mutable
DefineSSDParameter(m_checksumInRead, bool, false, "ChecksumInRead") // Mutable
DefineSSDParameter(m_cacheSize, int, 0, "CacheSizeGB") // Mutable
DefineSSDParameter(m_cacheShards, int, 1, "CacheShards") // Mutable
DefineSSDParameter(m_asyncAppendQueueSize, int, 0, "AsyncAppendQueueSize") // Mutable
DefineSSDParameter(m_allowZeroReplica, bool, false, "AllowZeroReplica")
DefineSSDParameter(m_centeringToZero, bool, false, "CenteringToZero")
    
// Iterative
DefineSSDParameter(m_headBatch, int, 32, "IterativeSearchHeadBatch") // Mutable

DefineSSDParameter(m_shareDB, bool, false, "ShareDB")

// Primary-head CSR bypass. The build emits one exact primary owner per vector;
// project-filtered searches can expand graph heads from RAM without posting IO.
DefineSSDParameter(m_buildPrimaryHeadCSR, bool, false, "BuildPrimaryHeadCSR")
DefineSSDParameter(m_primaryHeadCSRFile, std::string, std::string("primary_head_csr.bin"), "PrimaryHeadCSRFile")
DefineSSDParameter(m_enablePrimaryHeadBypass, bool, false, "EnablePrimaryHeadBypass")
DefineSSDParameter(m_primaryHeadBypassRerankL, int, 0, "PrimaryHeadBypassRerankL")

// In-posting quantization (unified): postings store a compact code [meta|code] in
// place of the full ValueType vector; the full vectors stay on disk for cold rerank.
// PostingQuantizer selects the in-posting codec; the head index is independently
// kept full-precision unless QuantizeHead=true.
DefineSSDParameter(m_postingQuantizer, std::string, std::string("None"), "PostingQuantizer") // None|RaBitQ|OPQ|PipePQ
DefineSSDParameter(m_postingQuantBits, int, 2, "PostingQuantBits")          // RaBitQ bits per dim (1 or 2)
DefineSSDParameter(m_postingQuantM, int, 0, "PostingQuantM")                // OPQ/PipePQ code bytes per vector (subvector/chunk count)
DefineSSDParameter(m_requantizeFromPipePQ, bool, false, "RequantizeFromPipePQ") // one-time same-stride PipePQ->OPQ posting rewrite
DefineSSDParameter(m_quantizeHead, bool, false, "QuantizeHead")             // build the head index on quantized vectors
DefineSSDParameter(m_postingQuantFile, std::string, std::string(""), "PostingQuantizerFile") // code sidecar (abs or rel to index dir)
DefineSSDParameter(m_pipePQPivotsFile, std::string, std::string(""), "PipePQPivotsFile")     // PipeANN pivot sidecar (abs or rel to index dir)
DefineSSDParameter(m_fullVectorFile, std::string, std::string(""), "FullVectorFile")         // full-precision base for cold rerank
DefineSSDParameter(m_rerankL, int, 0, "RerankL")                            // exact-rerank depth over screened survivors (0 = default)
DefineSSDParameter(m_quantADCOnly, bool, false, "QuantADCOnly")             // skip full-vector rerank, return ADC/estimate order

#endif
