// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef DefineSSDParameter

// Both Building and Searching
DefineSSDParameter(m_execute, bool, false, "isExecute")
DefineSSDParameter(m_buildSsdIndex, bool, false, "BuildSsdIndex")
DefineSSDParameter(m_internalResultNum, int, 64, "InternalResultNum")
DefineSSDParameter(m_iNumberOfThreads, int, 16, "NumberOfThreads")
DefineSSDParameter(m_headConfig, std::string, std::string(""), "HeadConfig") // must be in "Index" section

// Building
DefineSSDParameter(m_postingPageLimit, int, 3, "PostingPageLimit")
DefineSSDParameter(m_replicaCount, int, 8, "ReplicaCount")
DefineSSDParameter(m_outputEmptyReplicaID, bool, false, "OutputEmptyReplicaID")
DefineSSDParameter(m_batches, int, 1, "Batches")
DefineSSDParameter(m_tmpdir, std::string, std::string("."), "TmpDir")
DefineSSDParameter(m_rngFactor, float, 1.0f, "RNGFactor")
DefineSSDParameter(m_samples, int, 100, "RecallTestSampleNumber")

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
DefineSSDParameter(m_searchPostingPageLimit, int, (std::numeric_limits<int>::max)(), "SearchPostingPageLimit")
DefineSSDParameter(m_rerank, int, 0, "Rerank")
DefineSSDParameter(m_enableADC, bool, false, "EnableADC")
DefineSSDParameter(m_rerankFilePath, std::string, std::string(), "RerankFilePath")

#endif
