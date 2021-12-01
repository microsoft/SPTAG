// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef DefineKDTParameter

// DefineKDTParameter(VarName, VarType, DefaultValue, RepresentStr)
DefineKDTParameter(m_sKDTFilename, std::string, std::string("tree.bin"), "TreeFilePath")
DefineKDTParameter(m_sGraphFilename, std::string, std::string("graph.bin"), "GraphFilePath")
DefineKDTParameter(m_sDataPointsFilename, std::string, std::string("vectors.bin"), "VectorFilePath")
DefineKDTParameter(m_sDeleteDataPointsFilename, std::string, std::string("deletes.bin"), "DeleteVectorFilePath")

DefineKDTParameter(m_pTrees.m_iTreeNumber, int, 1L, "KDTNumber")
DefineKDTParameter(m_pTrees.m_numTopDimensionKDTSplit, int, 5L, "NumTopDimensionKDTSplit")
DefineKDTParameter(m_pTrees.m_iSamples, int, 100L, "Samples")
DefineKDTParameter(m_pTrees.m_bOldVersion, bool, false, "IsOldVersion")

DefineKDTParameter(m_pGraph.m_iTPTNumber, int, 32L, "TPTNumber")
DefineKDTParameter(m_pGraph.m_iTPTLeafSize, int, 2000L, "TPTLeafSize")
DefineKDTParameter(m_pGraph.m_numTopDimensionTPTSplit, int, 5L, "NumTopDimensionTPTSplit")

DefineKDTParameter(m_pGraph.m_iNeighborhoodSize, DimensionType, 32L, "NeighborhoodSize")
DefineKDTParameter(m_pGraph.m_fNeighborhoodScale, float, 2.0F, "GraphNeighborhoodScale")
DefineKDTParameter(m_pGraph.m_fCEFScale, float, 2.0F, "GraphCEFScale")
DefineKDTParameter(m_pGraph.m_iRefineIter, int, 2L, "RefineIterations")
DefineKDTParameter(m_pGraph.m_rebuild, int, 0L, "EnableRebuild")
DefineKDTParameter(m_pGraph.m_iCEF, int, 1000L, "CEF")
DefineKDTParameter(m_pGraph.m_iAddCEF, int, 500L, "AddCEF")
DefineKDTParameter(m_pGraph.m_iMaxCheckForRefineGraph, int, 8192L, "MaxCheckForRefineGraph")
DefineKDTParameter(m_pGraph.m_fRNGFactor, float, 1.0f, "RNGFactor")

DefineKDTParameter(m_pGraph.m_iGPUGraphType, int, 2, "GPUGraphType") // Have GPU construct KNN or RNG
DefineKDTParameter(m_pGraph.m_iGPURefineSteps, int, 0, "GPURefineSteps") // Steps of GPU neighbor-refinement
DefineKDTParameter(m_pGraph.m_iGPURefineDepth, int, 30, "GPURefineDepth") // Depth of graph search for refinement
DefineKDTParameter(m_pGraph.m_iGPULeafSize, int, 500, "GPULeafSize")
DefineKDTParameter(m_pGraph.m_iheadNumGPUs, int, 1, "HeadNumGPUs")
DefineKDTParameter(m_pGraph.m_iTPTBalanceFactor, int, 2, "TPTBalanceFactor")

DefineKDTParameter(m_iNumberOfThreads, int, 1L, "NumberOfThreads")
DefineKDTParameter(m_iDistCalcMethod, SPTAG::DistCalcMethod, SPTAG::DistCalcMethod::Cosine, "DistCalcMethod")

DefineKDTParameter(m_fDeletePercentageForRefine, float, 0.4F, "DeletePercentageForRefine")
DefineKDTParameter(m_addCountForRebuild, int, 1000, "AddCountForRebuild")
DefineKDTParameter(m_iMaxCheck, int, 8192L, "MaxCheck")
DefineKDTParameter(m_iThresholdOfNumberOfContinuousNoBetterPropagation, int, 3L, "ThresholdOfNumberOfContinuousNoBetterPropagation")
DefineKDTParameter(m_iNumberOfInitialDynamicPivots, int, 50L, "NumberOfInitialDynamicPivots")
DefineKDTParameter(m_iNumberOfOtherDynamicPivots, int, 4L, "NumberOfOtherDynamicPivots")
DefineKDTParameter(m_iHashTableExp, int, 2L, "HashTableExponent")
DefineKDTParameter(m_iDataBlockSize, int, 1024 * 1024, "DataBlockSize")
DefineKDTParameter(m_iDataCapacity, int, MaxSize, "DataCapacity")
DefineKDTParameter(m_iMetaRecordSize, int, 10, "MetaRecordSize")

#endif
