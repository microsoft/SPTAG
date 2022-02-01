// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef DefineBKTParameter

// DefineBKTParameter(VarName, VarType, DefaultValue, RepresentStr)
DefineBKTParameter(m_sBKTFilename, std::string, std::string("tree.bin"), "TreeFilePath")
DefineBKTParameter(m_sGraphFilename, std::string, std::string("graph.bin"), "GraphFilePath")
DefineBKTParameter(m_sDataPointsFilename, std::string, std::string("vectors.bin"), "VectorFilePath")
DefineBKTParameter(m_sDeleteDataPointsFilename, std::string, std::string("deletes.bin"), "DeleteVectorFilePath")

DefineBKTParameter(m_pTrees.m_bfs, int, 0L, "EnableBfs")
DefineBKTParameter(m_pTrees.m_iTreeNumber, int, 1L, "BKTNumber")
DefineBKTParameter(m_pTrees.m_iBKTKmeansK, int, 32L, "BKTKmeansK")
DefineBKTParameter(m_pTrees.m_iBKTLeafSize, int, 8L, "BKTLeafSize")
DefineBKTParameter(m_pTrees.m_iSamples, int, 1000L, "Samples")
DefineBKTParameter(m_pTrees.m_fBalanceFactor, float, 100.0F, "BKTLambdaFactor")

DefineBKTParameter(m_pGraph.m_iTPTNumber, int, 32L, "TPTNumber")
DefineBKTParameter(m_pGraph.m_iTPTLeafSize, int, 2000L, "TPTLeafSize")
DefineBKTParameter(m_pGraph.m_numTopDimensionTPTSplit, int, 5L, "NumTopDimensionTpTreeSplit")

DefineBKTParameter(m_pGraph.m_iNeighborhoodSize, DimensionType, 32L, "NeighborhoodSize")
DefineBKTParameter(m_pGraph.m_fNeighborhoodScale, float, 2.0F, "GraphNeighborhoodScale")
DefineBKTParameter(m_pGraph.m_fCEFScale, float, 2.0F, "GraphCEFScale")
DefineBKTParameter(m_pGraph.m_iRefineIter, int, 2L, "RefineIterations")
DefineBKTParameter(m_pGraph.m_rebuild, int, 0L, "EnableRebuild")
DefineBKTParameter(m_pGraph.m_iCEF, int, 1000L, "CEF")
DefineBKTParameter(m_pGraph.m_iAddCEF, int, 500L, "AddCEF")
DefineBKTParameter(m_pGraph.m_iMaxCheckForRefineGraph, int, 8192L, "MaxCheckForRefineGraph")
DefineBKTParameter(m_pGraph.m_fRNGFactor, float, 1.0f, "RNGFactor")

DefineBKTParameter(m_pGraph.m_iGPUGraphType, int, 2, "GPUGraphType") // Have GPU construct KNN,loose RNG or RNG
DefineBKTParameter(m_pGraph.m_iGPURefineSteps, int, 0, "GPURefineSteps") // Steps of GPU neighbor-refinement
DefineBKTParameter(m_pGraph.m_iGPURefineDepth, int, 30, "GPURefineDepth") // Depth of graph search for refinement
DefineBKTParameter(m_pGraph.m_iGPULeafSize, int, 500, "GPULeafSize")
DefineBKTParameter(m_pGraph.m_iheadNumGPUs, int, 1, "HeadNumGPUs")
DefineBKTParameter(m_pGraph.m_iTPTBalanceFactor, int, 2, "TPTBalanceFactor")

DefineBKTParameter(m_iNumberOfThreads, int, 1L, "NumberOfThreads")
DefineBKTParameter(m_iDistCalcMethod, SPTAG::DistCalcMethod, SPTAG::DistCalcMethod::Cosine, "DistCalcMethod")

DefineBKTParameter(m_fDeletePercentageForRefine, float, 0.4F, "DeletePercentageForRefine")
DefineBKTParameter(m_addCountForRebuild, int, 1000, "AddCountForRebuild")
DefineBKTParameter(m_iMaxCheck, int, 8192L, "MaxCheck")
DefineBKTParameter(m_iThresholdOfNumberOfContinuousNoBetterPropagation, int, 3L, "ThresholdOfNumberOfContinuousNoBetterPropagation")
DefineBKTParameter(m_iNumberOfInitialDynamicPivots, int, 50L, "NumberOfInitialDynamicPivots")
DefineBKTParameter(m_iNumberOfOtherDynamicPivots, int, 4L, "NumberOfOtherDynamicPivots")
DefineBKTParameter(m_iHashTableExp, int, 2L, "HashTableExponent")
DefineBKTParameter(m_iDataBlockSize, int, 1024 * 1024, "DataBlockSize")
DefineBKTParameter(m_iDataCapacity, int, MaxSize, "DataCapacity")
DefineBKTParameter(m_iMetaRecordSize, int, 10, "MetaRecordSize")

#endif
