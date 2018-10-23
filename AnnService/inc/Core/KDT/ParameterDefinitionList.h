#ifdef DefineParameter

// DefineKDTParameter(VarName, VarType, DefaultValue, RepresentStr)
DefineKDTParameter(m_sKDTFilename, std::string, std::string("tree.bin"), "TreeFilePath")
DefineKDTParameter(m_sGraphFilename, std::string, std::string("graph.bin"), "GraphFilePath")
DefineKDTParameter(m_sDataPointsFilename, std::string, std::string("vectors.bin"), "VectorFilePath")

DefineKDTParameter(m_iKDTNumber, int, 1L, "KDTNumber")
DefineKDTParameter(m_numTopDimensionKDTSplit, int, 5L, "NumTopDimensionKDTSplit")
DefineKDTParameter(m_numSamplesKDTSplitConsideration, int, 100L, "NumSamplesKDTSplitConsideration")
DefineKDTParameter(m_iNeighborhoodSize, int, 32L, "NeighborhoodSize")
DefineKDTParameter(m_iTPTNumber, int, 32L, "TPTNumber")
DefineKDTParameter(m_iTPTLeafSize, int, 2000L, "TPTLeafSize")
DefineKDTParameter(m_numTopDimensionTPTSplit, int, 5L, "NumTopDimensionTPTSplit")
DefineKDTParameter(m_numSamplesTPTSplitConsideration, int, 100L, "NumSamplesTPTSplitConsideration")
DefineKDTParameter(m_iCEF, int, 1000L, "CEF")
DefineKDTParameter(m_iMaxCheckForRefineGraph, int, 10000L, "MaxCheckForRefineGraph")
DefineKDTParameter(m_iMaxCheck, int, 8192L, "MaxCheck")
DefineKDTParameter(m_iNumberOfThreads, int, 1L, "NumberOfThreads")

DefineKDTParameter(g_iThresholdOfNumberOfContinuousNoBetterPropagation, int, 3L, "ThresholdOfNumberOfContinuousNoBetterPropagation")
DefineKDTParameter(g_iNumberOfInitialDynamicPivots, int, 50L, "NumberOfInitialDynamicPivots")
DefineKDTParameter(g_iNumberOfOtherDynamicPivots, int, 4L, "NumberOfOtherDynamicPivots")

DefineKDTParameter(m_iDistCalcMethod, SPTAG::DistCalcMethod, SPTAG::DistCalcMethod::Cosine, "DistCalcMethod")
DefineKDTParameter(m_iRefineIter, int, 0L, "RefineIterations")
DefineKDTParameter(m_iDebugLoad, int, -1, "NumTrains")
DefineKDTParameter(m_iCacheSize, int, -1, "CacheSize")
#endif
