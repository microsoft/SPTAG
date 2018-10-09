#ifdef DefineBKTParameter

// DefineBKTParameter(VarName, VarType, DefaultValue, RepresentStr)
DefineBKTParameter(m_sBKTFilename, std::string, std::string("tree.bin"), "TreeFilePath")
DefineBKTParameter(m_sGraphFilename, std::string, std::string("graph.bin"), "GraphFilePath")
DefineBKTParameter(m_sDataPointsFilename, std::string, std::string("vectors.bin"), "VectorFilePath")

DefineBKTParameter(m_iBKTNumber, int, 1L, "BKTNumber")
DefineBKTParameter(m_iBKTKmeansK, int, 32L, "BKTKmeansK")
DefineBKTParameter(m_iNeighborhoodSize, int, 32L, "NeighborhoodSize")
DefineBKTParameter(m_iBKTLeafSize, int, 8L, "BKTLeafSize")
DefineBKTParameter(m_iSamples, int, 1000L, "Samples")
DefineBKTParameter(m_iTptreeNumber, int, 32L, "TpTreeNumber")
DefineBKTParameter(m_iTPTLeafSize, int, 2000L, "TPTLeafSize")
DefineBKTParameter(m_numTopDimensionTpTreeSplit, int, 5L, "NumTopDimensionTpTreeSplit")
DefineBKTParameter(m_iCEF, int, 1000L, "CEF")
DefineBKTParameter(m_iMaxCheckForRefineGraph, int, 10000L, "MaxCheckForRefineGraph")
DefineBKTParameter(m_iMaxCheck, int, 8192L, "MaxCheck")
DefineBKTParameter(m_iNumberOfThreads, int, 1L, "NumberOfThreads")

DefineBKTParameter(g_iThresholdOfNumberOfContinuousNoBetterPropagation, int, 3L, "ThresholdOfNumberOfContinuousNoBetterPropagation")
DefineBKTParameter(g_iNumberOfInitialDynamicPivots, int, 50L, "NumberOfInitialDynamicPivots")
DefineBKTParameter(g_iNumberOfOtherDynamicPivots, int, 4L, "NumberOfOtherDynamicPivots")

DefineBKTParameter(m_iDistCalcMethod, SPTAG::DistCalcMethod, SPTAG::DistCalcMethod::Cosine, "DistCalcMethod")
DefineBKTParameter(m_iRefineIter, int, 0L, "RefineIterations")
DefineBKTParameter(m_iDebugLoad, int, -1, "NumTrains")
DefineBKTParameter(m_iCacheSize, int, -1, "CacheSize")
#endif
