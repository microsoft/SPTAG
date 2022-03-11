// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef DefineDiskANNParameter

// DefineDiskANNParameter(VarName, VarType, DefaultValue, RepresentStr)
DefineDiskANNParameter(m_sGraphFilename, std::string, std::string("graph.bin"), "GraphFilePath")
DefineDiskANNParameter(m_sDataPointsFilename, std::string, std::string("vectors.bin"), "VectorFilePath")

DefineDiskANNParameter(m_iNumberOfThreads, unsigned, 1, "NumberOfThreads")
DefineDiskANNParameter(m_distCalcMethod, SPTAG::DistCalcMethod, SPTAG::DistCalcMethod::Cosine, "DistCalcMethod")

DefineDiskANNParameter(R, unsigned, 1, "R")
DefineDiskANNParameter(L, unsigned, 1, "L")
DefineDiskANNParameter(C, unsigned, 750, "C")
DefineDiskANNParameter(alpha, float, 1.0f, "alpha")
DefineDiskANNParameter(saturate_graph, bool, false, "saturate_graph")

#endif
