// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef DefineBuildHeadParameter

// DefineBuildHeadParameter(VarName, VarType, DefaultValue, RepresentStr)
DefineBuildHeadParameter(m_inputFiles, std::string, std::string("vectors.bin"), "VectorFilePath")
DefineBuildHeadParameter(m_outputFolder, std::string, std::string("HeadVectors.bin"), "VectorValueType")
DefineBuildHeadParameter(m_indexAlgoType, SPTAG::IndexAlgoType, SPTAG::IndexAlgoType::BKT, "DistCalcMethod")
DefineBuildHeadParameter(m_builderConfigFile, std::string, std::string("builder.ini"), "VectorFilePath")
DefineBuildHeadParameter(m_inputValueType, SPTAG::VectorValueType, SPTAG::VectorValueType::Float, "VectorValueType")
DefineBuildHeadParameter(m_threadNum, std::uint32_t, omp_get_num_threads(), "DistCalcMethod")

#endif
