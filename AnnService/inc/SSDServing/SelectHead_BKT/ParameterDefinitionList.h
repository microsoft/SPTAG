// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef DefineSelectHeadParameter

// DefineSelectHeadParameter(VarName, VarType, DefaultValue, RepresentStr)
DefineSelectHeadParameter(m_vectorFile, std::string, std::string("vectors.bin"), "VectorFilePath")
DefineSelectHeadParameter(m_valueType, SPTAG::VectorValueType, SPTAG::VectorValueType::Float, "VectorValueType")
DefineSelectHeadParameter(m_iDistCalcMethod, SPTAG::DistCalcMethod, SPTAG::DistCalcMethod::L2, "DistCalcMethod")

DefineSelectHeadParameter(m_iTreeNumber, int, 1, "TreeNumber")
DefineSelectHeadParameter(m_iBKTKmeansK, int, 32, "BKTKmeansK")
DefineSelectHeadParameter(m_iBKTLeafSize, int, 8, "BKTLeafSize")
DefineSelectHeadParameter(m_iSamples, int, 1000, "SamplesNumber")
DefineSelectHeadParameter(m_saveBKT, bool, true, "SaveBKT")

DefineSelectHeadParameter(m_analyzeOnly, bool, true, "AnalyzeOnly")
DefineSelectHeadParameter(m_calcStd, bool, true, "CalcStd")
DefineSelectHeadParameter(m_selectDynamically, bool, true, "SelectDynamically")
DefineSelectHeadParameter(m_noOutput, bool, false, "NoOutput")

DefineSelectHeadParameter(m_selectThreshold, int, 6, "SelectThreshold")
DefineSelectHeadParameter(m_splitFactor, int, 5, "SplitFactor")
DefineSelectHeadParameter(m_splitThreshold, int, 25, "SplitThreshold")
DefineSelectHeadParameter(m_ratio, double, 0.2, "Ratio")
DefineSelectHeadParameter(m_recursiveCheckSmallCluster, bool, true, "RecursiveCheckSmallCluster")
DefineSelectHeadParameter(m_printSizeCount, bool, true, "PrintSizeCount")

DefineSelectHeadParameter(m_outputIDFile, std::string, std::string("HeadVectorIDs.bin"), "OutputIDFile")
DefineSelectHeadParameter(m_outputVectorFile, std::string, std::string("HeadVectors.bin"), "OutputVectorFile")


#endif
