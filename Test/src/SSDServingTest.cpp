// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"
#include <string>
#include <fstream>
#include <random>
#include <type_traits>
#include <functional>
#include <algorithm>

#include <boost/filesystem.hpp>

#include "inc/Core/Common.h"
#include "inc/Helper/StringConvert.h"
#include "inc/SSDServing/main.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/CommonUtils.h"

template<typename T>
void GenerateVectors(std::string fileName, SPTAG::SizeType rows, SPTAG::DimensionType dims, SPTAG::VectorFileType fileType) {
	if (boost::filesystem::exists(fileName))
	{
		fprintf(stdout, "%s was generated. Skip generation.\n", fileName.c_str());
		return;
	}

	std::ofstream of(fileName, std::ofstream::binary);
	if (!of.is_open())
	{
		fprintf(stderr, "%s can't be opened.\n", fileName.c_str());
		BOOST_CHECK(false);
		return;
	}

	std::uniform_real_distribution<float> ud(0, 126);
	std::mt19937 mt(543);
	std::vector<T> tmp(dims);

	if (fileType == SPTAG::VectorFileType::DEFAULT)
	{
		of.write(reinterpret_cast<char*>(&rows), 4);
		of.write(reinterpret_cast<char*>(&dims), 4);

		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < dims; j++)
			{
				float smt = ud(mt);
				tmp[j] = static_cast<T>(smt);
			}

			SPTAG::COMMON::Utils::Normalize(tmp.data(), dims, SPTAG::COMMON::Utils::GetBase<T>());
			of.write(reinterpret_cast<char*>(tmp.data()), dims * sizeof(T));
		}
	}
	else if (fileType == SPTAG::VectorFileType::XVEC)
	{
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < dims; j++)
			{
				float smt = ud(mt);
				tmp[j] = static_cast<T>(smt);
			}

			SPTAG::COMMON::Utils::Normalize(tmp.data(), dims, SPTAG::COMMON::Utils::GetBase<T>());
			of.write(reinterpret_cast<char*>(&dims), 4);
			of.write(reinterpret_cast<char*>(tmp.data()), dims * sizeof(T));
		}

	}

}

void GenVec(std::string vectorsName, SPTAG::VectorValueType vecType, SPTAG::VectorFileType vecFileType, SPTAG::SizeType rows = 1000, SPTAG::DimensionType dims = 100) {
	switch (vecType)
	{
#define DefineVectorValueType(Name, Type) \
case SPTAG::VectorValueType::Name: \
GenerateVectors<Type>(vectorsName, rows, dims, vecFileType); \
break; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
	default:
		break;
	}
}

std::string CreateBaseConfig(SPTAG::VectorValueType p_valueType, SPTAG::DistCalcMethod p_distCalcMethod, 
	SPTAG::IndexAlgoType p_indexAlgoType, SPTAG::DimensionType p_dim,
	std::string p_vectorPath, SPTAG::VectorFileType p_vectorType, SPTAG::SizeType p_vectorSize, std::string p_vectorDelimiter,
	std::string p_queryPath, SPTAG::VectorFileType p_queryType, SPTAG::SizeType p_querySize, std::string p_queryDelimiter,
	std::string p_warmupPath, SPTAG::VectorFileType p_warmupType, SPTAG::SizeType p_warmupSize, std::string p_warmupDelimiter,
	std::string p_truthPath, SPTAG::TruthFileType p_truthType,
	bool p_generateTruth,
	std::string p_headIDFile,
	std::string p_headVectorsFile,
	std::string p_headIndexFolder,
	std::string p_ssdIndex
) {
	std::ostringstream config;
	config << "[Base]" << std::endl;
	config << "ValueType=" << SPTAG::Helper::Convert::ConvertToString(p_valueType) << std::endl;
	config << "DistCalcMethod=" << SPTAG::Helper::Convert::ConvertToString(p_distCalcMethod) << std::endl;
	config << "IndexAlgoType=" << SPTAG::Helper::Convert::ConvertToString(p_indexAlgoType) << std::endl;
	config << "Dim=" << p_dim << std::endl;
	config << "VectorPath=" << p_vectorPath << std::endl;
	config << "VectorType=" << SPTAG::Helper::Convert::ConvertToString(p_vectorType) << std::endl;
	config << "VectorSize=" << p_vectorSize << std::endl;
	config << "VectorDelimiter=" << p_vectorDelimiter << std::endl;
	config << "QueryPath=" << p_queryPath << std::endl;
	config << "QueryType=" << SPTAG::Helper::Convert::ConvertToString(p_queryType) << std::endl;
	config << "QuerySize=" << p_querySize << std::endl;
	config << "QueryDelimiter=" << p_queryDelimiter << std::endl;
	config << "WarmupPath=" << p_warmupPath << std::endl;
	config << "WarmupType=" << SPTAG::Helper::Convert::ConvertToString(p_warmupType) << std::endl;
	config << "WarmupSize=" << p_warmupSize << std::endl;
	config << "WarmupDelimiter=" << p_warmupDelimiter << std::endl;
	config << "TruthPath=" << p_truthPath << std::endl;
	config << "TruthType=" << SPTAG::Helper::Convert::ConvertToString(p_truthType) << std::endl;
	config << "GenerateTruth=" << SPTAG::Helper::Convert::ConvertToString(p_generateTruth) << std::endl;
	config << "IndexDirectory=" << "zbtest" << std::endl;
	config << "HeadVectorIDs=" << p_headIDFile << std::endl;
	config << "HeadVectors=" << p_headVectorsFile << std::endl;
	config << "HeadIndexFolder=" << p_headIndexFolder << std::endl;
	config << "SSDIndex=" << p_ssdIndex << std::endl;
	config << std::endl;
	return config.str();
}

void TestHead(std::string configName, std::string OutputIDFile, std::string OutputVectorFile, std::string baseConfig) {

	std::ofstream config(configName);
	if (!config.is_open())
	{
		fprintf(stderr, "%s can't be opened.\n", configName.c_str());
		BOOST_CHECK(false);
		return;
	}

	config << baseConfig;

	config << "[SelectHead]" << std::endl;
	config << "isExecute=true" << std::endl;
	config << "TreeNumber=" << "1" << std::endl;
	config << "BKTKmeansK=" << 3 << std::endl;
	config << "BKTLeafSize=" << 6 << std::endl;
	config << "SamplesNumber=" << 100 << std::endl;
	config << "NumberOfThreads=" << "2" << std::endl;
	config << "SaveBKT=" << "false" << std::endl;

	config << "AnalyzeOnly=" << "false" << std::endl;
	config << "CalcStd=" << "true" << std::endl;
	config << "SelectDynamically=" << "true" << std::endl;
	config << "NoOutput=" << "false" << std::endl;

	config << "SelectThreshold=" << 12 << std::endl;
	config << "SplitFactor=" << 9 << std::endl;
	config << "SplitThreshold=" << 18 << std::endl;
	config << "Ratio=" << "0.2" << std::endl;
	config << "RecursiveCheckSmallCluster=" << "true" << std::endl;
	config << "PrintSizeCount=" << "true" << std::endl;

	config.close();
	std::map<std::string, std::map<std::string, std::string>> my_map;
	SPTAG::SSDServing::BootProgram(false, &my_map, configName.c_str());
}

void TestBuildHead(
	std::string configName,
	std::string p_headVectorFile,
	std::string p_headIndexFile,
	SPTAG::IndexAlgoType p_indexAlgoType,
	std::string p_builderFile,
	std::string baseConfig) {

	std::ofstream config(configName);
	if (!config.is_open())
	{
		fprintf(stderr, "%s can't be opened.\n", configName.c_str());
		BOOST_CHECK(false);
		return;
	}

	{
		if (p_builderFile.empty())
		{
			fprintf(stderr, "no builder file for head index build.\n");
			BOOST_CHECK(false);
			return;
		}

		if (boost::filesystem::exists(p_builderFile))
		{
			fprintf(stdout, "%s was generated. Skip generation.\n", p_builderFile.c_str());
		}
		else {
			std::ofstream bf(p_builderFile);
			if (!bf.is_open())
			{
				fprintf(stderr, "%s can't be opened.\n", p_builderFile.c_str());
				BOOST_CHECK(false);
				return;
			}
			bf.close();
		}
	}

	config << baseConfig;

	config << "[BuildHead]" << std::endl;
	config << "isExecute=true" << std::endl;
	config << "NumberOfThreads=" << 2 << std::endl;

	config.close();

	std::map<std::string, std::map<std::string, std::string>> my_map;
	SPTAG::SSDServing::BootProgram(false, &my_map, configName.c_str());
}

void TestBuildSSDIndex(std::string configName,
	std::string p_vectorIDTranslate,
	std::string p_headIndexFolder,
	std::string p_headConfig,
	std::string p_ssdIndex,
	bool p_outputEmptyReplicaID,
	std::string baseConfig
) {
	std::ofstream config(configName);
	if (!config.is_open())
	{
		fprintf(stderr, "%s can't be opened.\n", configName.c_str());
		BOOST_CHECK(false);
		return;
	}

	if (!p_headConfig.empty())
	{
		if (boost::filesystem::exists(p_headConfig))
		{
			fprintf(stdout, "%s was generated. Skip generation.\n", p_headConfig.c_str());
		}
		else {
			std::ofstream bf(p_headConfig);
			if (!bf.is_open())
			{
				fprintf(stderr, "%s can't be opened.\n", p_headConfig.c_str());
				BOOST_CHECK(false);
				return;
			}
			bf << "[Index]" << std::endl;
			bf.close();
		}
	}

	config << baseConfig;

	config << "[BuildSSDIndex]" << std::endl;
	config << "isExecute=true" << std::endl;
	config << "BuildSsdIndex=" << "true" << std::endl;
	config << "InternalResultNum=" << 60 << std::endl;
	config << "NumberOfThreads=" << 2 << std::endl;
	config << "HeadConfig=" << p_headConfig << std::endl;

	config << "ReplicaCount=" << 4 << std::endl;
	config << "PostingPageLimit=" << 2 << std::endl;
	config << "OutputEmptyReplicaID=" << p_outputEmptyReplicaID << std::endl;

	config.close();
	std::map<std::string, std::map<std::string, std::string>> my_map;
	SPTAG::SSDServing::BootProgram(false, &my_map, configName.c_str());
}

void TestSearchSSDIndex(
	std::string configName,
	std::string p_vectorIDTranslate,
	std::string p_headIndexFolder,
	std::string p_headConfig,
	std::string p_ssdIndex,
	std::string p_searchResult,
	std::string p_logFile,
	std::string baseConfig
) {
	std::ofstream config(configName);
	if (!config.is_open())
	{
		fprintf(stderr, "%s can't be opened.\n", configName.c_str());
		BOOST_CHECK(false);
		return;
	}

	if (!p_headConfig.empty())
	{
		if (boost::filesystem::exists(p_headConfig))
		{
			fprintf(stdout, "%s was generated. Skip generation.\n", p_headConfig.c_str());
		}
		else {
			std::ofstream bf(p_headConfig);
			if (!bf.is_open())
			{
				fprintf(stderr, "%s can't be opened.\n", p_headConfig.c_str());
				BOOST_CHECK(false);
				return;
			}
			bf << "[Index]" << std::endl;
			bf.close();
		}
	}

	config << baseConfig;

	config << "[SearchSSDIndex]" << std::endl;
	config << "isExecute=true" << std::endl;
	config << "BuildSsdIndex=" << "false" << std::endl;
	config << "InternalResultNum=" << 64 << std::endl;
	config << "NumberOfThreads=" << 2 << std::endl;
	config << "HeadConfig=" << p_headConfig << std::endl;

	config << "SearchResult=" << p_searchResult << std::endl;
	config << "LogFile=" << p_logFile << std::endl;
	config << "QpsLimit=" << 0 << std::endl;
	config << "ResultNum=" << 32 << std::endl;
	config << "MaxCheck=" << 2048 << std::endl;
	config << "SearchPostingPageLimit=" << 2 << std::endl;
	config << "QueryCountLimit=" << 10000 << std::endl;

	config.close();
	std::map<std::string, std::map<std::string, std::string>> my_map;
	SPTAG::SSDServing::BootProgram(false, &my_map, configName.c_str());
}

void RunFromMap() {
	std::map<std::string, std::map<std::string, std::string>> myMap;
	myMap["Base"]["IndexAlgoType"] = "KDT";
	std::string dataFilePath = "sddtest/vectors_Int8_DEFAULT.bin";
	std::string indexFilePath = "zbtest";
	SPTAG::SSDServing::BootProgram(
		true, 
		&myMap, 
		nullptr, 
		SPTAG::VectorValueType::Int8, 
		SPTAG::DistCalcMethod::L2, 
		dataFilePath.c_str(), 
		indexFilePath.c_str()
	);

	std::ostringstream config;
	config << "[Base]" << std::endl;
	config << "ValueType=" << "Int8" << std::endl;
	config << "DistCalcMethod=" << "L2" << std::endl;
	config << "IndexAlgoType=" << "KDT" << std::endl;
	config << "VectorPath=" << "sddtest/vectors_Int8_DEFAULT.bin"  << std::endl;
	config << "QueryPath=" << "sddtest/vectors_Int8_DEFAULT.query" << std::endl;
	config << "QueryType=" << "DEFAULT" << std::endl;
	config << "TruthPath=" << "sddtest/vectors_Int8_L2_DEFAULT_DEFAULT.truth" << std::endl;
	config << "TruthType=" << "DEFAULT" << std::endl;
	config << "IndexDirectory=" << "zbtest" << std::endl;
	config << "GenerateTruth=" << "true" << std::endl;
	config << std::endl;

	TestSearchSSDIndex(
		"run_from_map_search_config.ini",
		"",
		"",
		"",
		"",
		"",
		"",
		config.str()
		);
}

BOOST_AUTO_TEST_SUITE(SSDServingTest)

// #define RAW_VECTOR_NUM 1000
#define VECTOR_NUM 1000
#define QUERY_NUM 10
#define VECTOR_DIM 100

#define SSDTEST_DIRECTORY_NAME "sddtest"
#define SSDTEST_DIRECTORY SSDTEST_DIRECTORY_NAME "/"
// #define RAW_VECTORS(VT, FT)  "vectors_"#VT"_"#FT".bin"
#define VECTORS(VT, FT) SSDTEST_DIRECTORY "vectors_"#VT"_"#FT".bin"
#define QUERIES(VT, FT)  SSDTEST_DIRECTORY "vectors_"#VT"_"#FT".query"
#define TRUTHSET(VT, DM, FT, TFT) SSDTEST_DIRECTORY  "vectors_"#VT"_"#DM"_"#FT"_"#TFT".truth"
#define HEAD_IDS(VT, DM, FT)  "head_ids_"#VT"_"#DM"_"#FT".bin"
#define HEAD_VECTORS(VT, DM, FT)  "head_vectors_"#VT"_"#DM"_"#FT".bin"
#define HEAD_INDEX(VT, DM, ALGO, FT)  "head_"#VT"_"#DM"_"#ALGO"_"#FT".head_index"
#define SSD_INDEX(VT, DM, ALGO, FT)  "ssd_"#VT"_"#DM"_"#ALGO"_"#FT".ssd_index"

#define SELECT_HEAD_CONFIG(VT, DM, FT) SSDTEST_DIRECTORY "test_head_"#VT"_"#DM"_"#FT".ini"
#define BUILD_HEAD_CONFIG(VT, DM, ALGO) SSDTEST_DIRECTORY "test_build_head_"#VT"_"#DM"_"#ALGO".ini"
#define BUILD_HEAD_BUILDER_CONFIG(VT, DM, ALGO) SSDTEST_DIRECTORY "test_build_head_"#VT"_"#DM"_"#ALGO".builder.ini"
#define BUILD_SSD_CONFIG(VT, DM, ALGO) SSDTEST_DIRECTORY "test_build_ssd"#VT"_"#DM"_"#ALGO".ini"
#define BUILD_SSD_BUILDER_CONFIG(VT, DM, ALGO) SSDTEST_DIRECTORY "test_build_ssd"#VT"_"#DM"_"#ALGO".builder.ini"
#define SEARCH_SSD_CONFIG(VT, DM, ALGO) SSDTEST_DIRECTORY "test_search_ssd_"#VT"_"#DM"_"#ALGO".ini"
#define SEARCH_SSD_BUILDER_CONFIG(VT, DM, ALGO) SSDTEST_DIRECTORY "test_search_ssd_"#VT"_"#DM"_"#ALGO".builder.ini"
#define SEARCH_SSD_RESULT(VT, DM, ALGO, FT, TFT) SSDTEST_DIRECTORY "test_search_ssd_"#VT"_"#DM"_"#ALGO"_"#FT"_"#TFT".result"

#define GVQ(VT, FT) \
BOOST_AUTO_TEST_CASE(GenerateVectorsQueries##VT##FT) { \
boost::filesystem::create_directory(SSDTEST_DIRECTORY_NAME); \
GenVec(VECTORS(VT, FT), SPTAG::VectorValueType::VT, SPTAG::VectorFileType::FT, VECTOR_NUM, VECTOR_DIM); \
GenVec(QUERIES(VT, FT), SPTAG::VectorValueType::VT, SPTAG::VectorFileType::FT, QUERY_NUM, VECTOR_DIM); \
} \

GVQ(Float, DEFAULT)
GVQ(Int16, DEFAULT)
GVQ(UInt8, DEFAULT)
GVQ(Int8, DEFAULT)

GVQ(Float, XVEC)
GVQ(Int16, XVEC)
GVQ(UInt8, XVEC)
GVQ(Int8, XVEC)
#undef GVQ

#define WTEV(VT, DM, FT) \
BOOST_AUTO_TEST_CASE(TestHead##VT##DM##FT) { \
	std::string configName = SELECT_HEAD_CONFIG(VT, DM, FT); \
	std::string OutputIDFile = HEAD_IDS(VT, DM, FT); \
	std::string OutputVectorFile = HEAD_VECTORS(VT, DM, FT); \
	std::string base_config = CreateBaseConfig(SPTAG::VectorValueType::VT, SPTAG::DistCalcMethod::DM, SPTAG::IndexAlgoType::Undefined, VECTOR_DIM, \
		VECTORS(VT, FT), SPTAG::VectorFileType::FT, VECTOR_NUM, "", \
		"", SPTAG::VectorFileType::Undefined, -1, "", \
		"", SPTAG::VectorFileType::Undefined, -1, "", \
		"", SPTAG::TruthFileType::Undefined, \
		false, \
		OutputIDFile, \
		OutputVectorFile, \
		"", \
		"" \
		); \
	TestHead(configName, OutputIDFile, OutputVectorFile, base_config);  \
} \

WTEV(Float, L2, DEFAULT)
WTEV(Float, Cosine, DEFAULT)
WTEV(Int16, L2, DEFAULT)
WTEV(Int16, Cosine, DEFAULT)
WTEV(UInt8, L2, DEFAULT)
WTEV(UInt8, Cosine, DEFAULT)
WTEV(Int8, L2, DEFAULT)
WTEV(Int8, Cosine, DEFAULT)

WTEV(Float, L2, XVEC)
WTEV(Float, Cosine, XVEC)
WTEV(Int16, L2, XVEC)
WTEV(Int16, Cosine, XVEC)
WTEV(UInt8, L2, XVEC)
WTEV(UInt8, Cosine, XVEC)
WTEV(Int8, L2, XVEC)
WTEV(Int8, Cosine, XVEC)

#undef WTEV

#define BDHD(VT, DM, ALGO, FT) \
BOOST_AUTO_TEST_CASE(TestBuildHead##VT##DM##ALGO##FT) { \
	std::string configName = BUILD_HEAD_CONFIG(VT, DM, ALGO); \
	std::string builderFile = BUILD_HEAD_BUILDER_CONFIG(VT, DM, ALGO); \
	std::string base_config = CreateBaseConfig(SPTAG::VectorValueType::VT, SPTAG::DistCalcMethod::DM, SPTAG::IndexAlgoType::ALGO, VECTOR_DIM, \
		VECTORS(VT, FT), SPTAG::VectorFileType::FT, VECTOR_NUM, "", \
		"", SPTAG::VectorFileType::Undefined, -1, "", \
		"", SPTAG::VectorFileType::Undefined, -1, "", \
		"", SPTAG::TruthFileType::Undefined, \
		false, \
		"", \
		HEAD_VECTORS(VT, DM, FT), \
		HEAD_INDEX(VT, DM, ALGO, FT), \
		"" \
		); \
TestBuildHead( \
	configName, \
	HEAD_VECTORS(VT, DM, FT), \
	HEAD_INDEX(VT, DM, ALGO, FT), \
	SPTAG::IndexAlgoType::ALGO, \
	builderFile, \
	base_config \
); \
} \

BDHD(Float, L2, BKT, DEFAULT)
BDHD(Float, L2, KDT, DEFAULT)
BDHD(Float, Cosine, BKT, DEFAULT)
BDHD(Float, Cosine, KDT, DEFAULT)

BDHD(Int8, L2, BKT, DEFAULT)
BDHD(Int8, L2, KDT, DEFAULT)
BDHD(Int8, Cosine, BKT, DEFAULT)
BDHD(Int8, Cosine, KDT, DEFAULT)

BDHD(UInt8, L2, BKT, DEFAULT)
BDHD(UInt8, L2, KDT, DEFAULT)
BDHD(UInt8, Cosine, BKT, DEFAULT)
BDHD(UInt8, Cosine, KDT, DEFAULT)

BDHD(Int16, L2, BKT, DEFAULT)
BDHD(Int16, L2, KDT, DEFAULT)
BDHD(Int16, Cosine, BKT, DEFAULT)
BDHD(Int16, Cosine, KDT, DEFAULT)

//XVEC
BDHD(Float, L2, BKT, XVEC)
BDHD(Float, L2, KDT, XVEC)
BDHD(Float, Cosine, BKT, XVEC)
BDHD(Float, Cosine, KDT, XVEC)

BDHD(Int8, L2, BKT, XVEC)
BDHD(Int8, L2, KDT, XVEC)
BDHD(Int8, Cosine, BKT, XVEC)
BDHD(Int8, Cosine, KDT, XVEC)

BDHD(UInt8, L2, BKT, XVEC)
BDHD(UInt8, L2, KDT, XVEC)
BDHD(UInt8, Cosine, BKT, XVEC)
BDHD(UInt8, Cosine, KDT, XVEC)

BDHD(Int16, L2, BKT, XVEC)
BDHD(Int16, L2, KDT, XVEC)
BDHD(Int16, Cosine, BKT, XVEC)
BDHD(Int16, Cosine, KDT, XVEC)

#undef BDHD

#define BDSSD(VT, DM, ALGO, FT) \
BOOST_AUTO_TEST_CASE(TestBuildSSDIndex##VT##DM##ALGO##FT) { \
	std::string configName = BUILD_SSD_CONFIG(VT, DM, ALGO); \
	std::string base_config = CreateBaseConfig(SPTAG::VectorValueType::VT, SPTAG::DistCalcMethod::DM, SPTAG::IndexAlgoType::ALGO, VECTOR_DIM, \
		VECTORS(VT, FT), SPTAG::VectorFileType::FT, VECTOR_NUM, "", \
		"", SPTAG::VectorFileType::Undefined, -1, "", \
		"", SPTAG::VectorFileType::Undefined, -1, "", \
		"", SPTAG::TruthFileType::Undefined, \
		false, \
		HEAD_IDS(VT, DM, FT), \
		"", \
		HEAD_INDEX(VT, DM, ALGO, FT), \
		SSD_INDEX(VT, DM, ALGO, FT) \
		); \
TestBuildSSDIndex(\
	configName, \
	HEAD_IDS(VT, DM, FT), \
	HEAD_INDEX(VT, DM, ALGO, FT), \
	BUILD_SSD_BUILDER_CONFIG(VT, DM, ALGO), \
	SSD_INDEX(VT, DM, ALGO, FT), \
	true, \
	base_config \
);} \

// DEFAULT
BDSSD(Float, L2, BKT, DEFAULT)
BDSSD(Float, L2, KDT, DEFAULT)
BDSSD(Float, Cosine, BKT, DEFAULT)
BDSSD(Float, Cosine, KDT, DEFAULT)

BDSSD(Int8, L2, BKT, DEFAULT)
BDSSD(Int8, L2, KDT, DEFAULT)
BDSSD(Int8, Cosine, BKT, DEFAULT)
BDSSD(Int8, Cosine, KDT, DEFAULT)

BDSSD(UInt8, L2, BKT, DEFAULT)
BDSSD(UInt8, L2, KDT, DEFAULT)
BDSSD(UInt8, Cosine, BKT, DEFAULT)
BDSSD(UInt8, Cosine, KDT, DEFAULT)

BDSSD(Int16, L2, BKT, DEFAULT)
BDSSD(Int16, L2, KDT, DEFAULT)
BDSSD(Int16, Cosine, BKT, DEFAULT)
BDSSD(Int16, Cosine, KDT, DEFAULT)

// XVEC
BDSSD(Float, L2, BKT, XVEC)
BDSSD(Float, L2, KDT, XVEC)
BDSSD(Float, Cosine, BKT, XVEC)
BDSSD(Float, Cosine, KDT, XVEC)

BDSSD(Int8, L2, BKT, XVEC)
BDSSD(Int8, L2, KDT, XVEC)
BDSSD(Int8, Cosine, BKT, XVEC)
BDSSD(Int8, Cosine, KDT, XVEC)

BDSSD(UInt8, L2, BKT, XVEC)
BDSSD(UInt8, L2, KDT, XVEC)
BDSSD(UInt8, Cosine, BKT, XVEC)
BDSSD(UInt8, Cosine, KDT, XVEC)

BDSSD(Int16, L2, BKT, XVEC)
BDSSD(Int16, L2, KDT, XVEC)
BDSSD(Int16, Cosine, BKT, XVEC)
BDSSD(Int16, Cosine, KDT, XVEC)
#undef BDSSD


#define SCSSD(VT, DM, ALGO, FT, TFT) \
BOOST_AUTO_TEST_CASE(TestSearchSSDIndex##VT##DM##ALGO##FT##TFT) { \
	std::string configName = SEARCH_SSD_CONFIG(VT, DM, ALGO); \
	std::string base_config = CreateBaseConfig(SPTAG::VectorValueType::VT, SPTAG::DistCalcMethod::DM, SPTAG::IndexAlgoType::ALGO, VECTOR_DIM, \
		VECTORS(VT, FT), SPTAG::VectorFileType::FT, VECTOR_NUM, "", \
		QUERIES(VT, FT), SPTAG::VectorFileType::FT, QUERY_NUM, "", \
		QUERIES(VT, FT), SPTAG::VectorFileType::FT, QUERY_NUM, "", \
		TRUTHSET(VT, DM, FT, TFT), SPTAG::TruthFileType::TFT, \
		true, \
		HEAD_IDS(VT, DM, FT), \
		"", \
		HEAD_INDEX(VT, DM, ALGO, FT), \
		SSD_INDEX(VT, DM, ALGO, FT) \
		); \
TestSearchSSDIndex( \
	configName, \
	HEAD_IDS(VT, DM, FT), \
	HEAD_INDEX(VT, DM, ALGO, FT), \
	SEARCH_SSD_BUILDER_CONFIG(VT, DM, ALGO), \
	SSD_INDEX(VT, DM, ALGO, FT), \
	SEARCH_SSD_RESULT(VT, DM, ALGO, FT, TFT), \
	"", \
	base_config \
);} \

SCSSD(Float, L2, BKT, DEFAULT, TXT)
SCSSD(Float, L2, KDT, DEFAULT, TXT)
SCSSD(Float, Cosine, BKT, DEFAULT, TXT)
SCSSD(Float, Cosine, KDT, DEFAULT, TXT)

SCSSD(Int8, L2, BKT, DEFAULT, TXT)
SCSSD(Int8, L2, KDT, DEFAULT, TXT)
SCSSD(Int8, Cosine, BKT, DEFAULT, TXT)
SCSSD(Int8, Cosine, KDT, DEFAULT, TXT)

SCSSD(UInt8, L2, BKT, DEFAULT, TXT)
SCSSD(UInt8, L2, KDT, DEFAULT, TXT)
SCSSD(UInt8, Cosine, BKT, DEFAULT, TXT)
SCSSD(UInt8, Cosine, KDT, DEFAULT, TXT)

SCSSD(Int16, L2, BKT, DEFAULT, TXT)
SCSSD(Int16, L2, KDT, DEFAULT, TXT)
SCSSD(Int16, Cosine, BKT, DEFAULT, TXT)
SCSSD(Int16, Cosine, KDT, DEFAULT, TXT)


//Another
SCSSD(Float, L2, BKT, XVEC, XVEC)
SCSSD(Float, L2, KDT, XVEC, XVEC)
SCSSD(Float, Cosine, BKT, XVEC, XVEC)
SCSSD(Float, Cosine, KDT, XVEC, XVEC)

SCSSD(Int8, L2, BKT, XVEC, XVEC)
SCSSD(Int8, L2, KDT, XVEC, XVEC)
SCSSD(Int8, Cosine, BKT, XVEC, XVEC)
SCSSD(Int8, Cosine, KDT, XVEC, XVEC)

SCSSD(UInt8, L2, BKT, XVEC, XVEC)
SCSSD(UInt8, L2, KDT, XVEC, XVEC)
SCSSD(UInt8, Cosine, BKT, XVEC, XVEC)
SCSSD(UInt8, Cosine, KDT, XVEC, XVEC)

SCSSD(Int16, L2, BKT, XVEC, XVEC)
SCSSD(Int16, L2, KDT, XVEC, XVEC)
SCSSD(Int16, Cosine, BKT, XVEC, XVEC)
SCSSD(Int16, Cosine, KDT, XVEC, XVEC)
#undef SCSSD

BOOST_AUTO_TEST_CASE(RUN_FROM_MAP) {
	RunFromMap();
}

BOOST_AUTO_TEST_SUITE_END()