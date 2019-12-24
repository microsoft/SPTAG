#include "inc/Test.h"
#include <string>
#include <fstream>
#include <atlstr.h>
#include <random>
#include <type_traits>
#include <functional>

#include <boost/filesystem.hpp>

#include "inc/Core/Common.h"
#include "inc/Helper/StringConvert.h"
#include "inc/SSDServing/IndexBuildManager/main.h"

using namespace std;

void InternalGenerateVectors(string fileName, function<void(SPTAG::SizeType&, SPTAG::DimensionType&, ofstream&)> callback) {
	if (boost::filesystem::exists(fileName))
	{
		fprintf(stdout, "%s was generated. Skip generation.", fileName.c_str());
		return;
	}
	ofstream of(fileName, ofstream::binary);
	if (!of.is_open())
	{
		fprintf(stderr, "%s can't be opened. ", fileName.c_str());
		BOOST_CHECK(false);
		return;
	}
	SPTAG::SizeType rows = 10000;
	SPTAG::DimensionType dims = 128;
	of.write(reinterpret_cast<char*>(&rows), sizeof(rows));
	of.write(reinterpret_cast<char*>(&dims), sizeof(dims));
	callback(rows, dims, of);
	of.close();
}

template<typename T>
void GenerateVectors(string fileName);

template<>
void GenerateVectors<float>(string fileName) {

	InternalGenerateVectors(fileName, [&](SPTAG::SizeType& rows, SPTAG::DimensionType& dims, ofstream& of) {
		mt19937 mt(432432);
		uniform_real_distribution<float> dist(-1000, 1000);
		for (size_t i = 0; i < rows * dims; i++)
		{
			float cur = dist(mt);
			of.write(reinterpret_cast<char*>(&cur), sizeof(cur));
		}
		});
}

template<>
void GenerateVectors<int8_t>(string fileName) {

	InternalGenerateVectors(fileName, [&](SPTAG::SizeType& rows, SPTAG::DimensionType& dims, ofstream& of) {
		mt19937 mt(432432);
		uniform_int_distribution<int8_t> dist(INT8_MIN, INT8_MAX);
		for (size_t i = 0; i < rows * dims; i++)
		{
			int8_t cur = dist(mt);
			of.write(reinterpret_cast<char*>(&cur), sizeof(cur));
		}
		});
}

template<>
void GenerateVectors<uint8_t>(string fileName) {

	InternalGenerateVectors(fileName, [&](SPTAG::SizeType& rows, SPTAG::DimensionType& dims, ofstream& of) {
		mt19937 mt(432432);
		uniform_int_distribution<uint8_t> dist(0, UINT8_MAX);
		for (size_t i = 0; i < rows * dims; i++)
		{
			uint8_t cur = dist(mt);
			of.write(reinterpret_cast<char*>(&cur), sizeof(cur));
		}
		});
}

template<>
void GenerateVectors<int16_t>(string fileName) {

	InternalGenerateVectors(fileName, [&](SPTAG::SizeType& rows, SPTAG::DimensionType& dims, ofstream& of) {
		mt19937 mt(432432);
		uniform_int_distribution<int16_t> dist(-1000, 1000);
		for (size_t i = 0; i < rows * dims; i++)
		{
			int16_t cur = dist(mt);
			of.write(reinterpret_cast<char*>(&cur), sizeof(cur));
		}
		});
}

void TestHead(string vectorsName, string configName, string OutputIDFile, string OutputVectorFile, 
	SPTAG::VectorValueType vecType, SPTAG::DistCalcMethod distMethod) {

	switch (vecType)
	{
#define DefineVectorValueType(Name, Type) \
case SPTAG::VectorValueType::Name: \
GenerateVectors<Type>(vectorsName); \
break; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
	default:
		break;
	}

	ofstream config(configName);
	if (!config.is_open())
	{
		fprintf(stderr, "%s can't be opened. ", configName.c_str());
		BOOST_CHECK(false);
		return;
	}
	config << "[SelectHead]" << endl;
	config << "VectorFilePath=" << vectorsName << endl;
	config << "VectorValueType=" << SPTAG::Helper::Convert::ConvertToString(vecType) << endl;
	config << "DistCalcMethod=" << SPTAG::Helper::Convert::ConvertToString(distMethod) << endl;

	config << "TreeNumber=" << "1" << endl;
	config << "BKTKmeansK=" << "32" << endl;
	config << "BKTLeafSize=" << "8" << endl;
	config << "SamplesNumber=" << "1000" << endl;
	config << "SaveBKT=" << "true" <<endl;

	config << "AnalyzeOnly=" << "false" << endl;
	config << "CalcStd=" << "true" << endl;
	config << "SelectDynamically=" << "true" << endl;
	config << "NoOutput=" << "false" << endl;

	config << "SelectThreshold=" << "6" << endl;
	config << "SplitFactor=" << "10" << endl;
	config << "SplitThreshold=" << "25" << endl;
	config << "Ratio=" << "0.2" << endl;
	config << "RecursiveCheckSmallCluster=" << "true" << endl;
	config << "PrintSizeCount=" << "true" << endl;

	config << "OutputIDFile=" << OutputIDFile << endl;
	config << "OutputVectorFile=" << OutputVectorFile << endl;

	config.close();

	char* arg1 = new char[100];
	char* arg2 = new char[100];
	strcpy_s(arg1, 100, "SSDServing.exe");
	strcpy_s(arg2, 100, configName.c_str());
	char* params[2] = { arg1, arg2 };
	SPTAG::SSDServing::internalMain(2, params);
	delete[] arg1;
	delete[] arg2;
}

/*
OLD commands:

SelectHead_BKT.exe -v test_head_Float_L2.bin -t float -d l2 --kmeans 32 --leafsize 8 --clustersample 1000 --min 6 --split 10 --max 25 -r 0.2 --packsmallcluster --sizecount -oid old_head_vector_ids_Float_L2.bin -ov old_head_vectors_Float_L2.bin
SelectHead_BKT.exe -v test_head_Float_Cosine.bin -t float -d cosine --kmeans 32 --leafsize 8 --clustersample 1000 --min 6 --split 10 --max 25 -r 0.2 --packsmallcluster --sizecount -oid old_head_vector_ids_Float_Cosine.bin -ov old_head_vectors_Float_Cosine.bin
SelectHead_BKT.exe -v test_head_Int16_L2.bin -t int16 -d l2 --kmeans 32 --leafsize 8 --clustersample 1000 --min 6 --split 10 --max 25 -r 0.2 --packsmallcluster --sizecount -oid old_head_vector_ids_Int16_L2.bin -ov old_head_vectors_Int16_L2.bin
SelectHead_BKT.exe -v test_head_Int16_Cosine.bin -t int16 -d cosine --kmeans 32 --leafsize 8 --clustersample 1000 --min 6 --split 10 --max 25 -r 0.2 --packsmallcluster --sizecount -oid old_head_vector_ids_Int16_Cosine.bin -ov old_head_vectors_Int16_Cosine.bin
SelectHead_BKT.exe -v test_head_Int8_L2.bin -t int8 -d l2 --kmeans 32 --leafsize 8 --clustersample 1000 --min 6 --split 10 --max 25 -r 0.2 --packsmallcluster --sizecount -oid old_head_vector_ids_Int8_L2.bin -ov old_head_vectors_Int8_L2.bin
SelectHead_BKT.exe -v test_head_Int8_Cosine.bin -t int8 -d cosine --kmeans 32 --leafsize 8 --clustersample 1000 --min 6 --split 10 --max 25 -r 0.2 --packsmallcluster --sizecount -oid old_head_vector_ids_Int8_Cosine.bin -ov old_head_vectors_Int8_Cosine.bin

*/
BOOST_AUTO_TEST_SUITE(SSDServingTest)

#define WTEV(VT, DM) \
BOOST_AUTO_TEST_CASE(TestHead##VT##DM) \
{ \
	string vectorsName = "test_head_"#VT"_"#DM".bin"; \
	string configName = "test_head_"#VT"_"#DM".ini"; \
	string OutputIDFile = "new_head_vector_ids_"#VT"_"#DM".bin"; \
	string OutputVectorFile = "new_head_vectors_"#VT"_"#DM".bin"; \
\
	TestHead(vectorsName, configName, OutputIDFile, OutputVectorFile, \
		SPTAG::VectorValueType::VT, SPTAG::DistCalcMethod::DM); \
} \

// OLD std: 29.402
// NEW std: 43.329
WTEV(Float, L2)
// OLD std: 13.585
// NEW std: 25.394
WTEV(Float, Cosine)
// OLD std: 10.231
// NEW std: 41.102
WTEV(Int16, L2)
// OLD std: 8.387
// NEW std: 18.442
WTEV(Int16, Cosine)
// OLD: UInt8 not supported
// NEW std: 37.352
WTEV(UInt8, L2)
// OLD: UInt8 not supported
// NEW std: 146.950
WTEV(UInt8, Cosine)
// OLD std: 13.486
// NEW std: 37.944
WTEV(Int8, L2)
// OLD std: 12.597
// NEW std: 15.848
WTEV(Int8, Cosine)
#undef WTEV

BOOST_AUTO_TEST_SUITE_END()