// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#include "inc/Core/Common.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Core/Common/TruthSet.h"

#include "inc/SSDServing/main.h"
#include "inc/SSDServing/Utils.h"
#include "inc/SSDServing/SSDIndex.h"

using namespace SPTAG;

namespace SPTAG {
	namespace SSDServing {

		int BootProgram(bool forANNIndexTestTool, 
			std::map<std::string, std::map<std::string, std::string>>* config_map, 
			const char* configurationPath, 
			VectorValueType valueType,
			DistCalcMethod distCalcMethod,
			const char* dataFilePath, 
			const char* indexFilePath) {


			bool searchSSD = false;
			std::string QuantizerFilePath = "";
			if (forANNIndexTestTool) {
				(*config_map)[SEC_BASE]["ValueType"] = Helper::Convert::ConvertToString(valueType);
				(*config_map)[SEC_BASE]["DistCalcMethod"] = Helper::Convert::ConvertToString(distCalcMethod);
				(*config_map)[SEC_BASE]["VectorPath"] = dataFilePath;
				(*config_map)[SEC_BASE]["IndexDirectory"] = indexFilePath;

				(*config_map)[SEC_BUILD_HEAD]["KDTNumber"] = "2";
				(*config_map)[SEC_BUILD_HEAD]["NeighborhoodSize"] = "32";
				(*config_map)[SEC_BUILD_HEAD]["TPTNumber"] = "32";
				(*config_map)[SEC_BUILD_HEAD]["TPTLeafSize"] = "2000";
				(*config_map)[SEC_BUILD_HEAD]["MaxCheck"] = "4096";
				(*config_map)[SEC_BUILD_HEAD]["MaxCheckForRefineGraph"] = "4096";
				(*config_map)[SEC_BUILD_HEAD]["RefineIterations"] = "3";
				(*config_map)[SEC_BUILD_HEAD]["GraphNeighborhoodScale"] = "1";
				(*config_map)[SEC_BUILD_HEAD]["GraphCEFScale"] = "1";

				(*config_map)[SEC_BASE]["DeleteHeadVectors"] = "true";
				(*config_map)[SEC_SELECT_HEAD]["isExecute"] = "true";
				(*config_map)[SEC_BUILD_HEAD]["isExecute"] = "true";
				(*config_map)[SEC_BUILD_SSD_INDEX]["isExecute"] = "true";
				(*config_map)[SEC_BUILD_SSD_INDEX]["BuildSsdIndex"] = "true";

				std::map<std::string, std::string>::iterator iter;
				if ((iter = (*config_map)[SEC_BASE].find("QuantizerFilePath")) != (*config_map)[SEC_BASE].end()) {
					QuantizerFilePath = iter->second;
				}
			}
			else {
				Helper::IniReader iniReader;
				iniReader.LoadIniFile(configurationPath);
				(*config_map)[SEC_BASE] = iniReader.GetParameters(SEC_BASE);
				(*config_map)[SEC_SELECT_HEAD] = iniReader.GetParameters(SEC_SELECT_HEAD);
				(*config_map)[SEC_BUILD_HEAD] = iniReader.GetParameters(SEC_BUILD_HEAD);
				(*config_map)[SEC_BUILD_SSD_INDEX] = iniReader.GetParameters(SEC_BUILD_SSD_INDEX);

				valueType = iniReader.GetParameter(SEC_BASE, "ValueType", valueType);
				distCalcMethod = iniReader.GetParameter(SEC_BASE, "DistCalcMethod", distCalcMethod);
				bool buildSSD = iniReader.GetParameter(SEC_BUILD_SSD_INDEX, "isExecute", false);
				searchSSD = iniReader.GetParameter(SEC_SEARCH_SSD_INDEX, "isExecute", false);
				QuantizerFilePath = iniReader.GetParameter(SEC_BASE, "QuantizerFilePath", std::string(""));

				for (auto& KV : iniReader.GetParameters(SEC_SEARCH_SSD_INDEX)) {
					std::string param = KV.first, value = KV.second;
					if (buildSSD && Helper::StrUtils::StrEqualIgnoreCase(param.c_str(), "BuildSsdIndex")) continue;
					if (buildSSD && Helper::StrUtils::StrEqualIgnoreCase(param.c_str(), "isExecute")) continue;
					if (Helper::StrUtils::StrEqualIgnoreCase(param.c_str(), "PostingPageLimit")) param = "SearchPostingPageLimit";
					if (Helper::StrUtils::StrEqualIgnoreCase(param.c_str(), "InternalResultNum")) param = "SearchInternalResultNum";
					(*config_map)[SEC_BUILD_SSD_INDEX][param] = value;
				}
			}


			LOG(Helper::LogLevel::LL_Info, "Set QuantizerFile = %s\n", QuantizerFilePath.c_str());

			std::shared_ptr<VectorIndex> index = VectorIndex::CreateInstance(IndexAlgoType::SPANN, valueType);
			if (!QuantizerFilePath.empty() && index->LoadQuantizer(QuantizerFilePath) != ErrorCode::Success)
			{
				exit(1);
			}
			if (index == nullptr) {
				LOG(Helper::LogLevel::LL_Error, "Cannot create Index with ValueType %s!\n", (*config_map)[SEC_BASE]["ValueType"].c_str());
				return -1;
			}

			for (auto& sectionKV : *config_map) {
				for (auto& KV : sectionKV.second) {
					index->SetParameter(KV.first, KV.second, sectionKV.first);
				}
			}

			if (index->BuildIndex() != ErrorCode::Success) {
				LOG(Helper::LogLevel::LL_Error, "Failed to build index.\n");
				exit(1);
			}

			SPANN::Options* opts = nullptr;


#define DefineVectorValueType(Name, Type) \
	if (index->GetVectorValueType() == VectorValueType::Name) { \
		opts = ((SPANN::Index<Type>*)index.get())->GetOptions(); \
	} \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

			if (opts == nullptr) {
				LOG(Helper::LogLevel::LL_Error, "Cannot get options.\n");
				exit(1);
			}

			if (opts->m_generateTruth)
			{
				LOG(Helper::LogLevel::LL_Info, "Start generating truth. It's maybe a long time.\n");
				SizeType dim = opts->m_dim;
				if (index->m_pQuantizer)
				{
					valueType = VectorValueType::UInt8;
					dim = index->m_pQuantizer->GetNumSubvectors();
				}
				std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(valueType, dim, opts->m_vectorType, opts->m_vectorDelimiter));
				auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
				if (ErrorCode::Success != vectorReader->LoadFile(opts->m_vectorPath))
				{
					LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
					exit(1);
				}
				std::shared_ptr<Helper::ReaderOptions> queryOptions(new Helper::ReaderOptions(opts->m_valueType, opts->m_dim, opts->m_queryType, opts->m_queryDelimiter));
				auto queryReader = Helper::VectorSetReader::CreateInstance(queryOptions);
				if (ErrorCode::Success != queryReader->LoadFile(opts->m_queryPath))
				{
					LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
					exit(1);
				}
				auto vectorSet = vectorReader->GetVectorSet();
				auto querySet = queryReader->GetVectorSet();
				if (distCalcMethod == DistCalcMethod::Cosine && !index->m_pQuantizer) vectorSet->Normalize(opts->m_iSSDNumberOfThreads);

				omp_set_num_threads(opts->m_iSSDNumberOfThreads);

#define DefineVectorValueType(Name, Type) \
	if (opts->m_valueType == VectorValueType::Name) { \
		COMMON::TruthSet::GenerateTruth<Type>(querySet, vectorSet, opts->m_truthPath, \
			distCalcMethod, opts->m_resultNum, opts->m_truthType, index->m_pQuantizer); \
	} \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

				LOG(Helper::LogLevel::LL_Info, "End generating truth.\n");
			}

			if (searchSSD) {
#define DefineVectorValueType(Name, Type) \
	if (opts->m_valueType == VectorValueType::Name) { \
        SSDIndex::Search((SPANN::Index<Type>*)(index.get())); \
	} \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
			}
			return 0;
		}
	}
}

// switch between exe and static library by _$(OutputType)
#ifdef _exe

int main(int argc, char* argv[]) {
	if (argc < 2)
	{
		LOG(Helper::LogLevel::LL_Error,
			"ssdserving configFilePath\n");
		exit(-1);
	}

	std::map<std::string, std::map<std::string, std::string>> my_map;
	auto ret = SSDServing::BootProgram(false, &my_map, argv[1]);
	return ret;
}

#endif
