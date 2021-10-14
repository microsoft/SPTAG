#include <iostream>

#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/SSDServing/IndexBuildManager/main.h"
#include "inc/SSDServing/IndexBuildManager/CommonDefines.h"
#include "inc/SSDServing/IndexBuildManager/Utils.h"
#include "inc/SSDServing/SelectHead_BKT/BootSelectHead.h"
#include "inc/SSDServing/BuildHead/BootBuildHead.h"
#include "inc/SSDServing/VectorSearch/TimeUtils.h"
#include "inc/SSDServing/VectorSearch/BootVectorSearch.h"

using namespace SPTAG;

namespace SPTAG {
	namespace SSDServing {

		BaseOptions COMMON_OPTS;

		void BuildConfigFromMap(std::map< std::string, std::map<std::string, std::string> >* config_map,
			SSDServing::SelectHead_BKT::Options& select_head_opts,
			SSDServing::BuildHead::Options& build_head_opts,
			SSDServing::VectorSearch::Options& build_ssd_opts,
			SSDServing::VectorSearch::Options& search_ssd_opts
		) {
			for (auto& sectionKV : *config_map) {
				for (auto& KV : sectionKV.second) {
					if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(sectionKV.first.c_str(), SEC_BASE.c_str()))
					{
						COMMON_OPTS.SetParameter(KV.first.c_str(), KV.second.c_str());
					}
					else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(sectionKV.first.c_str(), SEC_SELECT_HEAD.c_str()))
					{
						select_head_opts.SetParameter(KV.first.c_str(), KV.second.c_str());
					}
					else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(sectionKV.first.c_str(), SEC_BUILD_HEAD.c_str()))
					{
						build_head_opts.SetParameter(KV.first.c_str(), KV.second.c_str());
					}
					else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(sectionKV.first.c_str(), SEC_BUILD_SSD_INDEX.c_str())) {
						build_ssd_opts.SetParameter(KV.first.c_str(), KV.second.c_str());
					}
					else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(sectionKV.first.c_str(), SEC_SEARCH_SSD_INDEX.c_str())) {
						search_ssd_opts.SetParameter(KV.first.c_str(), KV.second.c_str());
					}
				}
			}
		}

		void BuildConfigFromFile(const char* configurationPath,
			std::map< std::string, std::map<std::string, std::string> >* config_map,
			SSDServing::SelectHead_BKT::Options& select_head_opts,
			SSDServing::BuildHead::Options& build_head_opts,
			SSDServing::VectorSearch::Options& build_ssd_opts,
			SSDServing::VectorSearch::Options& search_ssd_opts
		) {
			Helper::IniReader iniReader;
			iniReader.LoadIniFile(configurationPath);
			(*config_map)[SEC_BASE] = iniReader.GetParameters(SEC_BASE);
			(*config_map)[SEC_SELECT_HEAD] = iniReader.GetParameters(SEC_SELECT_HEAD);
			(*config_map)[SEC_BUILD_HEAD] = iniReader.GetParameters(SEC_BUILD_HEAD);
			(*config_map)[SEC_BUILD_SSD_INDEX] = iniReader.GetParameters(SEC_BUILD_SSD_INDEX);
			(*config_map)[SEC_SEARCH_SSD_INDEX] = iniReader.GetParameters(SEC_SEARCH_SSD_INDEX);
			BuildConfigFromMap(config_map,
				select_head_opts,
				build_head_opts,
				build_ssd_opts,
				search_ssd_opts);
		}

		int BootProgram(bool forANNIndexTestTool, 
			std::map<std::string, std::map<std::string, std::string>>* config_map, 
			const char* configurationPath, 
			SPTAG::VectorValueType valueType,
			SPTAG::DistCalcMethod distCalcMethod,
			const char* dataFilePath, 
			const char* indexFilePath) {

			COMMON_OPTS = BaseOptions();
			SSDServing::SelectHead_BKT::Options slOpts;
			SSDServing::BuildHead::Options bhOpts;
			SSDServing::VectorSearch::Options build_ssd_opts;
			SSDServing::VectorSearch::Options search_ssd_opts;

			if (forANNIndexTestTool) {
				(*config_map)[SEC_BASE]["ValueType"] = SPTAG::Helper::Convert::ConvertToString(valueType);
				(*config_map)[SEC_BASE]["DistCalcMethod"] = SPTAG::Helper::Convert::ConvertToString(distCalcMethod);
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

				BuildConfigFromMap(config_map,
					slOpts,
					bhOpts,
					build_ssd_opts,
					search_ssd_opts
				);
			}
			else {
				BuildConfigFromFile(configurationPath,
					config_map,
					slOpts,
					bhOpts,
					build_ssd_opts,
					search_ssd_opts
				);
			}

			if (!COMMON_OPTS.m_quantizerFilePath.empty())
			{
				auto ptr = SPTAG::f_createIO();
				if (!ptr->Initialize(COMMON_OPTS.m_quantizerFilePath.c_str(), std::ios::binary | std::ios::in))
				{
					LOG(Helper::LogLevel::LL_Error, "Failed to read quantizer file.\n");
					exit(1);
				}
				auto code = SPTAG::COMMON::Quantizer::LoadQuantizer(ptr, QuantizerType::PQQuantizer, COMMON_OPTS.m_valueType);
				if (code != ErrorCode::Success)
				{
					LOG(Helper::LogLevel::LL_Error, "Failed to load quantizer.\n");
					exit(1);
				}
			}

			//Make directory if necessary
			std::string folderPath(COMMON_OPTS.m_indexDirectory);
			if (!folderPath.empty()) {
				if (*(folderPath.rbegin()) != FolderSep)
				{
					folderPath += FolderSep;
				}
	
				if (!direxists(folderPath.c_str()))
				{
					mkdir(folderPath.c_str());
				}
			}

			COMMON_OPTS.m_headIDFile = folderPath + COMMON_OPTS.m_headIDFile;
			COMMON_OPTS.m_headVectorFile = folderPath + COMMON_OPTS.m_headVectorFile;
			COMMON_OPTS.m_headIndexFolder = folderPath + COMMON_OPTS.m_headIndexFolder;
			COMMON_OPTS.m_ssdIndex = folderPath + COMMON_OPTS.m_ssdIndex;

			VectorSearch::TimeUtils::StopW sw;

			if (slOpts.m_execute) {
				SSDServing::SelectHead_BKT::Bootstrap(slOpts);
			}

			double selectHeadTime = sw.getElapsedSec();
			LOG(Helper::LogLevel::LL_Info, "select head time: %.2lf\n", selectHeadTime);
			sw.reset();

			if (bhOpts.m_execute) {
				SSDServing::BuildHead::Bootstrap(bhOpts, (*config_map)[SEC_BUILD_HEAD]);
			}

			double buildHeadTime = sw.getElapsedSec();
			LOG(Helper::LogLevel::LL_Info, "select head time: %.2lf build head time: %.2lf\n", selectHeadTime, buildHeadTime);
			sw.reset();

			if (build_ssd_opts.m_execute) {
				SSDServing::VectorSearch::Bootstrap(build_ssd_opts);
			}

			double buildSSDTime = sw.getElapsedSec();
			LOG(Helper::LogLevel::LL_Info, "select head time: %.2lf build head time: %.2lf build ssd time: %.2lf\n", selectHeadTime, buildHeadTime, buildSSDTime);
			sw.reset();

			if (COMMON_OPTS.m_generateTruth)
			{
				LOG(Helper::LogLevel::LL_Info, "Start generating truth. It's maybe a long time.\n");
				SPTAG::VectorValueType valueType = SPTAG::COMMON::DistanceUtils::Quantizer ? SPTAG::VectorValueType::UInt8 : COMMON_OPTS.m_valueType;
				std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(valueType, COMMON_OPTS.m_dim, COMMON_OPTS.m_vectorType, COMMON_OPTS.m_vectorDelimiter));
				auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
				if (ErrorCode::Success != vectorReader->LoadFile(COMMON_OPTS.m_vectorPath))
				{
					LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
					exit(1);
				}
				std::shared_ptr<Helper::ReaderOptions> queryOptions(new Helper::ReaderOptions(COMMON_OPTS.m_valueType, COMMON_OPTS.m_dim, COMMON_OPTS.m_queryType, COMMON_OPTS.m_queryDelimiter));
				auto queryReader = Helper::VectorSetReader::CreateInstance(queryOptions);
				if (ErrorCode::Success != queryReader->LoadFile(COMMON_OPTS.m_queryPath))
				{
					LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
					exit(1);
				}
				auto vectorSet = vectorReader->GetVectorSet();
				auto querySet = queryReader->GetVectorSet();
				if (COMMON_OPTS.m_distCalcMethod == DistCalcMethod::Cosine && !SPTAG::COMMON::DistanceUtils::Quantizer) vectorSet->Normalize(search_ssd_opts.m_iNumberOfThreads);

				omp_set_num_threads(search_ssd_opts.m_iNumberOfThreads);
#define DefineVectorValueType(Name, Type) \
	if (COMMON_OPTS.m_valueType == SPTAG::VectorValueType::Name) { \
		GenerateTruth<Type>(querySet, vectorSet, COMMON_OPTS.m_truthPath, \
			COMMON_OPTS.m_distCalcMethod, search_ssd_opts.m_resultNum, COMMON_OPTS.m_truthType); \
	} \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

				LOG(Helper::LogLevel::LL_Info, "End generating truth.\n");
			}

			if (search_ssd_opts.m_execute) {
				SSDServing::VectorSearch::Bootstrap(search_ssd_opts);
			}

			double searchSSDTime = sw.getElapsedSec();

			LOG(Helper::LogLevel::LL_Info, "select head time: %.2lf build head time: %.2lf build ssd time: %.2lf search ssd time: %.2lf\n",
				selectHeadTime,
				buildHeadTime,
				buildSSDTime,
				searchSSDTime
			);

			if (COMMON_OPTS.m_deleteHeadVectors) {
				if (remove(COMMON_OPTS.m_headVectorFile.c_str()) != 0) {
					LOG(Helper::LogLevel::LL_Warning, "Head vector file can't be removed.\n");
				}
			}
			return 0;
		}

		int BootProgram(const char* configurationPath) {
			std::map<std::string, std::map<std::string, std::string>> my_map;
			auto ret = BootProgram(false, &my_map, configurationPath);
			return ret;
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

	return SPTAG::SSDServing::BootProgram(argv[1]);
}

#endif