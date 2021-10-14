#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/SSDServing/IndexBuildManager/main.h"
#include "inc/SSDServing/IndexBuildManager/CommonDefines.h"
#include "inc/SSDServing/IndexBuildManager/Options.h"
#include "inc/SSDServing/IndexBuildManager/Utils.h"
#include "inc/SSDServing/SelectHead_BKT/BootSelectHead.h"
#include "inc/SSDServing/SelectHead_BKT/Options.h"
#include "inc/SSDServing/BuildHead/BootBuildHead.h"
#include "inc/SSDServing/BuildHead/Options.h"
#include "inc/SSDServing/VectorSearch/Options.h"
#include "inc/SSDServing/VectorSearch/TimeUtils.h"
#include "inc/SSDServing/VectorSearch/BootVectorSearch.h"

using namespace SPTAG;

namespace SPTAG {
	namespace SSDServing {

		BaseOptions COMMON_OPTS;

		int BootProgram(const char* configurationPath, bool forANNIndexTestTool,
			SPTAG::IndexAlgoType p_algoType,
			SPTAG::VectorValueType valueType,
			SPTAG::DistCalcMethod distCalcMethod,
			const char* dataFilePath, const char* indexFilePath) {
			Helper::IniReader iniReader;
			iniReader.LoadIniFile(configurationPath);

			auto& baseParameters = iniReader.GetParameters("Base");
			if (!baseParameters.empty())
			{
				for (const auto& iter : baseParameters)
				{
					COMMON_OPTS.SetParameter(iter.first.c_str(), iter.second.c_str());
				}
			}

			if (forANNIndexTestTool) {
				COMMON_OPTS.m_indexAlgoType = p_algoType;
				COMMON_OPTS.m_valueType = valueType;
				COMMON_OPTS.m_distCalcMethod = distCalcMethod;
				COMMON_OPTS.m_vectorPath = dataFilePath;
				COMMON_OPTS.m_indexDirectory = indexFilePath;
			}

			//Make directory if necessary
			std::string folderPath(COMMON_OPTS.m_indexDirectory);
			if (!folderPath.empty() && *(folderPath.rbegin()) != FolderSep)
			{
				folderPath += FolderSep;
			}

			//Maybe bug if folderPath == ""
			if (!direxists(folderPath.c_str()))
			{
				mkdir(folderPath.c_str());
			}

			VectorSearch::TimeUtils::StopW sw;

			SSDServing::SelectHead_BKT::Options slOpts;
			auto& selectHeadParameters = iniReader.GetParameters("SelectHead");
			for (const auto& iter : selectHeadParameters)
			{
				slOpts.SetParameter(iter.first.c_str(), iter.second.c_str());
			}
			if (slOpts.m_execute) {
				SSDServing::SelectHead_BKT::Bootstrap(slOpts);
			}

			double selectHeadTime = sw.getElapsedSec();
			LOG(Helper::LogLevel::LL_Info, "select head time: %.2lf\n", selectHeadTime);
			sw.reset();

			SSDServing::BuildHead::Options bhOpts;
			auto& buildHeadParameters = iniReader.GetParameters("BuildHead");
			for (const auto& iter : buildHeadParameters)
			{
				bhOpts.SetParameter(iter.first.c_str(), iter.second.c_str());
			}
			if (bhOpts.m_execute) {
				SSDServing::BuildHead::Bootstrap(bhOpts, buildHeadParameters);
			}

			double buildHeadTime = sw.getElapsedSec();
			LOG(Helper::LogLevel::LL_Info, "select head time: %.2lf\nbuild head time: %.2lf\n", selectHeadTime, buildHeadTime);
			sw.reset();

			SSDServing::VectorSearch::Options vsOpts;
			auto& buildSSDParameters = iniReader.GetParameters("BuildSSDIndex");
			for (const auto& iter : buildSSDParameters)
			{
				vsOpts.SetParameter(iter.first.c_str(), iter.second.c_str());
			}
			if (vsOpts.m_execute) {
				SSDServing::VectorSearch::Bootstrap(vsOpts);
			}

			double buildSSDTime = sw.getElapsedSec();
			LOG(Helper::LogLevel::LL_Info, "select head time: %.2lf\nbuild head time: %.2lf\nbuild ssd time: %.2lf\n", selectHeadTime, buildHeadTime, buildSSDTime);
			sw.reset();

			VectorSearch::Options opts;
			if (readSearchSSDSec(iniReader, opts))
			{
				if (COMMON_OPTS.m_generateTruth)
				{
					LOG(Helper::LogLevel::LL_Info, "Start generating truth. It's maybe a long time.\n");
					std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(COMMON_OPTS.m_valueType, COMMON_OPTS.m_dim, COMMON_OPTS.m_vectorType, COMMON_OPTS.m_vectorDelimiter));
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
					if (COMMON_OPTS.m_distCalcMethod == DistCalcMethod::Cosine) vectorSet->Normalize(opts.m_iNumberOfThreads);

#define DefineVectorValueType(Name, Type) \
	if (COMMON_OPTS.m_valueType == SPTAG::VectorValueType::Name) { \
		GenerateTruth<Type>(querySet, vectorSet, COMMON_OPTS.m_truthPath, \
			COMMON_OPTS.m_distCalcMethod, opts.m_resultNum, COMMON_OPTS.m_truthType); \
	} \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

					LOG(Helper::LogLevel::LL_Info, "End generating truth.\n");
				}

			}

			if (opts.m_execute) {
				SSDServing::VectorSearch::Bootstrap(opts);
			}

			double searchSSDTime = sw.getElapsedSec();

			LOG(Helper::LogLevel::LL_Info, "select head time: %.2lf\nbuild head time: %.2lf\nbuild ssd time: %.2lf\nsearch ssd time: %.2lf\n",
				selectHeadTime,
				buildHeadTime,
				buildSSDTime,
				searchSSDTime
			);
			return 0;
		}

		int BootProgram(const char* configurationPath) {
			return BootProgram(configurationPath, false);
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