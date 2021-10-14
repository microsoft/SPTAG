#include "inc/Helper/VectorSetReader.h"
#include "inc/SSDServing/SelectHead_BKT/BootSelectHead.h"
#include "inc/SSDServing/SelectHead_BKT/BuildBKT.h"
#include "inc/SSDServing/SelectHead_BKT/AnalyzeTree.h"
#include "inc/SSDServing/SelectHead_BKT/SelectHead.h"
#include "inc/SSDServing/VectorSearch/TimeUtils.h"
#include "inc/SSDServing/IndexBuildManager/CommonDefines.h"

namespace SPTAG {
	namespace SSDServing {
		namespace SelectHead_BKT {
			ErrorCode Bootstrap(Options& opts) {

				VectorSearch::TimeUtils::StopW sw;

				LOG(Helper::LogLevel::LL_Info, "Start loading vector file.\n");
				std::shared_ptr<Helper::ReaderOptions> options(new Helper::ReaderOptions(COMMON_OPTS.m_valueType, COMMON_OPTS.m_dim, COMMON_OPTS.m_vectorType, COMMON_OPTS.m_vectorDelimiter));
				auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
				if (ErrorCode::Success != vectorReader->LoadFile(COMMON_OPTS.m_vectorPath))
				{
					LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
					exit(1);
				}
				auto vectorSet = vectorReader->GetVectorSet();
				if (COMMON_OPTS.m_distCalcMethod == DistCalcMethod::Cosine) vectorSet->Normalize(opts.m_iNumberOfThreads);
				LOG(Helper::LogLevel::LL_Info, "Finish loading vector file.\n");

				LOG(Helper::LogLevel::LL_Info, "Start generating BKT.\n");
				std::shared_ptr<COMMON::BKTree> bkt;
				switch (COMMON_OPTS.m_valueType)
				{
#define DefineVectorValueType(Name, Type) \
    case VectorValueType::Name: \
        bkt = BuildBKT<Type>(vectorSet, opts); \
		break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

				default: break;
				}
				LOG(Helper::LogLevel::LL_Info, "Finish generating BKT.\n");

				std::unordered_map<int, int> counter;

				if (opts.m_calcStd)
				{
					CalcLeafSize(0, bkt, counter);
				}

				if (opts.m_analyzeOnly)
				{
					LOG(Helper::LogLevel::LL_Info, "Analyze Only.\n");

					std::vector<BKTNodeInfo> bktNodeInfos(bkt->size());

					// Always use the first tree
					DfsAnalyze(0, bkt, vectorSet, opts, 0, bktNodeInfos);

					LOG(Helper::LogLevel::LL_Info, "Analyze Finish.\n");
				}
				else {
					if (SelectHead(vectorSet, bkt, opts, counter) != ErrorCode::Success)
					{
						return ErrorCode::Fail;
					}
				}

				double elapsedMinutes = sw.getElapsedMin();
				LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedMinutes, elapsedMinutes / 60.0);

				return ErrorCode::Success;
			}
		}
	}
}