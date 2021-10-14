// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/CommonHelper.h"
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
			void AdjustOptions(Options& p_opts, int p_vectorCount)
			{
				if (p_opts.m_headVectorCount != 0) p_opts.m_ratio = p_opts.m_headVectorCount * 1.0 / p_vectorCount;
				int headCnt = CalcHeadCnt(p_opts.m_ratio, p_vectorCount);
				if (headCnt == 0)
				{
					for (double minCnt = 1; headCnt == 0; minCnt += 0.2)
					{
						p_opts.m_ratio = minCnt / p_vectorCount;
						headCnt = CalcHeadCnt(p_opts.m_ratio, p_vectorCount);
					}

					LOG(Helper::LogLevel::LL_Info, "Setting requires to select none vectors as head, adjusted it to %d vectors\n", headCnt);
				}

				if (p_opts.m_iBKTKmeansK > headCnt)
				{
					p_opts.m_iBKTKmeansK = headCnt;
					LOG(Helper::LogLevel::LL_Info, "Setting of cluster number is less than head count, adjust it to %d\n", headCnt);
				}

				if (p_opts.m_selectThreshold == 0)
				{
					p_opts.m_selectThreshold = min(p_vectorCount - 1, static_cast<int>(1 / p_opts.m_ratio));
					LOG(Helper::LogLevel::LL_Info, "Set SelectThreshold to %d\n", p_opts.m_selectThreshold);
				}

				if (p_opts.m_splitThreshold == 0)
				{
					p_opts.m_splitThreshold = min(p_vectorCount - 1, static_cast<int>(p_opts.m_selectThreshold * 2));
					LOG(Helper::LogLevel::LL_Info, "Set SplitThreshold to %d\n", p_opts.m_splitThreshold);
				}

				if (p_opts.m_splitFactor == 0)
				{
					p_opts.m_splitFactor = min(p_vectorCount - 1, static_cast<int>(std::round(1 / p_opts.m_ratio) + 0.5));
					LOG(Helper::LogLevel::LL_Info, "Set SplitFactor to %d\n", p_opts.m_splitFactor);
				}
			}

			ErrorCode Bootstrap(Options& opts) {

				VectorSearch::TimeUtils::StopW sw;

				LOG(Helper::LogLevel::LL_Info, "Start loading vector file.\n");
				auto valueType = COMMON::DistanceUtils::Quantizer ? SPTAG::VectorValueType::UInt8 : COMMON_OPTS.m_valueType;
				std::shared_ptr<Helper::ReaderOptions> options(new Helper::ReaderOptions(valueType, COMMON_OPTS.m_dim, COMMON_OPTS.m_vectorType, COMMON_OPTS.m_vectorDelimiter));
				auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
				if (ErrorCode::Success != vectorReader->LoadFile(COMMON_OPTS.m_vectorPath))
				{
					LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
					exit(1);
				}
				auto vectorSet = vectorReader->GetVectorSet();
				if (COMMON_OPTS.m_distCalcMethod == DistCalcMethod::Cosine) vectorSet->Normalize(opts.m_iNumberOfThreads);
				LOG(Helper::LogLevel::LL_Info, "Finish loading vector file.\n");

				AdjustOptions(opts, vectorSet->Count());

				std::vector<int> selected;

				if (Helper::StrUtils::StrEqualIgnoreCase(opts.m_selectType.c_str(), "Random")) {
					LOG(Helper::LogLevel::LL_Info, "Start generating Random head.\n");
					selected.resize(vectorSet->Count());
					for (int i = 0; i < vectorSet->Count(); i++) selected[i] = i;
					std::random_shuffle(selected.begin(), selected.end());
					
					int headCnt = CalcHeadCnt(opts.m_ratio, vectorSet->Count());
					selected.resize(headCnt);

				} else if (Helper::StrUtils::StrEqualIgnoreCase(opts.m_selectType.c_str(), "Clustering")) {
					LOG(Helper::LogLevel::LL_Info, "Start generating Clustering head.\n");
					int headCnt = CalcHeadCnt(opts.m_ratio, vectorSet->Count());
					
					switch (valueType)
					{
#define DefineVectorValueType(Name, Type) \
    case VectorValueType::Name: \
        Clustering<Type>(vectorSet, opts, selected, headCnt); \
		break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

					default: break;
					}

				} else if (Helper::StrUtils::StrEqualIgnoreCase(opts.m_selectType.c_str(), "BKT")) {
					LOG(Helper::LogLevel::LL_Info, "Start generating BKT.\n");
					std::shared_ptr<COMMON::BKTree> bkt;
					switch (valueType)
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
						if (SelectHead(vectorSet, bkt, opts, counter, selected) != ErrorCode::Success)
						{
							LOG(Helper::LogLevel::LL_Error, "Failed to select head.\n");
							exit(1);
						}
					}
				}

				LOG(Helper::LogLevel::LL_Info,
					"Seleted Nodes: %u, about %.2lf%% of total.\n",
					static_cast<unsigned int>(selected.size()),
					selected.size() * 100.0 / vectorSet->Count());

				if (!opts.m_noOutput)
				{
					std::sort(selected.begin(), selected.end());

					std::shared_ptr<Helper::DiskPriorityIO> output = SPTAG::f_createIO(), outputIDs = SPTAG::f_createIO();
					if (output == nullptr || outputIDs == nullptr ||
						!output->Initialize(COMMON_OPTS.m_headVectorFile.c_str(), std::ios::binary | std::ios::out) ||
						!outputIDs->Initialize(COMMON_OPTS.m_headIDFile.c_str(), std::ios::binary | std::ios::out)) {
						LOG(Helper::LogLevel::LL_Error, "Failed to create output file:%s %s\n", COMMON_OPTS.m_headVectorFile.c_str(), COMMON_OPTS.m_headIDFile.c_str());
						exit(1);
					}

					SizeType val = static_cast<SizeType>(selected.size());
					if (output->WriteBinary(sizeof(val), reinterpret_cast<char*>(&val)) != sizeof(val)) {
						LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
						exit(1);
					}
					DimensionType dt = vectorSet->Dimension();
					if (output->WriteBinary(sizeof(dt), reinterpret_cast<char*>(&dt)) != sizeof(dt)) {
						LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
						exit(1);
					}
					for (auto& ele : selected)
					{
						uint64_t vid = static_cast<uint64_t>(ele);
						if (outputIDs->WriteBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) != sizeof(vid)) {
							LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
							exit(1);
						}
						if (output->WriteBinary(vectorSet->PerVectorDataSize(), (char*)(vectorSet->GetVector((SizeType)vid))) != vectorSet->PerVectorDataSize()) {
							LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
							exit(1);
						}
					}
				}

				double elapsedMinutes = sw.getElapsedMin();
				LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedMinutes, elapsedMinutes / 60.0);

				return ErrorCode::Success;
			}
		}
	}
}