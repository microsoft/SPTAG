#pragma once
#include "inc/Core/Common/BKTree.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/SSDServing/VectorSearch/TimeUtils.h"
#include "inc/SSDServing/IndexBuildManager/CommonDefines.h"

namespace SPTAG {
	namespace SSDServing {
		namespace SelectHead_BKT {
			template<typename T>
			std::shared_ptr<COMMON::BKTree> BuildBKT(std::shared_ptr<VectorSet> p_vectorSet, const Options& opts) {
				std::shared_ptr<COMMON::BKTree> bkt = std::make_shared<COMMON::BKTree>();
				bkt->m_iBKTKmeansK = opts.m_iBKTKmeansK;
				bkt->m_iBKTLeafSize = opts.m_iBKTLeafSize;
				bkt->m_iSamples = opts.m_iSamples;
				bkt->m_iTreeNumber = opts.m_iTreeNumber;
				LOG(Helper::LogLevel::LL_Info, "Start invoking BuildTrees.\n");
				LOG(Helper::LogLevel::LL_Info, "BKTKmeansK: %d, BKTLeafSize: %d, Samples: %d, TreeNumber: %d, ThreadNum: %d.\n",
					bkt->m_iBKTKmeansK, bkt->m_iBKTLeafSize, bkt->m_iSamples, bkt->m_iTreeNumber, opts.m_iNumberOfThreads);
				VectorSearch::TimeUtils::StopW sw;
				COMMON::Dataset<T> data(p_vectorSet->Count(), p_vectorSet->Dimension(), (T*)(p_vectorSet->GetData()));
				bkt->BuildTrees<T>(data, COMMON_OPTS.m_distCalcMethod, opts.m_iNumberOfThreads, nullptr, nullptr, true);
				double elapsedMinutes = sw.getElapsedMin();

				LOG(Helper::LogLevel::LL_Info, "End invoking BuildTrees.\n");
				LOG(Helper::LogLevel::LL_Info, "Invoking BuildTrees used time: %.2lf minutes (about %.2lf hours).\n", elapsedMinutes, elapsedMinutes / 60.0);

				std::stringstream bktFileNameBuilder;
				bktFileNameBuilder << COMMON_OPTS.m_vectorPath << ".bkt."
					<< opts.m_iBKTKmeansK << "_"
					<< opts.m_iBKTLeafSize << "_"
					<< opts.m_iTreeNumber << "_"
					<< opts.m_iSamples << "_"
					<< static_cast<int>(COMMON_OPTS.m_distCalcMethod) << ".bin";

				std::string bktFileName = bktFileNameBuilder.str();
				if (opts.m_saveBKT) {
					bkt->SaveTrees(bktFileName);
				}

				return bkt;
			}
		}
	}
}
