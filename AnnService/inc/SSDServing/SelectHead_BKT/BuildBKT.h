#pragma once
#include "inc/Core/Common/BKTree.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/SSDServing/VectorSearch/TimeUtils.h"
#include "inc/SSDServing/IndexBuildManager/CommonDefines.h"

namespace SPTAG {
	namespace SSDServing {
		namespace SelectHead_BKT {
			template<typename T>
			void Clustering(std::shared_ptr<VectorSet> p_vectorSet, const Options& opts, std::vector<int>& selected, int clusternum) {
				std::vector<int> indices(p_vectorSet->Count(), 0);
				for (int i = 0; i < p_vectorSet->Count(); i++) indices[i] = i;
				std::random_shuffle(indices.begin(), indices.end());

				COMMON::Dataset<T> data(p_vectorSet->Count(), p_vectorSet->Dimension(), 1024 * 1024, p_vectorSet->Count() + 1, (T*)(p_vectorSet->GetData()));
				COMMON::KmeansArgs<T> args(clusternum, data.C(), (SizeType)indices.size(), opts.m_iNumberOfThreads, COMMON_OPTS.m_distCalcMethod);
				int clusters = SPTAG::COMMON::KmeansClustering(data, indices, 0, p_vectorSet->Count(), args, opts.m_iSamples, 1000000.0, true);

				SizeType first = 0;
				for (int k = 0; k < clusternum; k++) {
					if (args.counts[k] == 0) continue;
					SizeType cid = indices[first + args.counts[k] - 1];
					selected.push_back(cid);
					first += args.counts[k];
				}
			}

			template<typename T>
			std::shared_ptr<COMMON::BKTree> BuildBKT(std::shared_ptr<VectorSet> p_vectorSet, const Options& opts) {
				std::shared_ptr<COMMON::BKTree> bkt = std::make_shared<COMMON::BKTree>();
				bkt->m_iBKTKmeansK = opts.m_iBKTKmeansK;
				bkt->m_iBKTLeafSize = opts.m_iBKTLeafSize;
				bkt->m_iSamples = opts.m_iSamples;
				bkt->m_iTreeNumber = opts.m_iTreeNumber;
				bkt->m_fBalanceFactor = opts.m_fBalanceFactor;
				LOG(Helper::LogLevel::LL_Info, "Start invoking BuildTrees.\n");
				LOG(Helper::LogLevel::LL_Info, "BKTKmeansK: %d, BKTLeafSize: %d, Samples: %d, BKTLambdaFactor:%f TreeNumber: %d, ThreadNum: %d.\n",
					bkt->m_iBKTKmeansK, bkt->m_iBKTLeafSize, bkt->m_iSamples, bkt->m_fBalanceFactor, bkt->m_iTreeNumber, opts.m_iNumberOfThreads);
				VectorSearch::TimeUtils::StopW sw;
				COMMON::Dataset<T> data(p_vectorSet->Count(), p_vectorSet->Dimension(), 1024 * 1024, p_vectorSet->Count() + 1, (T*)(p_vectorSet->GetData()));
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
