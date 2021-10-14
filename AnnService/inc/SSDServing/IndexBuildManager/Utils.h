#pragma once
#include <string>
#include <vector>
#include "inc/Core/Common.h"
#include <algorithm>
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/VectorSet.h"
#include "inc/SSDServing/VectorSearch/Options.h"
#include "inc/Helper/SimpleIniReader.h"

namespace SPTAG {
    namespace SSDServing {

        template<typename T>
        float Std(T* nums, size_t size) {
            float var = 0;

            float mean = 0;
            for (size_t i = 0; i < size; i++)
            {
                mean += nums[i] / static_cast<float>(size);
            }

            for (size_t i = 0; i < size; i++)
            {
				float temp = nums[i] - mean;
                var += pow(temp, 2);
            }
            var /= static_cast<float>(size);

            return sqrt(var);
        }

		struct Neighbor
		{
			SPTAG::SizeType key;
			float dist;

			Neighbor(SPTAG::SizeType k, float d);

			Neighbor(const Neighbor& o);

			bool operator < (const Neighbor& another) const;
		};

		void writeTruthFile(const std::string truthFile, size_t queryNumber, const int K, std::vector<std::vector<SPTAG::SizeType>>& truthset, SPTAG::TruthFileType TFT);

		template<typename T>
		void GenerateTruth(std::shared_ptr<VectorSet> querySet, std::shared_ptr<VectorSet> vectorSet, const std::string truthFile,
			const SPTAG::DistCalcMethod distMethod, const int K, const SPTAG::TruthFileType p_truthFileType) {
			
			std::vector< std::vector< SPTAG::SizeType> > truthset(querySet->Count(), std::vector<SPTAG::SizeType>(K, 0));

#pragma omp parallel for
			for (int i = 0; i < querySet->Count(); ++i)
			{
				if (querySet->Dimension() != vectorSet->Dimension())
				{
					LOG(Helper::LogLevel::LL_Error, "query and vector have different dimensions.");
					exit(-1);
				}

				std::vector<Neighbor> neighbours;
				bool isFirst = true;
				for (SPTAG::SizeType j = 0; j < vectorSet->Count(); j++)
				{
					float dist = SPTAG::COMMON::DistanceUtils::ComputeDistance<T>(reinterpret_cast<T *>(querySet->GetVector(i)), reinterpret_cast<T*>(vectorSet->GetVector(j)), querySet->Dimension(), distMethod);

					Neighbor nei(j, dist);
					neighbours.push_back(nei);
					if (neighbours.size() == K && isFirst)
					{
						std::make_heap(neighbours.begin(), neighbours.end());
						isFirst = false;
					}
					if (neighbours.size() > K)
					{
						std::push_heap(neighbours.begin(), neighbours.end());
						std::pop_heap(neighbours.begin(), neighbours.end());
						neighbours.pop_back();
					}
				}

				if (K != neighbours.size())
				{
					LOG(Helper::LogLevel::LL_Error, "K is too big.\n");
					exit(-1);
				}

				std::sort(neighbours.begin(), neighbours.end());

				for (size_t k = 0; k < K; k++)
				{
					truthset[i][k] = neighbours[k].key;
				}

			}

			writeTruthFile(truthFile, querySet->Count(), K, truthset, p_truthFileType);
		}

		bool readSearchSSDSec(const char* iniFile, VectorSearch::Options& opts);
		bool readSearchSSDSec(const Helper::IniReader& iniFileReader, VectorSearch::Options& opts);
    }
}