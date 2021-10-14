#pragma once
#include <string>
#include <vector>
#include "inc/Core/Common.h"
#include <algorithm>
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
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
                var += pow(temp, 2.0);
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

		void writeTruthFile(const std::string truthFile, size_t queryNumber, const int K, std::vector<std::vector<SPTAG::SizeType>>& truthset, std::vector<std::vector<float>>& distset, SPTAG::TruthFileType TFT);

		template<typename T>
		void GenerateTruth(std::shared_ptr<VectorSet> querySet, std::shared_ptr<VectorSet> vectorSet, const std::string truthFile,
			const SPTAG::DistCalcMethod distMethod, const int K, const SPTAG::TruthFileType p_truthFileType) {
			if (querySet->Dimension() != vectorSet->Dimension() && !SPTAG::COMMON::DistanceUtils::Quantizer)
			{
				LOG(Helper::LogLevel::LL_Error, "query and vector have different dimensions.");
				exit(-1);
			}

			std::vector< std::vector<SPTAG::SizeType> > truthset(querySet->Count(), std::vector<SPTAG::SizeType>(K, 0));
			std::vector< std::vector<float> > distset(querySet->Count(), std::vector<float>(K, 0));
#pragma omp parallel for
			for (int i = 0; i < querySet->Count(); ++i)
			{
				SPTAG::COMMON::QueryResultSet<T> query((const T*)(querySet->GetVector(i)), K);
				for (SPTAG::SizeType j = 0; j < vectorSet->Count(); j++)
				{
					float dist = SPTAG::COMMON::DistanceUtils::ComputeDistance<T>(query.GetQuantizedTarget(), reinterpret_cast<T*>(vectorSet->GetVector(j)), vectorSet->Dimension(), distMethod);
					query.AddPoint(j, dist);
				}
				query.SortResult();

				for (int k = 0; k < K; k++)
				{
					truthset[i][k] = (query.GetResult(k))->VID;
					distset[i][k] = (query.GetResult(k))->Dist;
				}

			}

			writeTruthFile(truthFile, querySet->Count(), K, truthset, distset, p_truthFileType);

			auto ptr = SPTAG::f_createIO();
			if (ptr == nullptr || !ptr->Initialize((truthFile + ".dist.bin").c_str(), std::ios::out | std::ios::binary)) {
				LOG(Helper::LogLevel::LL_Error, "Fail to create the file:%s\n", (truthFile + ".dist.bin").c_str());
				exit(1);
			}

			int int32_queryNumber = (int)querySet->Count();
			ptr->WriteBinary(4, (char*)&int32_queryNumber);
			ptr->WriteBinary(4, (char*)&K);

			for (size_t i = 0; i < int32_queryNumber; i++)
			{
				for (int k = 0; k < K; k++) {
					if (ptr->WriteBinary(4, (char*)(&(truthset[i][k]))) != 4) {
						LOG(Helper::LogLevel::LL_Error, "Fail to write the truth dist file!\n");
						exit(1);
					}
					if (ptr->WriteBinary(4, (char*)(&(distset[i][k]))) != 4) {
						LOG(Helper::LogLevel::LL_Error, "Fail to write the truth dist file!\n");
						exit(1);
					}
				}
			}
		}

		bool readSearchSSDSec(const char* iniFile, VectorSearch::Options& opts);
		bool readSearchSSDSec(const Helper::IniReader& iniFileReader, VectorSearch::Options& opts);
    }
}