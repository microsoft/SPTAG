// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/TruthSet.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common/QueryResultSet.h"

#if defined(GPU)
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <typeinfo>
#include <cuda_fp16.h>

#include "inc/Core/Common/cuda/KNN.hxx"
#include "inc/Core/Common/cuda/params.h"
#endif

namespace SPTAG
{
    namespace COMMON
    {
#if defined(GPU)
        template<typename T>
        void TruthSet::GenerateTruth(std::shared_ptr<VectorSet> querySet, std::shared_ptr<VectorSet> vectorSet, const std::string truthFile,
            const SPTAG::DistCalcMethod distMethod, const int K, const SPTAG::TruthFileType p_truthFileType, const std::shared_ptr<IQuantizer>& quantizer) {
            if (querySet->Dimension() != vectorSet->Dimension() && !quantizer)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "query and vector have different dimensions.");
                exit(1);
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin to generate truth for query(%d,%d) and doc(%d,%d)...\n", querySet->Count(), querySet->Dimension(), vectorSet->Count(), vectorSet->Dimension());
            std::vector< std::vector<SPTAG::SizeType> > truthset(querySet->Count(), std::vector<SPTAG::SizeType>(K, 0));
            std::vector< std::vector<float> > distset(querySet->Count(), std::vector<float>(K, 0));

            GenerateTruthGPU<T>(querySet, vectorSet, truthFile, distMethod, K, p_truthFileType, quantizer, truthset, distset);

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start to write truth file...\n");
            writeTruthFile(truthFile, querySet->Count(), K, truthset, distset, p_truthFileType);

            auto ptr = SPTAG::f_createIO();
            if (ptr == nullptr || !ptr->Initialize((truthFile + ".dist.bin").c_str(), std::ios::out | std::ios::binary)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to create the file:%s\n", (truthFile + ".dist.bin").c_str());
                exit(1);
            }

            int int32_queryNumber = (int)querySet->Count();
            ptr->WriteBinary(4, (char*)&int32_queryNumber);
            ptr->WriteBinary(4, (char*)&K);

            for (size_t i = 0; i < int32_queryNumber; i++)
            {
                for (int k = 0; k < K; k++) {
                    if (ptr->WriteBinary(4, (char*)(&(truthset[i][k]))) != 4) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to write the truth dist file!\n");
                        exit(1);
                    }
                    if (ptr->WriteBinary(4, (char*)(&(distset[i][k]))) != 4) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to write the truth dist file!\n");
                        exit(1);
                    }
                }
            }
        }
#else
        template<typename T>
        void TruthSet::GenerateTruth(std::shared_ptr<VectorSet> querySet, std::shared_ptr<VectorSet> vectorSet, const std::string truthFile,
            const SPTAG::DistCalcMethod distMethod, const int K, const SPTAG::TruthFileType p_truthFileType, const std::shared_ptr<IQuantizer>& quantizer) {
            if (querySet->Dimension() != vectorSet->Dimension() && !quantizer)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "query and vector have different dimensions.");
                exit(1);
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin to generate truth for query(%d,%d) and doc(%d,%d)...\n", querySet->Count(), querySet->Dimension(), vectorSet->Count(), vectorSet->Dimension());
            std::vector< std::vector<SPTAG::SizeType> > truthset(querySet->Count(), std::vector<SPTAG::SizeType>(K, 0));
            std::vector< std::vector<float> > distset(querySet->Count(), std::vector<float>(K, 0));
            auto fComputeDistance = quantizer ? quantizer->DistanceCalcSelector<T>(distMethod) : COMMON::DistanceCalcSelector<T>(distMethod);
#pragma omp parallel for
            for (int i = 0; i < querySet->Count(); ++i)
            {
                SPTAG::COMMON::QueryResultSet<T> query((const T*)(querySet->GetVector(i)), K);
                query.SetTarget((const T*)(querySet->GetVector(i)), quantizer);
                for (SPTAG::SizeType j = 0; j < vectorSet->Count(); j++)
                {
                    float dist = fComputeDistance(query.GetQuantizedTarget(), reinterpret_cast<T*>(vectorSet->GetVector(j)), vectorSet->Dimension());
                    query.AddPoint(j, dist);
                }
                query.SortResult();

                for (int k = 0; k < K; k++)
                {
                    truthset[i][k] = (query.GetResult(k))->VID;
                    distset[i][k] = (query.GetResult(k))->Dist;
                }

            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start to write truth file...\n");
            writeTruthFile(truthFile, querySet->Count(), K, truthset, distset, p_truthFileType);

            auto ptr = SPTAG::f_createIO();
            if (ptr == nullptr || !ptr->Initialize((truthFile + ".dist.bin").c_str(), std::ios::out | std::ios::binary)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to create the file:%s\n", (truthFile + ".dist.bin").c_str());
                exit(1);
            }

            int int32_queryNumber = (int)querySet->Count();
            ptr->WriteBinary(4, (char*)&int32_queryNumber);
            ptr->WriteBinary(4, (char*)&K);

            for (size_t i = 0; i < int32_queryNumber; i++)
            {
                for (int k = 0; k < K; k++) {
                    if (ptr->WriteBinary(4, (char*)(&(truthset[i][k]))) != 4) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to write the truth dist file!\n");
                        exit(1);
                    }
                    if (ptr->WriteBinary(4, (char*)(&(distset[i][k]))) != 4) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to write the truth dist file!\n");
                        exit(1);
                    }
                }
            }
        }

#endif // (GPU)

#define DefineVectorValueType(Name, Type) template void TruthSet::GenerateTruth<Type>(std::shared_ptr<VectorSet> querySet, std::shared_ptr<VectorSet> vectorSet, const std::string truthFile, const SPTAG::DistCalcMethod distMethod, const int K, const SPTAG::TruthFileType p_truthFileType, const std::shared_ptr<IQuantizer>& quantizer);
#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
    }
}