/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Licensed under the MIT License.
 */

#ifndef _SPTAG_COMMON_CUDA_GPUQUANTIZER_H_
#define _SPTAG_COMMON_CUDA_GPUQUANTIZER_H_

#include<cuda.h>
#include<cstdint>
#include<vector>
#include<climits>
#include<float.h>
#include<unordered_set>
#include<cuda.h>
#include "params.h"

#include "../PQQuantizer.h"
#include "../OPQQuantizer.h"

#include "inc/Core/VectorIndex.h"

using namespace std;

using namespace SPTAG;

/*********************************************************************
* Object representing a Dim-dimensional point, with each coordinate
* represented by a element of datatype T
* NOTE: Dim must be templated so that we can store coordinate values in registers
*********************************************************************/

enum DistMetric { L2, Cosine };

enum RType {Int8, UInt8, Float, Int16};

namespace {

__host__ int GetRTypeWidth(RType type) {
  if(type == RType::Int8) return 1;
  if(type == RType::UInt8) return 1;
  if(type == RType::Int16) return 2;
  if(type == RType::Float) return 4;
}

class GPU_Quantizer {

//  private:
  public:
    int m_NumSubvectors;
    size_t m_KsPerSubvector;
    size_t m_BlockSize;
    int m_DimPerSubvector;
    
    float* m_DistanceTables; // Don't need L2 and cosine tables, just copy from correct tables on host

    RType m_type;

    void* m_codebooks; // Codebooks to reconstruct the original vectors

    __device__ inline size_t m_DistIndexCalc(size_t i, size_t j, size_t k) {
      return m_BlockSize * i + j * m_KsPerSubvector + k;
    }

  public:

    __host__ GPU_Quantizer() {}

    __host__ GPU_Quantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer, DistMetric metric) {

      // Make sure L2 is used, since other 
      if(metric != DistMetric::L2) {
        LOG(Helper::LogLevel::LL_Error, "Only L2 distance currently supported for PQ or OPQ\n");
        exit(1);
      }

      VectorValueType rType;
      
      if(quantizer->GetQuantizerType() == QuantizerType::PQQuantizer) {
        SPTAG::COMMON::PQQuantizer<int>* pq_quantizer = (SPTAG::COMMON::PQQuantizer<int>*)quantizer.get();

        m_NumSubvectors = pq_quantizer->GetNumSubvectors();
        m_KsPerSubvector = pq_quantizer->GetKsPerSubvector();
        m_BlockSize = pq_quantizer->GetBlockSize();
        m_DimPerSubvector = pq_quantizer->GetDimPerSubvector();

        LOG(Helper::LogLevel::LL_Debug, "Using PQ - numSubVectors:%d, KsPerSub:%ld, BlockSize:%ld, DimPerSub:%d, total size of tables:%ld\n", m_NumSubvectors, m_KsPerSubvector, m_BlockSize, m_DimPerSubvector, m_BlockSize*m_NumSubvectors*sizeof(float));

        rType = pq_quantizer->GetReconstructType();
        CUDA_CHECK(cudaMalloc(&m_DistanceTables, m_BlockSize * m_NumSubvectors * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(m_DistanceTables, pq_quantizer->GetL2DistanceTables(), m_BlockSize*m_NumSubvectors*sizeof(float), cudaMemcpyHostToDevice));
      }
      else if(quantizer->GetQuantizerType() == QuantizerType::OPQQuantizer) {
        SPTAG::COMMON::OPQQuantizer<int>* opq_quantizer = (SPTAG::COMMON::OPQQuantizer<int>*)quantizer.get();

        m_NumSubvectors = opq_quantizer->GetNumSubvectors();
        m_KsPerSubvector = opq_quantizer->GetKsPerSubvector();
        m_BlockSize = opq_quantizer->GetBlockSize();
        m_DimPerSubvector = opq_quantizer->GetDimPerSubvector();

        LOG(Helper::LogLevel::LL_Debug, "Using OPQ - numSubVectors:%d, KsPerSub:%ld, BlockSize:%ld, DimPerSub:%d, total size of tables:%ld\n", m_NumSubvectors, m_KsPerSubvector, m_BlockSize, m_DimPerSubvector, m_BlockSize*m_NumSubvectors*sizeof(float));

        rType = opq_quantizer->GetReconstructType();
        CUDA_CHECK(cudaMalloc(&m_DistanceTables, m_BlockSize * m_NumSubvectors * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(m_DistanceTables, opq_quantizer->GetL2DistanceTables(), m_BlockSize*m_NumSubvectors*sizeof(float), cudaMemcpyHostToDevice));
      }
      else {
        LOG(Helper::LogLevel::LL_Error, "Only PQ and OPQ quantizers are supported for GPU build\n");
        exit(1);
      }

      if(rType == SPTAG::VectorValueType::Float) m_type = RType::Float;
      else if(rType == SPTAG::VectorValueType::Int8) m_type = RType::Int8;
      else if(rType == SPTAG::VectorValueType::UInt8) m_type = RType::UInt8;
      else if(rType == SPTAG::VectorValueType::Int16) m_type = RType::Int16;

    }

    __host__ ~GPU_Quantizer()
    {
      cudaFree(m_DistanceTables);
    }

/*
    template<typename R>
    __device__ void ReconstructVector(uint8_t* qvec, R* vecout)
    {
      for (int i = 0; i < m_NumSubvectors; i++) {
        SizeType codebook_idx = (i * m_KsPerSubvector * m_DimPerSubvector) + (qvec[i] * m_DimPerSubvector);
        R* sub_vecout = &(vecout[i * m_DimPerSubvector]);
//        for (int j = 0; j < m_DimPerSubvector; j++) {
//          sub_vecout[j] = m_codebooks[codebook_idx + j];
//        }
        memcpy(sub_vecout, m_codebooks+codebook_idx, m_DimPerSubvector*sizeof(R));
      }
    }
*/

    // TODO - Optimize quantized distance comparator
    __forceinline__ __device__ float dist(uint8_t* pX, uint8_t* pY) {
      float totals[2]; totals[0]=0; totals[1]=0;
      for(int i = 0; i < m_NumSubvectors; i+=2) {
        totals[0] += m_DistanceTables[m_DistIndexCalc(i, pX[i], pY[i])];
        totals[1] += m_DistanceTables[m_DistIndexCalc(i+1, pX[i+1], pY[i+1])];
      } 

      return totals[0]+totals[1];
    }

    __device__ bool violatesRNG(uint8_t* a, uint8_t* b, float distance) {
      float between = dist(a, b);
      return between <= distance;
    }

// Attempt to improve perf by checking threshold during computation to short-circuit 
//   distance lookups
/*
    __forceinline__ __device__ float dist(uint8_t* pX, uint8_t* pY, float target) {
      float totals[2]; totals[0]=0; totals[1]=0;
      float temp;

      for(int i = 0; i < m_NumSubvectors; i+=2) {
        totals[0] += m_DistanceTables[m_DistIndexCalc(i, pX[i], pY[i])];
//        totals[1] += m_DistanceTables[m_DistIndexCalc(i+1, pX[i+1], pY[i+1])];
        temp = m_DistanceTables[m_DistIndexCalc(i+1, pX[i+1], pY[i+1])];

        if(totals[0] >= target || totals[0] >= (target*(i+1))/(float)m_NumSubvectors) {
          return target+1.0;
        }
        totals[0]+=temp;
      } 

      return totals[0];
    }

    __device__ bool violatesRNG_test(uint8_t* a, uint8_t* b, float distance) {
      float between = dist(a, b, distance);
      return between <= distance;
    }
*/

};

}

#endif // GPUQUANTIZER_H
