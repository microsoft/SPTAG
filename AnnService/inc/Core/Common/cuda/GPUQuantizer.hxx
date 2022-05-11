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

#include "inc/Core/VectorIndex.h"

using namespace std;

using namespace SPTAG;

/*********************************************************************
* Object representing a Dim-dimensional point, with each coordinate
* represented by a element of datatype T
* NOTE: Dim must be templated so that we can store coordinate values in registers
*********************************************************************/
class GPU_PQQuantizer {

  private:
    int m_NumSubvectors;
    size_t m_KsPerSubvector;
    size_t m_BlockSize;
    
    float* m_DistanceTables; // Don't need L2 and cosine tables, just copy from correct tables on host

    inline size_t m_DistIndexCalc(size_t i, size_t j, size_t k) {
      return m_BlockSize * i + j * m_KsPerSubvector + k;
    }

  public:

    __host__ GPU_PQQuantizer() {}
    template<typename T>
    __host__ GPU_PQQuantizer(std::shared_ptr<SPTAG::COMMON::PQQuantizer<T>> pQuantizer, DistMetric metric) {
      
//      SPTAG::COMMON::PQQuantizer<int>* pq_quantizer = dynamic_cast<SPTAG::COMMON::PQQuantizer<int>*>(quantizer);
//      int numSubvectors = COMMON::DistanceUtils::Quantizer->GetNumSubvectors();
      int numSubvectors = pQuantizer->GetNumSubvectors();
      int ksPer = pQuantizer->GetKsPerSubvector();
      int blockSize = pQuantizer->GetBlockSize();
      printf("ns:%d, ks:%d, bs:%d\n", numSubvectors, ksPer, blockSize);
    }

    __host__ GPU_PQQuantizer(size_t numSubvectors, size_t ksPerSubvector, size_t blockSize, std::unique_ptr<const float[]> distanceTable) : m_NumSubvectors(numSubvectors), m_KsPerSubvector(ksPerSubvector), m_BlockSize(blockSize) 
    {
      CUDA_CHECK(cudaMalloc(&m_DistanceTables, blockSize * numSubvectors * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(m_DistanceTables, distanceTable.get(), blockSize*numSubvectors*sizeof(float), cudaMemcpyHostToDevice));
    }

    __host__ ~GPU_PQQuantizer()
    {
      cudaFree(m_DistanceTables);
    }

    // TODO - Optimize quantized distance comparator
    __device__ float dist(const uint8_t* pX, const uint8_t* pY, DistMetric metric) {
      float out = 0;
      for(int i = 0; i < m_NumSubvectors; ++i) {
        out += m_DistanceTables[m_DistIndexCalc(i, pX[i], pY[i])];
      } 

      if(metric == DistMetric::Cosine) {
        out = 1 - out;
      }
      return out;
    }

};

#endif // GPUQUANTIZER_H
