/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef _SPTAG_COMMON_CUDA_PERFTEST_H_
#define _SPTAG_COMMON_CUDA_PERFTEST_H_

#include "Refine.hxx"
#include "log.hxx"
#include "ThreadHeap.hxx"
#include "TPtree.hxx"
#include "GPUQuantizer.hxx"

#include <cuda/std/type_traits>
#include <chrono>


/***************************************************************************************
 * Function called by SPTAG to create an initial graph on the GPU.  
 ***************************************************************************************/
template<typename T>
void benchmarkDist(SPTAG::VectorIndex* index, int m_iGraphSize, int m_iNeighborhoodSize, int trees, int* results, int refines, int refineDepth, int graph, int leafSize, int initSize, int NUM_GPUS, int balanceFactor) {

  int m_disttype = (int)index->GetDistCalcMethod();
  size_t dataSize = (size_t)m_iGraphSize;
  int KVAL = m_iNeighborhoodSize;
  int dim = index->GetFeatureDim();
  size_t rawSize = dataSize*dim;

  if(index->m_pQuantizer != NULL) {
    SPTAG::COMMON::PQQuantizer<int>* pq_quantizer = (SPTAG::COMMON::PQQuantizer<int>*)index->m_pQuantizer.get();
    dim = pq_quantizer->GetNumSubvectors();
  }

//  srand(time(NULL)); // random number seed for TP tree random hyperplane partitions
  srand(1); // random number seed for TP tree random hyperplane partitions

/*******************************
 * Error checking
********************************/
  int numDevicesOnHost;
  CUDA_CHECK(cudaGetDeviceCount(&numDevicesOnHost));
  if(numDevicesOnHost < NUM_GPUS) {
    SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "HeadNumGPUs parameter %d, but only %d devices available on system.  Exiting.\n", NUM_GPUS, numDevicesOnHost);
    exit(1);
  }
  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Building Head graph with %d GPUs...\n", NUM_GPUS);
  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Debug, "Total of %d GPU devices on system, using %d of them.\n", numDevicesOnHost, NUM_GPUS);

/**** Compute result batch sizes for each GPU ****/
  std::vector<size_t> batchSize(NUM_GPUS);
  std::vector<size_t> resPerGPU(NUM_GPUS);
  for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
    CUDA_CHECK(cudaSetDevice(gpuNum));

    resPerGPU[gpuNum] = dataSize / NUM_GPUS; // Results per GPU
    if(dataSize % NUM_GPUS > gpuNum) resPerGPU[gpuNum]++; 

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpuNum)); // Get avil. memory
    SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "GPU %d - %s\n", gpuNum, prop.name);

    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

    size_t rawDataSize, pointSetSize, treeSize, resMemAvail, maxEltsPerBatch;

    rawDataSize = rawSize*sizeof(T);
    pointSetSize = sizeof(PointSet<T>);
    treeSize = 20*dataSize;
    resMemAvail = (freeMem*0.9) - (rawDataSize+pointSetSize+treeSize); // Only use 90% of total memory to be safe
    maxEltsPerBatch = resMemAvail / (dim*sizeof(T) + KVAL*sizeof(int));

    batchSize[gpuNum] = (std::min)(maxEltsPerBatch, resPerGPU[gpuNum]);

    SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Debug, "Memory for rawData:%lu MiB, pointSet structure:%lu MiB, Memory for TP trees:%lu MiB, Memory left for results:%lu MiB, total vectors:%lu, batch size:%d, total batches:%d\n", rawSize/1000000, pointSetSize/1000000, treeSize/1000000, resMemAvail/1000000, resPerGPU[gpuNum], batchSize[gpuNum], (((batchSize[gpuNum]-1)+resPerGPU[gpuNum]) / batchSize[gpuNum]));

  // If GPU memory is insufficient or so limited that we need so many batches it becomes inefficient, return error
    if(batchSize[gpuNum] == 0 || ((int)resPerGPU[gpuNum]) / batchSize[gpuNum] > 10000) {
      SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Insufficient GPU memory to build Head index on GPU %d.  Available GPU memory:%lu MB, Points and tpt require:%lu MB, leaving a maximum batch size of %d results to be computed, which is too small to run efficiently.\n", gpuNum, (freeMem)/1000000, (rawDataSize+pointSetSize+treeSize)/1000000, maxEltsPerBatch);
      exit(1);
    }
  }
  std::vector<size_t> GPUOffset(NUM_GPUS);
  GPUOffset[0] = 0;
  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU 0: results:%lu, offset:%lu\n", resPerGPU[0], GPUOffset[0]);
  for(int gpuNum=1; gpuNum < NUM_GPUS; ++gpuNum) {
    GPUOffset[gpuNum] = GPUOffset[gpuNum-1] + resPerGPU[gpuNum-1];
    SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU %d: results:%lu, offset:%lu\n", gpuNum, resPerGPU[gpuNum], GPUOffset[gpuNum]);
  }

  if(index->m_pQuantizer != NULL) {
    buildGraphGPU<uint8_t, float>(index, (size_t)m_iGraphSize, KVAL, trees, results, graph, leafSize, NUM_GPUS, balanceFactor, batchSize.data(), GPUOffset.data(), resPerGPU.data(), dim);
  }
  else if(typeid(T) == typeid(float)) {
          buildGraphGPU<T, float>(index, (size_t)m_iGraphSize, KVAL, trees, results, graph, leafSize, NUM_GPUS, balanceFactor, batchSize.data(), GPUOffset.data(), resPerGPU.data(), dim);
  }
  else if(typeid(T) == typeid(uint8_t) || typeid(T) == typeid(int8_t)) {
          buildGraphGPU<T, int32_t>(index, (size_t)m_iGraphSize, KVAL, trees, results, graph, leafSize, NUM_GPUS, balanceFactor, batchSize.data(), GPUOffset.data(), resPerGPU.data(), dim);
  }
  else {
    SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Selected datatype not currently supported.\n");
    exit(1);
  }
}


#endif
