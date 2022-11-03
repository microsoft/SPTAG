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

#ifndef _SPTAG_COMMON_CUDA_KNN_H_
#define _SPTAG_COMMON_CUDA_KNN_H_

#include "Refine.hxx"
#include "log.hxx"
#include "ThreadHeap.hxx"
#include "TPtree.hxx"
#include "GPUQuantizer.hxx"

#include <cuda/std/type_traits>
#include <chrono>

template<int Dim>
__device__ void findRNG_PQ(PointSet<uint8_t>* ps, TPtree* tptree, int KVAL, int* results, size_t min_id, size_t max_id, DistPair<float>* threadList, GPU_Quantizer* quantizer) {

  uint8_t query[Dim];
  uint8_t* candidate_vec;

  float max_dist = INFTY<float>();
  DistPair<float> temp;
  int read_id, write_id;

  for(int i=0; i<KVAL; i++) {
    threadList[i].dist = INFTY<float>();
  }

  size_t queryId;

  DistPair<float> target;
  DistPair<float> candidate;

  int blocks_per_leaf = gridDim.x / tptree->num_leaves;
  int threads_per_leaf = blocks_per_leaf*blockDim.x;
  int thread_id_in_leaf = blockIdx.x % blocks_per_leaf * blockDim.x + threadIdx.x;
  int leafIdx= blockIdx.x / blocks_per_leaf;
  size_t leaf_offset = tptree->leafs[leafIdx].offset;

  bool good;


  // Each point in the leaf is handled by a separate thread
  for(int i=thread_id_in_leaf; leafIdx < tptree->num_leaves && i<tptree->leafs[leafIdx].size; i+=threads_per_leaf) {
    if(tptree->leaf_points[leaf_offset+i] >= min_id && tptree->leaf_points[leaf_offset+i] < max_id) {
      queryId = tptree->leaf_points[leaf_offset + i];

      for(int j=0; j<Dim; ++j) {
        query[j] = ps->getVec(queryId)[j];
      }

      // Load results from previous iterations into shared memory heap
      // and re-compute distances since they are not stored in result set
      for(int j=0; j<KVAL; j++) {
        threadList[j].idx = results[(((long long int)(queryId-min_id))*(long long int)(KVAL))+j];
        if(threadList[j].idx != -1) {
          threadList[j].dist = quantizer->dist(query, ps->getVec(threadList[j].idx));
        }
        else {
          threadList[j].dist = INFTY<float>();
        }
      }
      max_dist = threadList[KVAL-1].dist;

      // Compare source query with all points in the leaf
      for(size_t j=0; j<tptree->leafs[leafIdx].size; ++j) {
        if(j!=i) {
          good = true;
	  candidate.idx = tptree->leaf_points[leaf_offset+j];
          candidate_vec = ps->getVec(candidate.idx);
          candidate.dist = quantizer->dist(query, candidate_vec);

          if(candidate.dist < max_dist){ // If it is a candidate to be added to neighbor list
	    for(read_id=0; candidate.dist > threadList[read_id].dist && good; read_id++) {
              if(quantizer->violatesRNG(candidate_vec, ps->getVec(threadList[read_id].idx), candidate.dist)) {
                good = false;
	      }
	    }
	    if(candidate.idx == threadList[read_id].idx)  // Ignore duplicates
              good = false;

	    if(good) { // candidate should be in RNG list
              target = threadList[read_id];
	      threadList[read_id] = candidate;
	      read_id++;
              for(write_id = read_id; read_id < KVAL && threadList[read_id].idx != -1; read_id++) {
                if(!quantizer->violatesRNG(ps->getVec(threadList[read_id].idx), candidate_vec, threadList[read_id].dist)) {
                  if(read_id == write_id) {
                    temp = threadList[read_id];
		    threadList[write_id] = target;
		    target = temp;
		  }
		  else {
                    threadList[write_id] = target;
		    target = threadList[read_id];
		  }
		  write_id++;
                }
	      }
	      if(write_id < KVAL) {
                threadList[write_id] = target;
                write_id++;
              }
	      for(int k=write_id; k<KVAL && threadList[k].idx != -1; k++) {
                threadList[k].dist = INFTY<float>();
	        threadList[k].idx = -1;
	      }
              max_dist = threadList[KVAL-1].dist;
	    }
          }
        }
      }
      for(size_t j=0; j<KVAL; j++) {
        results[(size_t)(queryId-min_id)*KVAL+j] = threadList[j].idx;
      }
    } // End if within batch
  } // End leaf node loop
}



/*****************************************************************************************
 * Perform the brute-force graph construction on each leaf node, while STRICTLY maintaining RNG properties.  May end up with less than K neighbors per vector.
 *****************************************************************************************/
template<typename T, typename SUMTYPE, int Dim>
__device__ void findRNG(PointSet<T>* ps, TPtree* tptree, int KVAL, int* results, size_t min_id, size_t max_id, DistPair<SUMTYPE>* threadList, SUMTYPE (*dist_comp)(T*,T*)) {

  T query[Dim];

  SUMTYPE max_dist = INFTY<SUMTYPE>();
  DistPair<SUMTYPE> temp;
  int read_id, write_id;

  for(int i=0; i<KVAL; i++) {
    threadList[i].dist = INFTY<SUMTYPE>();
  }

  size_t queryId;

  DistPair<SUMTYPE> target;
  DistPair<SUMTYPE> candidate;

  int blocks_per_leaf = gridDim.x / tptree->num_leaves;
  int threads_per_leaf = blocks_per_leaf*blockDim.x;
  int thread_id_in_leaf = blockIdx.x % blocks_per_leaf * blockDim.x + threadIdx.x;
  int leafIdx= blockIdx.x / blocks_per_leaf;
  size_t leaf_offset = tptree->leafs[leafIdx].offset;

  bool good;

  T* candidate_vec;

  // Each point in the leaf is handled by a separate thread
  for(int i=thread_id_in_leaf; leafIdx < tptree->num_leaves && i<tptree->leafs[leafIdx].size; i+=threads_per_leaf) {
    if(tptree->leaf_points[leaf_offset+i] >= min_id && tptree->leaf_points[leaf_offset+i] < max_id) {
      queryId = tptree->leaf_points[leaf_offset + i];

      for(int j=0; j<ps->dim; ++j) {
        query[j] = ps->getVec(queryId)[j];
      }

      // Load results from previous iterations into shared memory heap
      // and re-compute distances since they are not stored in result set
      for(int j=0; j<KVAL; j++) {
        threadList[j].idx = results[(((long long int)(queryId-min_id))*(long long int)(KVAL))+j];
        if(threadList[j].idx != -1) {
          threadList[j].dist = dist_comp(query, ps->getVec(threadList[j].idx));
        }
        else {
          threadList[j].dist = INFTY<SUMTYPE>();
        }
      }
      max_dist = threadList[KVAL-1].dist;

      // Compare source query with all points in the leaf
      for(size_t j=0; j<tptree->leafs[leafIdx].size; ++j) {
        if(j!=i) {
          good = true;
	  candidate.idx = tptree->leaf_points[leaf_offset+j];
          candidate_vec = ps->getVec(candidate.idx);
          candidate.dist = dist_comp(query, candidate_vec);

          if(max_dist >= candidate.dist){ // If it is a candidate to be added to neighbor list
	    for(read_id=0; candidate.dist > threadList[read_id].dist && good; read_id++) {
              if(violatesRNG<T,SUMTYPE>(candidate_vec, ps->getVec(threadList[read_id].idx), candidate.dist, dist_comp)) {
                good = false;
	      }
	    }
	    if(candidate.idx == threadList[read_id].idx)  // Ignore duplicates
              good = false;

	    if(good) { // candidate should be in RNG list
              target = threadList[read_id];
	      threadList[read_id] = candidate;
	      read_id++;
              for(write_id = read_id; read_id < KVAL && threadList[read_id].idx != -1; read_id++) {
                if(!violatesRNG<T,SUMTYPE>(ps->getVec(threadList[read_id].idx), candidate_vec, threadList[read_id].dist, dist_comp)) {
                  if(read_id == write_id) {
                    temp = threadList[read_id];
		    threadList[write_id] = target;
		    target = temp;
		  }
		  else {
                    threadList[write_id] = target;
		    target = threadList[read_id];
		  }
		  write_id++;
                }
	      }
	      if(write_id < KVAL) {
                threadList[write_id] = target;
                write_id++;
              }
	      for(int k=write_id; k<KVAL && threadList[k].idx != -1; k++) {
                threadList[k].dist = INFTY<SUMTYPE>();
	        threadList[k].idx = -1;
	      }
              max_dist = threadList[KVAL-1].dist;
	    }
          }
        }
      }
      for(size_t j=0; j<KVAL; j++) {
        results[(size_t)(queryId-min_id)*KVAL+j] = threadList[j].idx;
      }
    } // End if within batch
  } // End leaf node loop
}


#define RUN_KERNEL(size)          \
  if(dim <= size) {               \
    DistPair<SUMTYPE>* threadList = (&((DistPair<SUMTYPE>*)sharememory)[KVAL*threadIdx.x]); \
    SUMTYPE (*dist_comp)(T*,T*);    \
    dist_comp = &dist<T,SUMTYPE,size,metric>;  \
    findRNG<T,SUMTYPE,size>(ps, tptree, KVAL, results, min_id, max_id, threadList, dist_comp); \
    return; \
  } 

#define RUN_KERNEL_QUANTIZED(size) \
  if(dim <= size) {\
    DistPair<float>* threadList = (&((DistPair<float>*)sharememory)[KVAL*threadIdx.x]); \
    findRNG_PQ<size>((PointSet<uint8_t>*)ps, tptree, KVAL, results, min_id, max_id, threadList, quantizer); \
    return; \
  } 

#define MAX_SHAPE 1024

template<typename T, typename SUMTYPE, int metric>
__global__ void findRNG_selector(PointSet<T>* ps, TPtree* tptree, int KVAL, int* results, size_t min_id, size_t max_id, int dim, GPU_Quantizer* quantizer) {

  extern __shared__ char sharememory[];

// Enable dimension of dataset that you will be using for maximum performance
  if(quantizer == NULL) {
    RUN_KERNEL(64);
    RUN_KERNEL(100);
//    RUN_KERNEL(200);
//    RUN_KERNEL(MAX_SHAPE);
  }
  else {
//    RUN_KERNEL_QUANTIZED(25);
    RUN_KERNEL_QUANTIZED(50);
//    RUN_KERNEL_QUANTIZED(64);
    RUN_KERNEL_QUANTIZED(100);
  }

}


template<typename DTYPE, typename SUMTYPE, int metric>
void run_TPT_batch_multigpu(size_t dataSize, int** d_results, TPtree** tptrees, TPtree** d_tptrees, int iters, int levels, int NUM_GPUS, int KVAL, cudaStream_t* streams, std::vector<size_t> batch_min, std::vector<size_t> batch_max, int balanceFactor, PointSet<DTYPE>** d_pointset, int dim, GPU_Quantizer* quantizer, SPTAG::VectorIndex* index)
{
  // Set num blocks for all GPU kernel calls
  int KNN_blocks= max(tptrees[0]->num_leaves, BLOCKS);


  for(int tree_id=0; tree_id < iters; ++tree_id) { // number of TPTs used to create approx. KNN graph

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0)); // Get avil. memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "Starting TPtree number %d, Total memory of GPU 0:%ld, GPU 0 memory used:%ld\n", tree_id, totalMem, freeMem);

    auto before_tpt = std::chrono::high_resolution_clock::now(); // Start timer for TPT building

    create_tptree_multigpu<DTYPE>(tptrees, d_pointset, dataSize, levels, NUM_GPUS, streams, balanceFactor, index);
    CUDA_CHECK(cudaDeviceSynchronize());

            // Copy TPTs to each GPU
    for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
      CUDA_CHECK(cudaSetDevice(gpuNum));
      CUDA_CHECK(cudaMemcpy(d_tptrees[gpuNum], tptrees[gpuNum], sizeof(TPtree), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto after_tpt = std::chrono::high_resolution_clock::now();

    for(int gpuNum=0; gpuNum < NUM_GPUS; gpuNum++) {
      CUDA_CHECK(cudaSetDevice(gpuNum));
      // Compute the STRICT RNG for each leaf node
      findRNG_selector<DTYPE, SUMTYPE, metric>
                    <<<KNN_blocks,THREADS, sizeof(DistPair<SUMTYPE>)*KVAL*THREADS>>>
                    (d_pointset[gpuNum], d_tptrees[gpuNum], KVAL, d_results[gpuNum], 
                    batch_min[gpuNum], batch_max[gpuNum], dim, quantizer);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto after_work = std::chrono::high_resolution_clock::now();
 
    double loop_tpt_time = GET_CHRONO_TIME(before_tpt, after_tpt);
    double loop_work_time = GET_CHRONO_TIME(after_tpt, after_work);
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "All GPUs finished tree %d - tree build time:%.2lf, neighbor compute time:%.2lf\n", tree_id, loop_tpt_time, loop_work_time);

  } 
}

/****************************************************************************************
 * Create graph on the GPU in a series of 1 or more batches, graph is saved into @results and is stored on the CPU
 ***************************************************************************************/

template<typename DTYPE, typename SUMTYPE>
void buildGraphGPU(SPTAG::VectorIndex* index, size_t dataSize, int KVAL, int trees, int* results, int graphtype, int leafSize, int NUM_GPUS, int balanceFactor, size_t* batchSize, size_t* GPUOffset, size_t* resPerGPU, int dim) {

/**** Variables ****/
  int metric = (int)index->GetDistCalcMethod();
  bool use_q = (index->m_pQuantizer != NULL); // Using quantization?
  int levels = (int)std::log2(dataSize/leafSize); // TPT levels
  size_t rawSize = dataSize*dim;
  cudaError_t resultErr;

  printf("leafSize:%d, levels:%d\n", leafSize, levels);

  // Timers
  double tree_time=0.0;
  double KNN_time=0.0;
  double D2H_time = 0.0;
  double prep_time = 0.0;
  auto start_t = std::chrono::high_resolution_clock::now();

  // Host structures 
  TPtree** tptrees = new TPtree*[NUM_GPUS];
  std::vector<cudaStream_t> streams(NUM_GPUS);

  // GPU arrays / structures
  int** d_results = new int*[NUM_GPUS];
  TPtree** d_tptrees = new TPtree*[NUM_GPUS];
  DTYPE** d_data_raw = new DTYPE*[NUM_GPUS];
  PointSet<DTYPE>** d_pointset = new PointSet<DTYPE>*[NUM_GPUS];


/**** Varibales if PQ is enabled ****/
  GPU_Quantizer* d_quantizer = NULL;  // Only use if quantizer is enabled
  GPU_Quantizer* h_quantizer = NULL; 

  if(use_q) {
    h_quantizer = new GPU_Quantizer(index->m_pQuantizer, (DistMetric)metric);
    CUDA_CHECK(cudaMalloc(&d_quantizer, sizeof(GPU_Quantizer)));
    CUDA_CHECK(cudaMemcpy(d_quantizer, h_quantizer, sizeof(GPU_Quantizer), cudaMemcpyHostToDevice));
  }

/********* Allocate and transfer data to each GPU ********/
  // Extract and copy raw data
  for(int gpuNum=0; gpuNum < NUM_GPUS; gpuNum++) {
    CUDA_CHECK(cudaSetDevice(gpuNum));
    CUDA_CHECK(cudaStreamCreate(&streams[gpuNum]));

      LOG(SPTAG::Helper::LogLevel::LL_Debug, "Allocating raw coordinate data on GPU %d, total of %zu bytes\n", gpuNum, rawSize*sizeof(DTYPE));
      CUDA_CHECK(cudaMalloc(&d_data_raw[gpuNum], rawSize*sizeof(DTYPE)));
  }

  copyRawDataToMultiGPU<DTYPE>(index, d_data_raw, dataSize, dim, NUM_GPUS, streams.data());


// Allocate and copy structures ofr each GPU
  for(int gpuNum=0; gpuNum < NUM_GPUS; gpuNum++) {
    // Create PointSet structures and copy to each GPU
    CUDA_CHECK(cudaSetDevice(gpuNum));
    CUDA_CHECK(cudaMalloc(&d_pointset[gpuNum], sizeof(PointSet<DTYPE>)));
    PointSet<DTYPE> temp_ps;
    temp_ps.dim = dim;
    temp_ps.data = d_data_raw[gpuNum];
    temp_ps.metric = (DistMetric)metric;
    CUDA_CHECK(cudaMemcpy(d_pointset[gpuNum], &temp_ps, sizeof(PointSet<DTYPE>), cudaMemcpyHostToDevice));

    // Allocate and initialize TPT structures
    CUDA_CHECK(cudaMalloc(&d_tptrees[gpuNum], sizeof(TPtree)));
    tptrees[gpuNum] = new TPtree;

    tptrees[gpuNum]->initialize(dataSize, levels, dim);
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "TPT structure initialized for %lu points, %d levels, leaf size:%d\n", dataSize, levels, leafSize);

    // Allocate results buffer on each GPU
    CUDA_CHECK(cudaMalloc(&d_results[gpuNum], (size_t)batchSize[gpuNum]*KVAL*sizeof(int)));
  }

  // Temp variables for running multi-GPU batches 
  std::vector<size_t> curr_batch_size(NUM_GPUS);
  std::vector<size_t> batchOffset(NUM_GPUS);
  std::vector<size_t> batch_min(NUM_GPUS);
  std::vector<size_t> batch_max(NUM_GPUS);
  for(int gpuNum=0; gpuNum<NUM_GPUS; ++gpuNum) {
    batchOffset[gpuNum]=0;
  }

  auto batch_start_t = std::chrono::high_resolution_clock::now();
  prep_time = GET_CHRONO_TIME(start_t, batch_start_t);

  bool done = false;
  while(!done) { // Continue until all GPUs have completed all of their batches
  
    // Prep next batch for each GPU
    for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
      curr_batch_size[gpuNum] = batchSize[gpuNum];
     // Check for final batch
      if(batchOffset[gpuNum]+batchSize[gpuNum] > resPerGPU[gpuNum]) {
        curr_batch_size[gpuNum] = resPerGPU[gpuNum]-batchOffset[gpuNum];
      }
      batch_min[gpuNum] = GPUOffset[gpuNum]+batchOffset[gpuNum];
      batch_max[gpuNum] = batch_min[gpuNum] + curr_batch_size[gpuNum];

      LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU %d - starting batch with min_id:%ld, max_id:%ld\n", gpuNum, batch_min[gpuNum], batch_max[gpuNum]);

      CUDA_CHECK(cudaSetDevice(gpuNum));
      // Initialize results to all -1 (special value that is set to distance INFTY)
      CUDA_CHECK(cudaMemset(d_results[gpuNum], -1, (size_t)batchSize[gpuNum]*KVAL*sizeof(int)));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /***** Run batch on GPU (all TPT iters) *****/
    if(metric == (int)DistMetric::Cosine) {
      if(d_quantizer != NULL) {
        LOG(Helper::LogLevel::LL_Error, "Cosine distance not currently supported when using quantization.\n");
        exit(1);
      }
      run_TPT_batch_multigpu<DTYPE, SUMTYPE, (int)DistMetric::Cosine>(dataSize, d_results, tptrees, d_tptrees, trees, levels, NUM_GPUS, KVAL, streams.data(), batch_min, batch_max, balanceFactor, d_pointset, dim, d_quantizer, index);
    }
    else {
      run_TPT_batch_multigpu<DTYPE, SUMTYPE, (int)DistMetric::L2>(dataSize, d_results, tptrees, d_tptrees, trees, levels, NUM_GPUS, KVAL, streams.data(), batch_min, batch_max, balanceFactor, d_pointset, dim, d_quantizer, index);
    }

    auto before_copy = std::chrono::high_resolution_clock::now();
    for(int gpuNum=0; gpuNum < NUM_GPUS; gpuNum++) {
      CUDA_CHECK(cudaMemcpy(&results[(GPUOffset[gpuNum]+batchOffset[gpuNum])*KVAL], d_results[gpuNum], curr_batch_size[gpuNum]*KVAL*sizeof(int), cudaMemcpyDeviceToHost));
    }
    auto after_copy = std::chrono::high_resolution_clock::now();
    
    double batch_copy_time = GET_CHRONO_TIME(before_copy, after_copy);
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "All GPUs finished batch - time to copy result:%.2lf\n", batch_copy_time);
    D2H_time += batch_copy_time;

    // Update all batchOffsets and check if done
    done=true;
    for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
      batchOffset[gpuNum] += curr_batch_size[gpuNum];
      if(batchOffset[gpuNum] < resPerGPU[gpuNum]) {
        done=false;
      }
    }
  } // Batches loop (while !done) 

auto end_t = std::chrono::high_resolution_clock::now();

  LOG(SPTAG::Helper::LogLevel::LL_Info, "Total times - prep time:%0.3lf, tree build:%0.3lf, neighbor compute:%0.3lf, Copy results:%0.3lf, Total runtime:%0.3lf\n", prep_time, tree_time, KNN_time, D2H_time, GET_CHRONO_TIME(start_t, end_t));

  for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
    tptrees[gpuNum]->destroy();
    delete tptrees[gpuNum];
    CUDA_CHECK(cudaFree(d_tptrees[gpuNum]));
    CUDA_CHECK(cudaFree(d_results[gpuNum]));
    CUDA_CHECK(cudaFree(d_data_raw[gpuNum]));
    CUDA_CHECK(cudaFree(d_pointset[gpuNum]));
  }
  delete[] tptrees;
  delete[] d_results;
  delete[] d_tptrees;
  delete[] d_data_raw;
  delete[] d_pointset;

  if(use_q) {
    CUDA_CHECK(cudaFree(d_quantizer));
    delete h_quantizer;
  }

}

/***************************************************************************************
 * Function called by SPTAG to create an initial graph on the GPU.  
 ***************************************************************************************/
template<typename T>
void buildGraph(SPTAG::VectorIndex* index, int m_iGraphSize, int m_iNeighborhoodSize, int trees, int* results, int refines, int refineDepth, int graph, int leafSize, int initSize, int NUM_GPUS, int balanceFactor) {

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
    LOG(SPTAG::Helper::LogLevel::LL_Error, "HeadNumGPUs parameter %d, but only %d devices available on system.  Exiting.\n", NUM_GPUS, numDevicesOnHost);
    exit(1);
  }
  LOG(SPTAG::Helper::LogLevel::LL_Info, "Building Head graph with %d GPUs...\n", NUM_GPUS);
  LOG(SPTAG::Helper::LogLevel::LL_Debug, "Total of %d GPU devices on system, using %d of them.\n", numDevicesOnHost, NUM_GPUS);

/**** Compute result batch sizes for each GPU ****/
  std::vector<size_t> batchSize(NUM_GPUS);
  std::vector<size_t> resPerGPU(NUM_GPUS);
  for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
    CUDA_CHECK(cudaSetDevice(gpuNum));

    resPerGPU[gpuNum] = dataSize / NUM_GPUS; // Results per GPU
    if(dataSize % NUM_GPUS > gpuNum) resPerGPU[gpuNum]++; 

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpuNum)); // Get avil. memory
    LOG(SPTAG::Helper::LogLevel::LL_Info, "GPU %d - %s\n", gpuNum, prop.name);

    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

    size_t rawDataSize, pointSetSize, treeSize, resMemAvail, maxEltsPerBatch;

    rawDataSize = rawSize*sizeof(T);
    pointSetSize = sizeof(PointSet<T>);
    treeSize = 20*dataSize;
    resMemAvail = (freeMem*0.9) - (rawDataSize+pointSetSize+treeSize); // Only use 90% of total memory to be safe
    maxEltsPerBatch = resMemAvail / (dim*sizeof(T) + KVAL*sizeof(int));

    batchSize[gpuNum] = std::min(maxEltsPerBatch, resPerGPU[gpuNum]);

    LOG(SPTAG::Helper::LogLevel::LL_Debug, "Memory for rawData:%lu MiB, pointSet structure:%lu MiB, Memory for TP trees:%lu MiB, Memory left for results:%lu MiB, total vectors:%lu, batch size:%d, total batches:%d\n", rawSize/1000000, pointSetSize/1000000, treeSize/1000000, resMemAvail/1000000, resPerGPU[gpuNum], batchSize[gpuNum], (((batchSize[gpuNum]-1)+resPerGPU[gpuNum]) / batchSize[gpuNum]));

  // If GPU memory is insufficient or so limited that we need so many batches it becomes inefficient, return error
    if(batchSize[gpuNum] == 0 || ((int)resPerGPU[gpuNum]) / batchSize[gpuNum] > 10000) {
      LOG(SPTAG::Helper::LogLevel::LL_Error, "Insufficient GPU memory to build Head index on GPU %d.  Available GPU memory:%lu MB, Points and tpt require:%lu MB, leaving a maximum batch size of %d results to be computed, which is too small to run efficiently.\n", gpuNum, (freeMem)/1000000, (rawDataSize+pointSetSize+treeSize)/1000000, maxEltsPerBatch);
      exit(1);
    }
  }
  std::vector<size_t> GPUOffset(NUM_GPUS);
  GPUOffset[0] = 0;
  LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU 0: results:%lu, offset:%lu\n", resPerGPU[0], GPUOffset[0]);
  for(int gpuNum=1; gpuNum < NUM_GPUS; ++gpuNum) {
    GPUOffset[gpuNum] = GPUOffset[gpuNum-1] + resPerGPU[gpuNum-1];
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU %d: results:%lu, offset:%lu\n", gpuNum, resPerGPU[gpuNum], GPUOffset[gpuNum]);
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
    LOG(SPTAG::Helper::LogLevel::LL_Error, "Selected datatype not currently supported.\n");
    exit(1);
  }
}


/****************** DEPRICATED CODE BELOW ***********************/

/*****************************************************************************************
 * Perform brute-force KNN on each leaf node, where only list of point ids is stored as leafs.
 * Returns for each point: the K nearest neighbors within the leaf node containing it.
 *****************************************************************************************/
/*
template<typename T, typename KEY_T, typename SUMTYPE, int Dim, int BLOCK_DIM>
__global__ void findKNN_leaf_nodes(Point<T,SUMTYPE,Dim>* data, TPtree<T,KEY_T,SUMTYPE,Dim>* tptree, int KVAL, int* results, int metric) {

  extern __shared__ char sharememory[];
  ThreadHeap<T, SUMTYPE, Dim, BLOCK_DIM> heapMem;

  heapMem.initialize(&((DistPair<SUMTYPE>*)sharememory)[(KVAL-1) * threadIdx.x], KVAL-1);

  Point<T,SUMTYPE,Dim> query;

  DistPair<SUMTYPE> max_K; // Stores largest of the K nearest neighbor of the query point
  bool dup; // Is the target already in the KNN list?

  DistPair<SUMTYPE> target;

  int blocks_per_leaf = gridDim.x / tptree->num_leaves;
  int threads_per_leaf = blocks_per_leaf*blockDim.x;
  int thread_id_in_leaf = blockIdx.x % blocks_per_leaf * blockDim.x + threadIdx.x;
  int leafIdx= blockIdx.x / blocks_per_leaf;
  long long int leaf_offset = tptree->leafs[leafIdx].offset;

  // Each point in the leaf is handled by a separate thread
  for(int i=thread_id_in_leaf; i<tptree->leafs[leafIdx].size; i+=threads_per_leaf) {
    query = data[tptree->leaf_points[leaf_offset + i]];

    heapMem.reset();

    // Load results from previous iterations into shared memory heap
    // and re-compute distances since they are not stored in result set
    heapMem.load_mem_sorted(data, &results[(long long int)query.id*KVAL], query, metric);

    max_K.idx = results[((long long int)query.id+1)*KVAL-1];
    if(max_K.idx == -1) {
      max_K.dist = INFTY<SUMTYPE>();
    }
    else {
      if(metric == 0) {
        max_K.dist = query.l2(&data[max_K.idx]);
      }
      else if(metric == 1) {
        max_K.dist = query.cosine(&data[max_K.idx]);
      }
    }

    // Compare source query with all points in the leaf
    for(long long int j=0; j<tptree->leafs[leafIdx].size; ++j) {
      if(j!=i) {
        if(metric == 0) {
          target.dist = query.l2(&data[tptree->leaf_points[leaf_offset+j]]);
        }
        else if (metric == 1) {
          target.dist = query.cosine(&data[tptree->leaf_points[leaf_offset+j]]);
        }
        if(target.dist < max_K.dist) {
          target.idx = tptree->leaf_points[leaf_offset+j];

          if(target.dist <= heapMem.top()) {
            dup=false;
            for(int dup_id=0; dup_id < KVAL-1; ++dup_id) {
              if(heapMem.vals[dup_id].idx == target.idx) {
                dup = true;
                dup_id=KVAL;
              }
            }
            if(!dup) { // Only consider it if not already in the KNN list
              max_K.dist = heapMem.vals[0].dist;
              max_K.idx = heapMem.vals[0].idx;
              heapMem.insert(target.dist, target.idx);
            }
          }
          else {
            max_K.dist = target.dist;
            max_K.idx = target.idx;
          }
        }
      }
    }
    // Write KNN to result list in sorted order
    results[((long long int)query.id+1)*KVAL-1] = max_K.idx;
    for(int j=KVAL-2; j>=0; j--) {
      results[(long long int)query.id*KVAL+j] = heapMem.vals[0].idx;
      heapMem.vals[0].dist=-1;
      heapMem.heapify();
    }
  }
}
*/

inline void updateKNNResults(std::vector< std::vector<SPTAG::SizeType> >& batch_truth, std::vector< std::vector<float> >& batch_dist, std::vector< std::vector<SPTAG::SizeType> >temp_truth, std::vector< std::vector<float> >temp_dist,
    int result_size, int K) {
    //LOG(SPTAG::Helper::LogLevel::LL_Info, "Entered updateFile.\n");
    std::vector< std::vector<SPTAG::SizeType> > result_truth = batch_truth;
    std::vector< std::vector<float> > result_dist = batch_dist;
    for (int i = 0; i < result_size; i++) {
        //For each result, we need to compare the 2K DistPairs and take top K. 
        int reader1 = 0;
        int reader2 = 0;
        for (int writer_ptr = 0; writer_ptr < K; writer_ptr++) {
            if (batch_dist[i][reader1] <= temp_dist[i][reader2]) {
                result_truth[i][writer_ptr] = batch_truth[i][reader1];
                result_dist[i][writer_ptr] = batch_dist[i][reader1++];
            }
            else {
                result_truth[i][writer_ptr] = temp_truth[i][reader2];
                result_dist[i][writer_ptr] = temp_dist[i][reader2++];
            }
        }
    }
    batch_truth = result_truth;
    batch_dist = result_dist;
}

template<typename DTYPE, int Dim, int BLOCK_DIM, typename SUMTYPE>
__global__ void query_KNN(Point<DTYPE, SUMTYPE, Dim>* querySet, Point<DTYPE, SUMTYPE, Dim>* data, int dataSize, int idx_offset, int numQueries, DistPair<SUMTYPE>* results, int KVAL, int metric) {
    extern __shared__ char sharememory[];
    __shared__ ThreadHeap<DTYPE, SUMTYPE, Dim, BLOCK_DIM> heapMem[BLOCK_DIM];
    DistPair<SUMTYPE> extra; // extra variable to store the largest distance/id for all KNN of the point
    //TransposePoint<DTYPE, Dim, BLOCK_DIM, SUMTYPE> query;  // Stores in strided memory to avoid bank conflicts
    //__shared__ DTYPE transpose_mem[Dim * BLOCK_DIM];
    Point<DTYPE, SUMTYPE, Dim> query;

    //if (cuda::std::is_same<DTYPE, uint8_t>::value || cuda::std::is_same<DTYPE, int8_t>::value) {
    //    query.setMem(&transpose_mem[threadIdx.x*4]);
    //}
    //else {
    //    query.setMem(&transpose_mem[threadIdx.x]);
    //}

    heapMem[threadIdx.x].initialize(&((DistPair<SUMTYPE>*)sharememory)[(KVAL - 1) * threadIdx.x], KVAL - 1);

    SUMTYPE dist;
    // Loop through all query points
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numQueries; i += blockDim.x * gridDim.x) {
        heapMem[threadIdx.x].reset();
        extra.dist = INFTY<SUMTYPE>();
        query=querySet[i]; 
        //query.loadPoint(querySet[i]); // Load into shared memory
        // Compare with all points in the dataset
        for (int j = 0; j < dataSize; j++) {
            if (metric == 0) {
                dist = query.l2(&data[j]);
            }
            else if (metric == 1){
                dist = query.cosine(&data[j]);
            }
            if (dist < extra.dist) {
                if (dist < heapMem[threadIdx.x].top()) {
                    extra.dist = heapMem[threadIdx.x].vals[0].dist;
                    extra.idx = heapMem[threadIdx.x].vals[0].idx + idx_offset;
                    //When recording the index, remember the off_set to discriminate different subvectorSets
                    heapMem[threadIdx.x].insert(dist, j + idx_offset);
                }
                else {
                    extra.dist = dist;
                    extra.idx = j + idx_offset;
                }
            }
        }
        // Write KNN to result list in sorted order
        results[(i + 1) * KVAL - 1].idx = extra.idx;
        results[(i + 1) * KVAL - 1].dist = extra.dist;
        for (int j = KVAL - 2; j >= 0; j--) {
            results[i * KVAL + j].idx = heapMem[threadIdx.x].vals[0].idx;
            results[i * KVAL + j].dist = heapMem[threadIdx.x].vals[0].dist;
            heapMem[threadIdx.x].vals[0].dist = -1;
            heapMem[threadIdx.x].heapify();

        }
    }

}

template <typename DTYPE, typename SUMTYPE, int MAX_DIM>
__host__ void GenerateTruthGPUCore(std::shared_ptr<VectorSet> querySet, std::shared_ptr<VectorSet> vectorSet, const std::string truthFile,
    const SPTAG::DistCalcMethod distMethod, const int K, const SPTAG::TruthFileType p_truthFileType, const std::shared_ptr<SPTAG::COMMON::IQuantizer>& quantizer,
    std::vector< std::vector<SPTAG::SizeType> >&truthset, std::vector< std::vector<float> >&distset) {
    int numDevicesOnHost;
    CUDA_CHECK(cudaGetDeviceCount(&numDevicesOnHost));
    int NUM_GPUS = numDevicesOnHost;
    int metric = distMethod == DistCalcMethod::Cosine ? 1 : 0;
    //const int NUM_GPUS = 4;

    std::vector< std::vector<SPTAG::SizeType> > temp_truth = truthset;
    std::vector< std::vector<float> > temp_dist = distset;

    int result_size = querySet->Count();
    int vector_size = vectorSet->Count();
    int dim = querySet->Dimension();
    auto vectors = (DTYPE*)vectorSet->GetData();
    int per_gpu_result = (result_size)*K;

    LOG_INFO("QueryDatatype: %s, Rows:%ld, Columns:%d\n", (STR(DTYPE)), result_size, dim);
    LOG_INFO("Datatype: %s, Rows:%ld, Columns:%d\n", (STR(DTYPE)), vector_size, dim);

    //Assign vectors to every GPU
    std::vector<size_t> vectorsPerGPU(NUM_GPUS);
    for (int gpuNum = 0; gpuNum < NUM_GPUS; ++gpuNum) {
        vectorsPerGPU[gpuNum] = vector_size / NUM_GPUS;
        if (vector_size % NUM_GPUS > gpuNum) vectorsPerGPU[gpuNum]++;
    }
    //record the offset in different GPU
    std::vector<size_t> GPUOffset(NUM_GPUS);
    GPUOffset[0] = 0;
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU 0: results:%lu, offset:%lu\n", vectorsPerGPU[0], GPUOffset[0]);
    for (int gpuNum = 1; gpuNum < NUM_GPUS; ++gpuNum) {
        GPUOffset[gpuNum] = GPUOffset[gpuNum - 1] + vectorsPerGPU[gpuNum - 1];
        LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU %d: results:%lu, offset:%lu\n", gpuNum, vectorsPerGPU[gpuNum], GPUOffset[gpuNum]);
    }

    std::vector<cudaStream_t> streams(NUM_GPUS);
    std::vector<size_t> batchSize(NUM_GPUS);

    cudaError_t resultErr;
    
    //KNN 
    Point<DTYPE, SUMTYPE, MAX_DIM>* points = convertMatrix < DTYPE, SUMTYPE, MAX_DIM >((DTYPE*)querySet->GetData(), result_size, dim);
    Point<DTYPE, SUMTYPE, MAX_DIM>** d_points = new Point<DTYPE, SUMTYPE, MAX_DIM>*[NUM_GPUS];
    Point<DTYPE, SUMTYPE, MAX_DIM>** d_check_points = new Point<DTYPE, SUMTYPE, MAX_DIM>*[NUM_GPUS];
    Point<DTYPE, SUMTYPE, MAX_DIM>** sub_vectors_points = new Point<DTYPE, SUMTYPE, MAX_DIM>*[NUM_GPUS];
    DistPair<SUMTYPE>** d_results = new DistPair<SUMTYPE>*[NUM_GPUS]; // malloc space for each gpu to save bf result

    //Point<DTYPE, SUMTYPE, MAX_DIM>** d_points = new Point<DTYPE, SUMTYPE, MAX_DIM>*[NUM_GPUS];
    //TPtree<DTYPE, KEYTYPE, SUMTYPE, MAX_DIM>** tptrees = new TPtree<DTYPE, KEYTYPE, SUMTYPE, MAX_DIM>*[NUM_GPUS];
    //TPtree<DTYPE, KEYTYPE, SUMTYPE, MAX_DIM>** d_tptrees = new TPtree<DTYPE, KEYTYPE, SUMTYPE, MAX_DIM>*[NUM_GPUS];
    //int** d_results = new int* [NUM_GPUS];
    // Calculate GPU memory usage on each device
    for (int gpuNum = 0; gpuNum < NUM_GPUS; gpuNum++) {
        CUDA_CHECK(cudaSetDevice(gpuNum));
        CUDA_CHECK(cudaStreamCreate(&streams[gpuNum]));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, gpuNum));

        LOG(SPTAG::Helper::LogLevel::LL_Info, "GPU %d - %s\n", gpuNum, prop.name);

        size_t freeMem, totalMem;
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

        size_t queryPointSize = querySet->Count() * sizeof(Point<DTYPE, SUMTYPE, MAX_DIM>);
        size_t resultSetSize = querySet->Count() * K * 4;//size(int) = 4
        size_t resMemAvail = (freeMem * 0.9) - (queryPointSize + resultSetSize); // Only use 90% of total memory to be safe
        size_t cpuSaveMem = resMemAvail / NUM_GPUS;//using 90% of multi-GPUs might be to much for CPU. 
        int maxEltsPerBatch = cpuSaveMem / (sizeof(Point<DTYPE, SUMTYPE, MAX_DIM>));
        batchSize[gpuNum] = min(maxEltsPerBatch, (int)(vectorsPerGPU[gpuNum]));
        // If GPU memory is insufficient or so limited that we need so many batches it becomes inefficient, return error
        if (batchSize[gpuNum] == 0 || ((int)vectorsPerGPU[gpuNum]) / batchSize[gpuNum] > 10000) {
            LOG(SPTAG::Helper::LogLevel::LL_Error, "Insufficient GPU memory to build Head index on GPU %d.  Available GPU memory:%lu MB, Points and resultsSet require: %lu MB, leaving a maximum batch size of %d results to be computed, which is too small to run efficiently.\n", gpuNum, (freeMem) / 1000000, (queryPointSize + resultSetSize) / 1000000, maxEltsPerBatch);
            exit(1);
        }

        size_t total_batches = ((batchSize[gpuNum] - 1) + vectorsPerGPU[gpuNum]) / batchSize[gpuNum];
        LOG(SPTAG::Helper::LogLevel::LL_Info, "Memory for  query Points vectors:%lu MiB, Memory for resultSet:%lu MiB, Memory left for vectors:%lu MiB, total vectors:%lu, batch size:%d, total batches:%lu\n", queryPointSize / 1000000, resultSetSize / 1000000, resMemAvail / 1000000, vectorsPerGPU[gpuNum], batchSize[gpuNum], total_batches);

    }

    int KNN_blocks = querySet->Count() / KNN_THREADS;
    std::vector<size_t> curr_batch_size(NUM_GPUS);
    std::vector<size_t> batchOffset(NUM_GPUS);
    for (int gpuNum = 0; gpuNum < NUM_GPUS; ++gpuNum) {
        batchOffset[gpuNum] = 0;
    }

    // Allocate space on gpu
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]); // Copy data over on a separate stream for each GPU
        // Device result memory on each GPU
        CUDA_CHECK(cudaMalloc(&d_results[i], per_gpu_result * sizeof(DistPair<SUMTYPE>)));
        
        //Copy Queryvectors
        CUDA_CHECK(cudaMalloc(&d_points[i], querySet->Count() * sizeof(Point<DTYPE, SUMTYPE, MAX_DIM>)));
        CUDA_CHECK(cudaMemcpyAsync(d_points[i], points, querySet->Count() * sizeof(Point<DTYPE, SUMTYPE, MAX_DIM>), cudaMemcpyHostToDevice, streams[i]));
        //Batchvectors
        CUDA_CHECK(cudaMalloc(&d_check_points[i], batchSize[i] * sizeof(Point<DTYPE, SUMTYPE, MAX_DIM>)));
    }
    
    LOG_INFO("Starting KNN Kernel timer\n");
    auto t1 = std::chrono::high_resolution_clock::now();
    bool done = false;
    bool update = false;
    //RN, The querys are copied, and the space for result are malloced. 
    // Need to copy the result, and copy back to host to update. 
    // The vectors split, malloc, copy neeed to be done in here.
    while (!done) { // Continue until all GPUs have completed all of their batches
        size_t freeMem, totalMem;
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
        LOG(SPTAG::Helper::LogLevel::LL_Debug, "Avaliable Memory out of %lu MiB total Memory for GPU 0 is %lu MiB\n", totalMem / 1000000, freeMem / 1000000);

    // Prep next batch for each GPU
        for (int gpuNum = 0; gpuNum < NUM_GPUS; ++gpuNum) {
            curr_batch_size[gpuNum] = batchSize[gpuNum];
            // Check if final batch is smaller than previous
            if (batchOffset[gpuNum] + batchSize[gpuNum] > vectorsPerGPU[gpuNum]) {
                curr_batch_size[gpuNum] = vectorsPerGPU[gpuNum] - batchOffset[gpuNum];
            }
            LOG(SPTAG::Helper::LogLevel::LL_Info, "GPU %d - starting batch with offset:%lu, with GPU offset:%lu, size:%lu, TailRows:%lld\n", gpuNum, batchOffset[gpuNum], GPUOffset[gpuNum], curr_batch_size[gpuNum], vectorsPerGPU[gpuNum] -batchOffset[gpuNum]);
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        for (int i = 0; i < NUM_GPUS; i++) {
            CUDA_CHECK(cudaSetDevice(i));
            
            size_t start = GPUOffset[i] + batchOffset[i];
            sub_vectors_points[i] = convertMatrix < DTYPE, SUMTYPE, MAX_DIM >(&vectors[start * dim], curr_batch_size[i], dim);
            CUDA_CHECK(cudaMemcpyAsync(d_check_points[i], sub_vectors_points[i], curr_batch_size[i] * sizeof(Point<DTYPE, SUMTYPE, MAX_DIM>), cudaMemcpyHostToDevice, streams[i]));
            
            // Perfrom brute-force KNN from the subsets assigned to the GPU for the querySets
            int KNN_blocks = (KNN_THREADS - 1 + querySet->Count()) / KNN_THREADS;
            size_t dynamicSharedMem = KNN_THREADS * sizeof(DistPair < SUMTYPE>) * (K - 1); 
            //64*9*8=4608 for K=10 , KNN_Threads = 64
            //64*99*8=50688 for K = 100, KNN_Threads = 64
            if (dynamicSharedMem > (1024 * 48)) {
                LOG(SPTAG::Helper::LogLevel::LL_Error, "Cannot Launch CUDA kernel on %d, because of using to much shared memory size, %zu\n", i, dynamicSharedMem);
                exit(1);
            }
            LOG(SPTAG::Helper::LogLevel::LL_Info, "Launching kernel on %d\n", i);
            LOG(SPTAG::Helper::LogLevel::LL_Info, "Launching Parameters: KNN_blocks = %d, KNN_Thread = %d , dynamicSharedMem = %d, \n", KNN_blocks, KNN_THREADS, dynamicSharedMem);

            query_KNN<DTYPE, MAX_DIM, KNN_THREADS, SUMTYPE> << <KNN_blocks, KNN_THREADS, dynamicSharedMem, streams[i]>> > (d_points[i], d_check_points[i], curr_batch_size[i], start, result_size, d_results[i], K, metric);
            cudaError_t c_ret = cudaGetLastError();
            LOG(SPTAG::Helper::LogLevel::LL_Debug, "Error: %s\n", cudaGetErrorString(c_ret));            
        }
        
        //Copy back to result
        for (int gpuNum = 0; gpuNum < NUM_GPUS; gpuNum++) {
            cudaSetDevice(gpuNum);
            cudaDeviceSynchronize();
        }
        DistPair<SUMTYPE>* results = (DistPair<SUMTYPE>*)malloc(per_gpu_result * sizeof(DistPair<SUMTYPE>) * NUM_GPUS);
        for (int gpuNum = 0; gpuNum < NUM_GPUS; gpuNum++) {
            cudaMemcpy(&results[(gpuNum * per_gpu_result)], d_results[gpuNum], per_gpu_result * sizeof(DistPair<SUMTYPE>), cudaMemcpyDeviceToHost);
        }

        // To combine the results, we need to compare value from different GPU 
        //Results [KNN from GPU0][KNN from GPU1][KNN from GPU2][KNN from GPU3]
        // Every block will be [result_size*K] [K|K|K|...|K]
        // The real KNN is selected from 4*KNN
        size_t* reader_ptrs = (size_t*)malloc(NUM_GPUS * sizeof(size_t));
        for (size_t i = 0; i < result_size; i++) {
            //For each result, there should be writer pointer, four reader pointers
            // reader ptr is assigned to i+gpu_idx*(result_size*K)/4
            for (size_t gpu_idx = 0; gpu_idx < NUM_GPUS; gpu_idx++) {
                reader_ptrs[gpu_idx] = i * K + gpu_idx * (result_size * K);
            }
            for (size_t writer_ptr = i * K; writer_ptr < (i + 1) * K; writer_ptr++) {
                int selected = 0;
                for (size_t gpu_idx = 1; gpu_idx < NUM_GPUS; gpu_idx++) {
                    //if the other gpu has shorter dist
                    if (results[reader_ptrs[selected]].dist > results[reader_ptrs[gpu_idx]].dist) {
                        selected = gpu_idx;
                    }
                }
                temp_truth[i][writer_ptr - (i * K)] = results[reader_ptrs[selected]].idx;
                temp_dist[i][writer_ptr - (i * K)] = results[reader_ptrs[selected]++].dist;
            }
        }
        LOG(SPTAG::Helper::LogLevel::LL_Info, "Collected all the result from GPUs.\n");
        //update results, just assign to truthset for first batch
        if (update) {
            updateKNNResults(truthset, distset, temp_truth, temp_dist, result_size, K);
        }
        else {
            truthset = temp_truth;
            distset = temp_dist;
            update = true;
        }
        LOG(SPTAG::Helper::LogLevel::LL_Info, "Updated batch result.\n");
        // Update all batchOffsets and check if done
        done = true;
        for (int gpuNum = 0; gpuNum < NUM_GPUS; ++gpuNum) {
            batchOffset[gpuNum] += curr_batch_size[gpuNum];
            if (batchOffset[gpuNum] < vectorsPerGPU[gpuNum]) {
                done = false;
            }
            //avoid memory leak
            free(sub_vectors_points[gpuNum]);
        }
        free(results);
    } // Batches loop (while !done) 

    auto t2 = std::chrono::high_resolution_clock::now();
    double gpuRunTime = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    double gpuRunTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    LOG_ALL("Total GPU time (sec): %ld.%ld\n", (int)gpuRunTime, (((int)gpuRunTimeMs)%1000));
}

template <typename T>
void GenerateTruthGPU(std::shared_ptr<VectorSet> querySet, std::shared_ptr<VectorSet> vectorSet, const std::string truthFile,
    const SPTAG::DistCalcMethod distMethod, const int K, const SPTAG::TruthFileType p_truthFileType, const std::shared_ptr<SPTAG::COMMON::IQuantizer>& quantizer,
    std::vector< std::vector<SPTAG::SizeType> > &truthset, std::vector< std::vector<float> > &distset) {
    using SUMTYPE = std::conditional_t<std::is_same<T, float>::value, float, int32_t>;
    int m_iFeatureDim = querySet->Dimension();
    if (typeid(T) == typeid(float)) {
        if (m_iFeatureDim <= 64) {
            GenerateTruthGPUCore<T, float, 64>(querySet, vectorSet, truthFile, distMethod, K, p_truthFileType, quantizer, truthset, distset);
        }
        else if (m_iFeatureDim <= 100) {
            GenerateTruthGPUCore<T, float, 100>(querySet, vectorSet, truthFile, distMethod, K, p_truthFileType, quantizer, truthset, distset);
        }
        else if (m_iFeatureDim <= 128) {
            GenerateTruthGPUCore<T, float, 128>(querySet, vectorSet, truthFile, distMethod, K, p_truthFileType, quantizer, truthset, distset);
        }
        else if (m_iFeatureDim <= 184) {
            GenerateTruthGPUCore<T, float, 184>(querySet, vectorSet, truthFile, distMethod, K, p_truthFileType, quantizer, truthset, distset);
        }
        else if (m_iFeatureDim <= 384) {
            GenerateTruthGPUCore<T, float, 384>(querySet, vectorSet, truthFile, distMethod, K, p_truthFileType, quantizer, truthset, distset);
        }
        else if (m_iFeatureDim <= 768) {
            GenerateTruthGPUCore<T, float, 768>(querySet, vectorSet, truthFile, distMethod, K, p_truthFileType, quantizer, truthset, distset);
        }
        //else if (m_iFeatureDim <= 1024) {
        //    GenerateTruthGPUCore<T, SUMTYPE, 1024>(querySet, vectorSet, truthFile, distMethod, K, p_truthFileType, quantizer, truthset, distset);
        //}
        //else if (m_iFeatureDim <= 2048) {
        //    GenerateTruthGPUCore<T, SUMTYPE, 2048>(querySet, vectorSet, truthFile, distMethod, K, p_truthFileType, quantizer, truthset, distset);
        //}
        //else if (m_iFeatureDim <= 4096) {
        //    GenerateTruthGPUCore<T, SUMTYPE, 4096>(querySet, vectorSet, truthFile, distMethod, K, p_truthFileType, quantizer, truthset, distset);
        //}
        else {
            LOG(SPTAG::Helper::LogLevel::LL_Error, "%d dimensions not currently supported for GPU generate Truth.\n");
            exit(1);
        }
    }
    else if (typeid(T) == typeid(uint8_t) || typeid(T) == typeid(int8_t)) {
        if (m_iFeatureDim <= 64) {
            GenerateTruthGPUCore<T, int32_t, 64>(querySet, vectorSet, truthFile, distMethod, K, p_truthFileType, quantizer, truthset, distset);
        }
        else if (m_iFeatureDim <= 100) {
            GenerateTruthGPUCore<T, int32_t, 100>(querySet, vectorSet, truthFile, distMethod, K, p_truthFileType, quantizer, truthset, distset);
        }
        else if (m_iFeatureDim <= 128) {
            GenerateTruthGPUCore<T, int32_t, 128>(querySet, vectorSet, truthFile, distMethod, K, p_truthFileType, quantizer, truthset, distset);
        }
        else if (m_iFeatureDim <= 184) {
            GenerateTruthGPUCore<T, int32_t, 184>(querySet, vectorSet, truthFile, distMethod, K, p_truthFileType, quantizer, truthset, distset);
        }
        else if (m_iFeatureDim <= 384) {
            GenerateTruthGPUCore<T, int32_t, 384>(querySet, vectorSet, truthFile, distMethod, K, p_truthFileType, quantizer, truthset, distset);
        }
        else if (m_iFeatureDim <= 768) {
            GenerateTruthGPUCore<T, int32_t, 768>(querySet, vectorSet, truthFile, distMethod, K, p_truthFileType, quantizer, truthset, distset);
        }
        else {
            LOG(SPTAG::Helper::LogLevel::LL_Error, "%d dimensions not currently supported for GPU generate Truth.\n");
            exit(1);
        }
    }
    else {
        LOG(SPTAG::Helper::LogLevel::LL_Error, "Selected datatype not currently supported.\n");
        exit(1);
    }
}
#define DefineVectorValueType(Name, Type) template void GenerateTruthGPU<Type>(std::shared_ptr<VectorSet> querySet, std::shared_ptr<VectorSet> vectorSet, const std::string truthFile, const SPTAG::DistCalcMethod distMethod, const int K, const SPTAG::TruthFileType p_truthFileType, const std::shared_ptr<SPTAG::COMMON::IQuantizer>& quantizer, std::vector< std::vector<SPTAG::SizeType> > &truthset, std::vector< std::vector<float> > &distset);
#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

#endif
