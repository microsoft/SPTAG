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
#include "ThreadHeap.hxx"
#include "TPtree.hxx"
#include "GPUQuantizer.hxx"


template<int Dim>
__device__ void findRNG_PQ(PointSet<uint8_t>* ps, TPtree* tptree, int KVAL, int* results, size_t min_id, size_t max_id, DistPair<float>* threadList, GPU_PQQuantizer* quantizer) {

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

          if(candidate.dist < max_dist){ // If it is a candidate to be added to neighbor list
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
__global__ void findRNG_selector(PointSet<T>* ps, TPtree* tptree, int KVAL, int* results, size_t min_id, size_t max_id, int dim, GPU_PQQuantizer* quantizer) {

  extern __shared__ char sharememory[];

  if(quantizer == NULL) {
//    RUN_KERNEL(64);
    RUN_KERNEL(100);
    RUN_KERNEL(200);
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
void run_TPT_batch_multigpu(size_t dataSize, int** d_results, TPtree** tptrees, TPtree** d_tptrees, int iters, int levels, int NUM_GPUS, int KVAL, cudaStream_t* streams, std::vector<size_t> batch_min, std::vector<size_t> batch_max, int balanceFactor, PointSet<DTYPE>** d_pointset, int dim, GPU_PQQuantizer* quantizer) 
{
  // Set num blocks for all GPU kernel calls
  int KNN_blocks= max(tptrees[0]->num_leaves, BLOCKS);

  for(int tree_id=0; tree_id < iters; ++tree_id) { // number of TPTs used to create approx. KNN graph

    auto before_tpt = std::chrono::high_resolution_clock::now(); // Start timer for TPT building

    create_tptree_multigpu<DTYPE>(tptrees, d_pointset, dataSize, levels, NUM_GPUS, streams, balanceFactor);
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
  GPU_PQQuantizer* d_quantizer = NULL;  // Only use if quantizer is enabled
  GPU_PQQuantizer* h_quantizer = NULL; 

  if(use_q) {
printf("Using quantizer, and metric:%d (L2?:%d)\n", (metric), (DistMetric)metric == DistMetric::L2);
    h_quantizer = new GPU_PQQuantizer(index->m_pQuantizer, (DistMetric)metric);
    CUDA_CHECK(cudaMalloc(&d_quantizer, sizeof(GPU_PQQuantizer)));
    CUDA_CHECK(cudaMemcpy(d_quantizer, h_quantizer, sizeof(GPU_PQQuantizer), cudaMemcpyHostToDevice));


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
      run_TPT_batch_multigpu<DTYPE, SUMTYPE, (int)DistMetric::Cosine>(dataSize, d_results, tptrees, d_tptrees, trees, levels, NUM_GPUS, KVAL, streams.data(), batch_min, batch_max, balanceFactor, d_pointset, dim, d_quantizer);
    }
    else {
      run_TPT_batch_multigpu<DTYPE, SUMTYPE, (int)DistMetric::L2>(dataSize, d_results, tptrees, d_tptrees, trees, levels, NUM_GPUS, KVAL, streams.data(), batch_min, batch_max, balanceFactor, d_pointset, dim, d_quantizer);
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
    cudaFree(d_tptrees[gpuNum]);
    cudaFree(d_results[gpuNum]);
  }
  delete[] tptrees;
  delete[] d_results;
  delete[] d_tptrees;
  delete[] d_data_raw;
  delete[] d_pointset;

  if(use_q) {
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

#endif
