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


template<typename T, typename SUMTYPE>
__forceinline__ __device__ bool violatesRNG_PS(T* a, T* b, SUMTYPE dist, int dim) {
  SUMTYPE between;

  between = cosine(a, b, dim);

  return between <= dist;
}

/*
template<typename T, typename SUMTYPE, int Dim>
__device__ bool violatesRNG_PS2(T* a, T* b, SUMTYPE dist) {
  SUMTYPE between;

  if(metric == 0) {
//    between = data[closer.idx].dist(&data[farther.idx], metric);
  }
  else {
    between = data[closer.idx].cosine(&data[farther.idx]);
  }
  return between <= farther.dist;
}
*/


/*****************************************************************************************
 * Perform the brute-force graph construction on each leaf node, while STRICTLY maintaining RNG properties.  May end up with less than K neighbors per vector.
 *****************************************************************************************/
template<typename T, typename SUMTYPE>
__device__ void findRNG(PointSet<T>* ps, TPtree* tptree, int KVAL, int* results, DistMetric metric, size_t min_id, size_t max_id, T* query, DistPair<SUMTYPE>* threadList, int dim) {

  SUMTYPE max_dist = INFTY<SUMTYPE>();
  DistPair<SUMTYPE> temp;
  int read_id, write_id;

  for(int i=0; i<KVAL; i++) {
    threadList[i].dist = INFTY<SUMTYPE>();
  }

  size_t queryId;
  T* candidate_vec;

  DistPair<SUMTYPE> target;
  DistPair<SUMTYPE> candidate;

  int blocks_per_leaf = gridDim.x / tptree->num_leaves;
  int threads_per_leaf = blocks_per_leaf*blockDim.x;
  int thread_id_in_leaf = blockIdx.x % blocks_per_leaf * blockDim.x + threadIdx.x;
  int leafIdx= blockIdx.x / blocks_per_leaf;
  size_t leaf_offset = tptree->leafs[leafIdx].offset;

  bool good;

  T* temp_ptr;


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

          threadList[j].dist = ps->cosine(query, threadList[j].idx);
/*
          if(metric == 0) {
            threadList[j].dist = query.l2(&data[threadList[j].idx]);
	  }
          else {
            threadList[j].dist = query.cosine(&data[threadList[j].idx]);
	  }
*/
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
//          candidate.dist = ps->dist(query, candidate.idx, metric);
//          candidate.dist = ps->cosine(query, candidate.idx, (threadIdx.x==0 && blockIdx.x==0));
          candidate.dist = ps->cosine(query, candidate.idx);
/*
          if(metric == 0) {
            candidate.dist = query.l2(&data[candidate.idx]);
          }
          else if(metric == 1) {
            candidate.dist = query.cosine(&data[candidate.idx]);
          }
*/

          if(candidate.dist < max_dist){ // If it is a candidate to be added to neighbor list
            candidate_vec = ps->getVec(candidate.idx);
	    for(read_id=0; candidate.dist > threadList[read_id].dist && good; read_id++) {
              if(violatesRNG_PS<T, SUMTYPE>(candidate_vec, ps->getVec(threadList[read_id].idx),candidate.dist, ps->dim)) {
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
                if(!violatesRNG_PS<T, SUMTYPE>(ps->getVec(threadList[read_id].idx), candidate_vec, threadList[read_id].dist, ps->dim)) {
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


/*
template<typename DTYPE, typename SUMTYPE, int MAX_DIM, int QDIM>
void run_TPT_quantized(Point<uint8_t,float,QDIM>** d_qpoints, GPU_PQQuantizer* d_quantizer, Point<DTYPE,SUMTYPE,MAX_DIM>** d_points, size_t dataSize, int** d_results, TPtree<DTYPE,KEYTYPE,SUMTYPE,MAX_DIM>** tptrees, TPtree<DTYPE,KEYTYPE,SUMTYPE,MAX_DIM>** d_tptrees, int iters, int levels, int NUM_GPUS, int KVAL, cudaStream_t* streams, std::vector<size_t> batch_min, std::vector<size_t> batch_max, int balanceFactor, int metric)
{

  // Set num blocks for all GPU kernel calls
  int KNN_blocks= max(tptrees[0]->num_leaves, BLOCKS);

  for(int tree_id=0; tree_id < iters; ++tree_id) { // number of TPTs used to create approx. KNN graph

    auto before_tpt = std::chrono::high_resolution_clock::now(); // Start timer for TPT building

    // Reset and create each TPT
    for(int gpuNum=0; gpuNum < NUM_GPUS; gpuNum++) {
      CUDA_CHECK(cudaSetDevice(gpuNum));
      tptrees[gpuNum]->reset();
    }

//    create_tptree_quantized<DTYPE,KEYTYPE,SUMTYPE,MAX_DIM>(tptrees, d_points, dataSize, levels, NUM_GPUS, streams, balanceFactor);
    CUDA_CHECK(cudaDeviceSynchronize());

            // Copy TPTs to each GPU
    for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
      CUDA_CHECK(cudaSetDevice(gpuNum));
      CUDA_CHECK(cudaMemcpy(d_tptrees[gpuNum], tptrees[gpuNum], sizeof(TPtree<DTYPE,KEYTYPE,SUMTYPE,MAX_DIM>), cudaMemcpyHostToDevice));
    }
    auto after_tpt = std::chrono::high_resolution_clock::now();

    for(int gpuNum=0; gpuNum < NUM_GPUS; gpuNum++) {

      CUDA_CHECK(cudaSetDevice(gpuNum));
      // Compute the STRICT RNG for each leaf node
//      findRNG_quantized<DTYPE, KEYTYPE, SUMTYPE, MAX_DIM, THREADS><<<KNN_blocks,THREADS, sizeof(DistPair<float>) * (KVAL) * THREADS>>>(d_qpoints[gpuNum], d_tptrees[gpuNum], KVAL, d_results[gpuNum], metric, batch_min[gpuNum], batch_max[gpuNum]);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto after_work = std::chrono::high_resolution_clock::now();
 
    double loop_tpt_time = ((double)std::chrono::duration_cast<std::chrono::seconds>(after_tpt - before_tpt).count()) + (((double)std::chrono::duration_cast<std::chrono::milliseconds>(after_tpt - before_tpt).count())/1000);
    double loop_work_time = ((double)std::chrono::duration_cast<std::chrono::seconds>(after_work - after_tpt).count()) + (((double)std::chrono::duration_cast<std::chrono::milliseconds>(after_work - after_tpt).count())/1000);
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "All GPUs finished tree %d - tree build time:%.2lf, neighbor compute time:%.2lf\n", tree_id, loop_tpt_time, loop_work_time);

  } 

printf("TODO - implement quantized batch!\n");

}
*/

#define MAX_SHAPE 1024

#define RUN_KERNEL(size) \
  if(dim <= size) {\
    T query[size]; \
    findRNG(ps, tptree, KVAL, results, metric, min_id, max_id, query, threadList, dim); \
    return; \
  } \

template<typename T, typename SUMTYPE>
__global__ void findRNG_selector(PointSet<T>* ps, TPtree* tptree, int KVAL, int* results, DistMetric metric, size_t min_id, size_t max_id, int dim, T* dynamic_temp) {

  extern __shared__ char sharememory[];
  DistPair<SUMTYPE>* threadList = (&((DistPair<SUMTYPE>*)sharememory)[KVAL*threadIdx.x]);

  RUN_KERNEL(64);
  RUN_KERNEL(128);
  RUN_KERNEL(200);
  RUN_KERNEL(768);
  RUN_KERNEL(MAX_SHAPE);

  // If dimension is larger than any static option
  T* query = dynamic_temp;
  findRNG(ps, tptree, KVAL, results, metric, min_id, max_id, query, threadList, dim);
}


template<typename DTYPE, typename SUMTYPE>
void run_TPT_batch_multigpu(size_t dataSize, int** d_results, TPtree** tptrees, TPtree** d_tptrees, int iters, int levels, int NUM_GPUS, int KVAL, cudaStream_t* streams, std::vector<size_t> batch_min, std::vector<size_t> batch_max, int balanceFactor, int metric, PointSet<DTYPE>** d_pointset, int dim) 
{

  // temp global memory needed for very large dimsneions
  DTYPE* dynamic_temp;
  if(dim > MAX_SHAPE) {
    CUDA_CHECK(cudaMalloc(&dynamic_temp, dim*sizeof(DTYPE)));
  }

  // Set num blocks for all GPU kernel calls
  int KNN_blocks= max(tptrees[0]->num_leaves, BLOCKS);

  for(int tree_id=0; tree_id < iters; ++tree_id) { // number of TPTs used to create approx. KNN graph

    auto before_tpt = std::chrono::high_resolution_clock::now(); // Start timer for TPT building

    // Reset and create each TPT
    for(int gpuNum=0; gpuNum < NUM_GPUS; gpuNum++) {
      CUDA_CHECK(cudaSetDevice(gpuNum));
      tptrees[gpuNum]->reset();
    }
    create_tptree_multigpu<DTYPE>(tptrees, d_pointset, dataSize, levels, NUM_GPUS, streams, balanceFactor);
    CUDA_CHECK(cudaDeviceSynchronize());


            // Copy TPTs to each GPU
    for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
      CUDA_CHECK(cudaSetDevice(gpuNum));
      CUDA_CHECK(cudaMemcpy(d_tptrees[gpuNum], tptrees[gpuNum], sizeof(TPtree), cudaMemcpyHostToDevice));
    }

    auto after_tpt = std::chrono::high_resolution_clock::now();

    for(int gpuNum=0; gpuNum < NUM_GPUS; gpuNum++) {
      CUDA_CHECK(cudaSetDevice(gpuNum));
      // Compute the STRICT RNG for each leaf node
      findRNG_selector<DTYPE, SUMTYPE>
                    <<<KNN_blocks,THREADS, sizeof(DistPair<SUMTYPE>)*KVAL*THREADS>>>
                    (d_pointset[gpuNum], d_tptrees[gpuNum], KVAL, d_results[gpuNum], 
                    (DistMetric)metric, batch_min[gpuNum], batch_max[gpuNum], dim, dynamic_temp);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto after_work = std::chrono::high_resolution_clock::now();
 
    double loop_tpt_time = GET_CHRONO_TIME(before_tpt, after_tpt);
    double loop_work_time = GET_CHRONO_TIME(after_tpt, after_work);
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "All GPUs finished tree %d - tree build time:%.2lf, neighbor compute time:%.2lf\n", tree_id, loop_tpt_time, loop_work_time);

  } 
}


/** QUANTIZED GRAPH BUILD CODE 
  GPU_PQQuantizer* d_quantizer;  // Only use if quantizer is enabled
  Point<uint8_t,float,QDIM>** d_qpoints;
  Point<uint8_t, float, QDIM>* q_points;

    QuantizerType qType = COMMON::DistanceUtils::Quantizer->GetQuantizerType();

    if(qType ==  QuantizerType::PQQuantizer) {
      printf("Using PQ quantization\n");


      d_qpoints = new Point<uint8_t,float,QDIM>*[NUM_GPUS]; // Only use if using quantization



      copyPointsToDevice<uint8_t,float,QDIM>(

// DEBUG INFO
      int qdim = COMMON::DistanceUtils::Quantizer->GetNumSubvectors();
      printf("qdim:%d, QDIM:%d\n", qdim, QDIM);
      float* distanceTables = COMMON::DistanceUtils::Quantizer->GetCosineDistanceTables();
      for(int i=0; i<10; ++i) {
        printf("%f, ", distanceTables[i]);
      }
      printf("\n");


      d_quantizer = new GPU_PQQuantizer(DistMetric::Cosine);
      q_points = convertMatrix<uint8_t,float,QDIM>(index, dataSize, qdim);
      for(size_t i=0;  i<dataSize; i++) {
        q_points[i].id = i;
      }

      printf("Quantized vectors:\n");
      for(int i=0; i<10; i++) {
        for(int j=0; j<qdim; j++) {
          std::cout << static_cast<int16_t>(q_points[i].getVal(j)) << ", ";
        }
        std::cout << std::endl;
      }

    }


      // Auto-compute batch size based on available memory on the GPU
      dataPointSize = dataSize*sizeof(Point<uint8_t, float, QDIM>);
      treeSize = 20*dataSize;
      resMemAvail = (freeMem*0.9) - (dataPointSize+treeSize); // Only use 90% of total memory to be safe

      maxEltsPerBatch = resMemAvail / (sizeof(Point<uint8_t, float, QDIM>) + KVAL*sizeof(int));
      batchSize[gpuNum] = min(maxEltsPerBatch, (int)(resPerGPU[gpuNum]));

      LOG(SPTAG::Helper::LogLevel::LL_Debug, "Memory for quantized points:%lu MiB, Memory for TP trees:%lu MiB, Memory left for results and reconstructed vectors:%lu MiB, total vectors:%lu, tree batch size:%d, ANN batch size:%d, total batches:%d\n", dataPointSize/1000000, treeSize/1000000, resMemAvail/1000000, resPerGPU[gpuNum], treeBatchSize[gpuNum], batchSize[gpuNum], (((batchSize[gpuNum]-1)+resPerGPU[gpuNum]) / batchSize[gpuNum]));

      // Allocate memory on GPUs and copy points to each GPU
      LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU:%d - Alloc'ing QPoints on device: %zu bytes and initializing TPTree memory.\n", gpuNum, dataSize*sizeof(Point<uint8_t, float, QDIM>));
      CUDA_CHECK(cudaMalloc(&d_qpoints[gpuNum], dataSize*sizeof(Point<uint8_t, float, QDIM>)));
      CUDA_CHECK(cudaMemcpy(d_qpoints[gpuNum], q_points, dataSize*sizeof(Point<uint8_t, float, QDIM>), cudaMemcpyHostToDevice));

      run_TPT_quantized<DTYPE, SUMTYPE, MAX_DIM, QDIM>(d_qpoints, d_quantizer, d_points, dataSize, d_results, tptrees, d_tptrees, trees, levels, NUM_GPUS, KVAL, streams.data(), batch_min, batch_max, balanceFactor, metric);

*/

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
  DTYPE** d_data_raw = new DTYPE*[NUM_GPUS];
  TPtree** d_tptrees = new TPtree*[NUM_GPUS];
  PointSet<DTYPE>** d_pointset = new PointSet<DTYPE>*[NUM_GPUS];


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
    run_TPT_batch_multigpu<DTYPE, SUMTYPE>(dataSize, d_results, tptrees, d_tptrees, trees, levels, NUM_GPUS, KVAL, streams.data(), batch_min, batch_max, balanceFactor, metric, d_pointset, dim);

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

}

/***************************************************************************************
 * Function called by SPTAG to create an initial graph on the GPU.  
 ***************************************************************************************/
template<typename T>
void buildGraph(SPTAG::VectorIndex* index, int m_iGraphSize, int m_iNeighborhoodSize, int trees, int* results, int refines, int refineDepth, int graph, int leafSize, int initSize, int NUM_GPUS, int balanceFactor) {

std::cout << "T:" << typeid(T).name() << std::endl;

  int m_iFeatureDim = index->GetFeatureDim();
  int m_disttype = (int)index->GetDistCalcMethod();
  size_t dataSize = (size_t)m_iGraphSize;
  int KVAL = m_iNeighborhoodSize;
  int dim = index->GetFeatureDim();
  size_t rawSize = dataSize*dim;

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

  // Make sure that neighborhood size is a power of 2
  if(m_iNeighborhoodSize == 0 || (m_iNeighborhoodSize & (m_iNeighborhoodSize-1)) != 0) {
    LOG(SPTAG::Helper::LogLevel::LL_Error, "NeighborhoodSize (with scaling factor applied) is %d but must be a power of 2 for GPU construction.\n", m_iNeighborhoodSize);
    exit(1);
  }

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

/*
    if(index->m_pQuantizer) {
      rawDataSize = rawSize*sizeof(uint8_t);
      pointSetSize = sizeof(PointSet<uint8_t>);
      treeSize = 20*dataSize;
      resMemAvail = (freeMem*0.9) - (rawDataSize+pointSetSize+treeSize); // Only use 90% of total memory to be safe
      maxEltsPerBatch = resMemAvail / (dim*sizeof(uint8_t) + KVAL*sizeof(int));
    }
    else {
*/
      rawDataSize = rawSize*sizeof(T);
      pointSetSize = sizeof(PointSet<T>);
      treeSize = 20*dataSize;
      resMemAvail = (freeMem*0.9) - (rawDataSize+pointSetSize+treeSize); // Only use 90% of total memory to be safe
      maxEltsPerBatch = resMemAvail / (dim*sizeof(T) + KVAL*sizeof(int));
//    }

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

  if(typeid(T) == typeid(float)) {
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

/****************************************************************************************
 * Non-batched version of graph construction on GPU, only supports creating KNN or "loosely"
 * enforced RNG.
 *
 * Create either graph on the GPU, graph is saved into @results and is stored on the CPU
 * graphType: KNN=0, RNG=1
 * Note, vectors of MAX_DIM number dimensions are used, so an upper-bound must be determined
 * at compile time
 ***************************************************************************************/
/*
template<typename DTYPE, typename SUMTYPE, int MAX_DIM>
void buildGraphGPU(SPTAG::VectorIndex* index, int dataSize, int KVAL, int trees, int* results, int refines, int graphtype, int initSize, int refineDepth, int leafSize) {

  LOG(SPTAG::Helper::LogLevel::LL_Info, "Starting buildGraphGPU\n");

  int dim = index->GetFeatureDim();
  int metric = (int)index->GetDistCalcMethod();
  DTYPE* data = (DTYPE*)index->GetSample(0);

  // Number of levels set to have approximately 500 points per leaf
  int levels = (int)std::log2(dataSize/leafSize);

  int KNN_blocks; // number of threadblocks used

  Point<DTYPE,SUMTYPE,MAX_DIM>* points = convertMatrix<DTYPE,SUMTYPE,MAX_DIM>(index, dataSize, dim);

  for(int i=0;  i<dataSize; i++) {
    points[i].id = i;
  }


  Point<DTYPE, SUMTYPE, MAX_DIM>* d_points;
  LOG(SPTAG::Helper::LogLevel::LL_Debug, "Alloc'ing Points on device: %ld bytes.\n", dataSize*sizeof(Point<DTYPE,SUMTYPE,MAX_DIM>));
  CUDA_CHECK(cudaMalloc(&d_points, dataSize*sizeof(Point<DTYPE,SUMTYPE,MAX_DIM>)));

  LOG(SPTAG::Helper::LogLevel::LL_Debug, "Copying to device.\n");
  CUDA_CHECK(cudaMemcpy(d_points, points, dataSize*sizeof(Point<DTYPE,SUMTYPE,MAX_DIM>), cudaMemcpyHostToDevice));

  LOG(SPTAG::Helper::LogLevel::LL_Debug, "Alloc'ing TPtree memory\n");
  TPtree<DTYPE,KEYTYPE,SUMTYPE,MAX_DIM>* tptree;
  CUDA_CHECK(cudaMallocManaged(&tptree, sizeof(TPtree<DTYPE,KEYTYPE,SUMTYPE, MAX_DIM>)));
  tptree->initialize(dataSize, levels);
//  KNN_blocks= max(tptree->num_leaves, BLOCKS);
  KNN_blocks= tptree->num_leaves;

  LOG(SPTAG::Helper::LogLevel::LL_Debug, "Alloc'ing memory for results on device: %lld bytes.\n", (long long int)dataSize*KVAL*sizeof(int));
  int* d_results;
  CUDA_CHECK(cudaMalloc(&d_results, (long long int)dataSize*KVAL*sizeof(int)));
  // Initialize results to all -1 (special value that is set to distance INFTY)
  CUDA_CHECK(cudaMemset(d_results, -1, (long long int)dataSize*KVAL*sizeof(int)));

  CUDA_CHECK(cudaDeviceSynchronize());

//  srand(time(NULL)); // random number seed for TP tree random hyperplane partitions
  srand(1); // random number seed for TP tree random hyperplane partitions


  double tree_time=0.0;
  double KNN_time=0.0;
  double refine_time = 0.0;
//  struct timespec start, end;
  time_t start_t, end_t;


  for(int tree_id=0; tree_id < trees; ++tree_id) { // number of TPTs used to create approx. KNN graph
  CUDA_CHECK(cudaDeviceSynchronize());

    LOG(SPTAG::Helper::LogLevel::LL_Debug, "TPT iteartion %d - ", tree_id);
    start_t = clock();
   // Create TPT
    tptree->reset();
    create_tptree<DTYPE, KEYTYPE, SUMTYPE,MAX_DIM>(tptree, d_points, dataSize, levels, 0, dataSize);
  CUDA_CHECK(cudaDeviceSynchronize());

    end_t = clock();

    tree_time += (double)(end_t-start_t)/CLOCKS_PER_SEC;

    start_t = clock();
   // Compute the KNN for each leaf node

    if(graphtype == 0) {
      findKNN_leaf_nodes<DTYPE, KEYTYPE, SUMTYPE, MAX_DIM, THREADS><<<KNN_blocks,THREADS, sizeof(DistPair<SUMTYPE>) * (KVAL-1) * THREADS >>>(d_points, tptree, KVAL, d_results, metric);
    }
    else if(graphtype==1) {
      findRNG_leaf_nodes<DTYPE, KEYTYPE, SUMTYPE, MAX_DIM, THREADS><<<KNN_blocks,THREADS, sizeof(DistPair<SUMTYPE>) * (KVAL-1) * THREADS >>>(d_points, tptree, KVAL, d_results, metric);
    }
    else if(graphtype==2) {
      findRNG_strict<DTYPE, KEYTYPE, SUMTYPE, MAX_DIM, THREADS><<<KNN_blocks,THREADS, sizeof(DistPair<SUMTYPE>) * (KVAL) * THREADS >>>(d_points, tptree, KVAL, d_results, metric, 0, dataSize);
    }

  CUDA_CHECK(cudaDeviceSynchronize());
   

    //clock_gettime(CLOCK_MONOTONIC, &end);
    end_t = clock();

    KNN_time += (double)(end_t-start_t)/CLOCKS_PER_SEC;

    LOG(SPTAG::Helper::LogLevel::LL_Debug, "tree:%lfms, graph build:%lfms\n", tree_time, KNN_time);

  } // end TPT loop

  start_t = clock();

  if(refines > 0) { // Only call refinement if need to do at least 1 step
    refineGraphGPU<DTYPE, SUMTYPE, MAX_DIM>(index, d_points, d_results, dataSize, KVAL, initSize, refineDepth, refines, metric);
  }

  end_t = clock();
  refine_time += (double)(end_t-start_t)/CLOCKS_PER_SEC;


  LOG(SPTAG::Helper::LogLevel::LL_Debug, "%0.3lf, %0.3lf, %0.3lf, %0.3lf, ", tree_time, KNN_time, refine_time, tree_time+KNN_time+refine_time);
  CUDA_CHECK(cudaMemcpy(results, d_results, (long long int)dataSize*KVAL*sizeof(int), cudaMemcpyDeviceToHost));


  tptree->destroy();
  CUDA_CHECK(cudaFree(d_points));
  CUDA_CHECK(cudaFree(tptree));
  CUDA_CHECK(cudaFree(d_results));

}
*/
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

/*****************************************************************************************
 * Improve KNN graph using neighbors' neighbors lookup process
 * Significantly improves accuracy once the approximate KNN is created.
 *****************************************************************************************/
/*
template<typename T, typename SUMTYPE, int Dim, int BLOCK_DIM>
__global__ void neighbors_KNN(Point<T,SUMTYPE,Dim>* data, int* results, int N, int KVAL, int metric) {

  extern __shared__ char sharememory[];
  ThreadHeap<T, SUMTYPE, Dim, BLOCK_DIM> heapMem;

  heapMem.initialize(&((DistPair<SUMTYPE>*)sharememory)[(KVAL-1) * threadIdx.x], KVAL-1);

  Point<T,SUMTYPE,Dim> query;
  DistPair<SUMTYPE> max_K;

  int neighbor;

  for(int src_id=blockIdx.x*blockDim.x + threadIdx.x; src_id<N; src_id+= blockDim.x*gridDim.x) {
    query = data[src_id]; // Load query into registers

    heapMem.reset();
    // Load current result set into heap
    heapMem.load_mem_sorted(data, &results[src_id*KVAL], query, metric);

    max_K.dist = INFTY<SUMTYPE>();
    max_K.idx = results[(src_id+1)*KVAL-1];

    if(metric == 0) {
      if(max_K.idx != -1) {
        max_K.dist = query.l2(&data[max_K.idx]);
      }
    }
    else if (metric == 1) {
      if(max_K.idx != -1) {
        max_K.dist = query.cosine(&data[max_K.idx]);
      }
    }

    // check nearest neighbor first
    neighbor = max_K.idx;
    check_neighbors<T, SUMTYPE, Dim, BLOCK_DIM>(data, results, KVAL, &heapMem, &max_K, neighbor, src_id, &query, metric);

    for(int i=0; i<KVAL-1 && neighbor != -1; ++i) {
      neighbor = heapMem.vals[i].idx;
      if(neighbor != -1) {
        check_neighbors<T, SUMTYPE, Dim, BLOCK_DIM>(data, results, KVAL, &heapMem, &max_K, neighbor, src_id, &query, metric);
      }
    }

    results[(src_id+1)*KVAL-1] = max_K.idx;
    // Write KNN to result list in sorted order
    for(int j=KVAL-2; j>=0; j--) {
      results[src_id*KVAL+j] = heapMem.vals[0].idx;
      heapMem.vals[0].dist=-1;
      heapMem.heapify();
    }
  }
}
*/

/*****************************************************************************************
 * Compute the "accessibility score" of a given point (target) from a source (id).
 * This is computed as the number of neighbors that have an edge to the target.
 * Costly operation but useful to improve the accuracy of the graph by increasing the
 * connectivity.
 ****************************************************************************************/
/*
template<typename T>
__device__ int compute_accessibility(int* results, int id, int target, int KVAL) {
  int access=0;
  int* ptr;
  for(int i=0; i<KVAL; i++) {
    access += (results[id*KVAL+i] == target);
    if(results[id*KVAL+i] != -1) {
      ptr = &results[results[id*KVAL+i]*KVAL];
      for(int j=0; j<KVAL; j++) {
        access += (ptr[j] == target);
      }
    }
  }
  return access;
}

*/
/*****************************************************************************************
 * Perform the brute-force graph construction on each leaf node, but tries to maintain 
 * RNG properties when adding/removing vectors from the neighbor list.  Also only
 * inserts new vectors into the neighbor list when the "accessibility score" is 0 (i.e.,
 * it is not already accessible by any neighbors.
 *****************************************************************************************/
/*
template<typename T, typename KEY_T, typename SUMTYPE, int Dim, int BLOCK_DIM>
__global__ void findRNG_leaf_nodes(Point<T,SUMTYPE,Dim>* data, TPtree<T,KEY_T,SUMTYPE,Dim>* tptree, int KVAL, int* results, int metric) {

  extern __shared__ char sharememory[];
  ThreadHeap<T, SUMTYPE, Dim, BLOCK_DIM> heapMem; // Stored in registers but contains a pointer 

  // Assigns the memory pointers in the heap to shared memory
  heapMem.initialize(&((DistPair<SUMTYPE>*)sharememory)[(KVAL-1) * threadIdx.x], KVAL-1);

  SUMTYPE nearest_dist;
  Point<T,SUMTYPE,Dim> query;

  DistPair<SUMTYPE> max_K; // Stores largest of the K nearest neighbor of the query point
  bool dup; // Is the target already in the KNN list?

  DistPair<SUMTYPE> target;

  int blocks_per_leaf = gridDim.x / tptree->num_leaves;
  int threads_per_leaf = blocks_per_leaf*blockDim.x;
  int thread_id_in_leaf = blockIdx.x % blocks_per_leaf * blockDim.x + threadIdx.x;
  int leafIdx= blockIdx.x / blocks_per_leaf;
  long long int leaf_offset = tptree->leafs[leafIdx].offset;

  int write_id;
  SUMTYPE write_dist;

  // Each point in the leaf is handled by a separate thread
  for(int i=thread_id_in_leaf; i<tptree->leafs[leafIdx].size; i+=threads_per_leaf) {
    query = data[tptree->leaf_points[leaf_offset + i]];
    nearest_dist = INFTY<SUMTYPE>();

    heapMem.reset();

    // Load results from previous iterations into shared memory heap
    // and re-compute distances since they are not stored in result set
    heapMem.load_mem_sorted(data, &results[(long long int)query.id*KVAL], query, metric);

    for(int k=0; k<KVAL-1; k++) {
      if(heapMem.vals[k].dist < nearest_dist) {
        nearest_dist = heapMem.vals[k].dist;
      }
    }

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
        else if(metric == 1) {
          target.dist = query.cosine(&data[tptree->leaf_points[leaf_offset+j]]);
       }

        if(target.dist < max_K.dist){ // If it is to be added as a new near neighbor
          target.idx = tptree->leaf_points[leaf_offset+j];

          if(target.dist <= heapMem.top()) {
            dup=false;
            for(int dup_id=0; dup_id < KVAL-1; ++dup_id) {
              if(heapMem.vals[dup_id].idx == target.idx) {
                dup = true;
                dup_id=KVAL;
              }
            }
// Only consider it if not already in the KNN list and it is not already accessible
            if(!dup) {
              write_dist = INFTY<SUMTYPE>();
              write_id=0;

              if(metric == 0) { // L2
                for(int k=0; k<KVAL-1 && heapMem.vals[k].dist > target.dist; k++) {
                  if(heapMem.vals[k].idx != -1 && (data[target.idx].l2(&data[heapMem.vals[k].idx]) < write_dist)) {
                    write_id = k;
                    write_dist = data[target.idx].l2(&data[heapMem.vals[k].idx]);
                  }
                }
              }
              else if(metric == 1){ // Cosine
                for(int k=0; k<KVAL-1 && heapMem.vals[k].dist > target.dist; k++) {
                  if(heapMem.vals[k].idx != -1 && (data[target.idx].cosine(&data[heapMem.vals[k].idx]) < write_dist)) {
                    write_id = k;
                    write_dist = data[target.idx].cosine(&data[heapMem.vals[k].idx]);
                  }
                }
              }

              if(max_K.idx == -1) {
                max_K.dist = heapMem.vals[0].dist;
                max_K.idx = heapMem.vals[0].idx;
                heapMem.insert(target.dist, target.idx);
              }
              else {
                heapMem.insertAt(target.dist, target.idx, write_id); // Replace RNG violating vector
              }
              if(target.dist < nearest_dist)
                nearest_dist = target.dist;
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

/*****************************************************************************************
 * Compare distance of all neighbor's points to see if they are nearer neighbors
 * RNG properties when adding/removing vectors from the neighbor list.  Also only
 * inserts new vectors into the neighbor list when the "accessibility score" is 0 (i.e.,
 * it is not already accessible by any neighbors.
 *****************************************************************************************/
/*
template<typename T, typename SUMTYPE, int Dim, int BLOCK_DIM>
__device__ void check_neighbors_RNG(Point<T,SUMTYPE,Dim>* data, int* results, int KVAL, int* RNG_id, int* RNG_dist, long long int neighbor, long long int src_id, Point<T,SUMTYPE,Dim>* query, int metric) {
  DistPair<SUMTYPE> target;
  bool dup;
  int write_id;
  SUMTYPE write_dist;

  for(long long int j=0; j<KVAL; ++j) { // Check each neighbor of this neighbor
    target.idx = results[neighbor*KVAL + j];
    if(target.idx != src_id && target.idx != -1) { // Don't include the source itself
      dup=false;
      for(int dup_id=0; dup_id < KVAL && dup==false; ++dup_id) {
        dup = (RNG_id[dup_id] == target.idx);
      }
      target.dist = INFTY<SUMTYPE>();
      if(!dup) {
        if(metric == 0) {
          target.dist = query->l2(&data[target.idx]);
        }
        else if(metric == 1) {
          target.dist = query->cosine(&data[target.idx]);
        }
      }
      if(target.dist < RNG_dist[KVAL-1]) {

        if(RNG_dist[KVAL-1] != -1) {
          for(int k=0; target.dist > RNG_dist[k] && dup==false; k++) { // check to enforce RNG property
            if(metric == 0) {
              dup = (data[target.idx].l2(&data[RNG_id[k]]) < target.dist);
            }
            else if(metric == 1) {
              dup = (data[target.idx].cosine(&data[RNG_id[k]]) < target.dist);
            }
          }
        }
        if(!dup) { // If new point doesn't violate RNG property
            // Clean up RNG list

          write_id=KVAL-1;
          write_dist=INFTY<SUMTYPE>();
          if(RNG_id[KVAL-1] != -1) {
            if(metric == 0) {
              for(int k=KVAL-1; RNG_dist[k] > target.dist && k>=0; k--) { // Remove farthest neighbor that will now violate RNG
                if(RNG_id[k] != -1 && (write_dist > data[target.idx].l2(&data[RNG_id[k]]))) {
                  write_id = k;
                  write_dist = data[target.idx].l2(&data[RNG_id[k]]);
//              k=-1;
                }
              }
            }
            else if(metric == 1) {
              for(int k=KVAL-1; RNG_dist[k] > target.dist && k>=0; k--) { // Remove farthest neighbor that will now violate RNG
                if(RNG_id[k] != -1 && (write_dist > data[target.idx].cosine(&data[RNG_id[k]]))) {
                  write_id = k;
                  write_dist = data[target.idx].l2(&data[RNG_id[k]]);
                }
              }
            }
          }

          for(int k=write_id; k>=0; k--) {
            if(k==0 || RNG_dist[k-1] < target.dist) {
              RNG_dist[k] = target.dist;
              RNG_id[k] = target.idx;
              k = -1;
            }
            else {
              RNG_dist[k] = RNG_dist[k-1];
              RNG_id[k] = RNG_id[k-1];
            }
          }
        }
      }
    }
  }
}
*/

/************************************************************************
 * Refine graph by performing check_neighbors_RNG on every node in the graph
 ************************************************************************/
/*
template<typename T, typename SUMTYPE, int Dim, int BLOCK_DIM>
__global__ void neighbors_RNG(Point<T,SUMTYPE,Dim>* data, int* results, int N, int KVAL, int metric) {

  extern __shared__ char sharememory[];
  int* RNG_id = &((int*)sharememory)[2*KVAL*threadIdx.x];
  int* RNG_dist = &((int*)sharememory)[2*KVAL*threadIdx.x + KVAL];

  Point<T,SUMTYPE,Dim> query;
  int neighbor;

  for(long long int src_id=blockIdx.x*blockDim.x + threadIdx.x; src_id<N; src_id+= blockDim.x*gridDim.x) {
    query = data[src_id];

    // Load current result set into registers
    for(int i=0; i<KVAL; i++) {
      RNG_id[i] = results[src_id*KVAL + i];
      if(RNG_id[i] == -1) {
        RNG_dist[i] = INFTY<SUMTYPE>();
      }
      else {
        if(metric == 0) {
          RNG_dist[i] = query.l2(&data[RNG_id[i]]);
        }
        else if(metric == 1) {
          RNG_dist[i] = query.cosine(&data[RNG_id[i]]);
        }
      }
    }

    for(int i=0; i<KVAL && neighbor != -1; ++i) {
      neighbor = RNG_id[i];
      if(neighbor != -1) {
        check_neighbors_RNG<T, SUMTYPE, Dim, BLOCK_DIM>(data, results, KVAL, RNG_id, RNG_dist, neighbor, src_id, &query, metric);
      }
    }

    // Write KNN to result list in sorted order
    for(int i=0; i<KVAL; i++) {
      results[src_id*KVAL + i] = RNG_id[i];
    }
  }
}
*/
/*****************************************************************************************
 * For a given point, @src_id, looks at all neighbors' neighbors to refine KNN if any nearer
 * neighbors are found.  Recursively continues based on @DEPTH macro value.
 *****************************************************************************************/
/*
template<typename T, typename SUMTYPE, int Dim, int BLOCK_DIM>
__device__ void check_neighbors(Point<T,SUMTYPE,Dim>* data, int* results, int KVAL, ThreadHeap<T, SUMTYPE, Dim, BLOCK_DIM>* heapMem, DistPair<SUMTYPE>* max_K, int neighbor, int src_id, Point<T,SUMTYPE,Dim>* query, int metric) {

  DistPair<SUMTYPE> target;
  bool dup;

  for(long long int j=0; j<KVAL; ++j) { // Check each neighbor of this neighbor
    target.idx = results[neighbor*KVAL + j];
    if(target.idx != src_id && target.idx != -1) { // Don't include the source itself

      dup = (target.idx == max_K->idx);
      for(int dup_id=0; dup_id < KVAL-1 && dup==false; ++dup_id) {
        dup = (heapMem->vals[dup_id].idx == target.idx);
      }
      target.dist = INFTY<SUMTYPE>();
      if(!dup) {
        if(metric == 0) {
          target.dist = query->l2(&data[target.idx]);
        }
        else if(metric == 1) {
          target.dist = query->cosine(&data[target.idx]);
        }
      }
      if(target.dist < max_K->dist) {
        if(target.dist < heapMem->top()) {
          max_K->dist = heapMem->vals[0].dist;
          max_K->idx = heapMem->vals[0].idx;
          heapMem->insert(target.dist, target.idx);
        }
        else {
          max_K->dist = target.dist;
          max_K->idx = target.idx;
        }
      }
    }
  }
}

*/

#endif
