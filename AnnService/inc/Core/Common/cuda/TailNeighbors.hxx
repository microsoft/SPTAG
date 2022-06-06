
#ifndef _SPTAG_COMMON_CUDA_TAILNEIGHBORS_H
#define _SPTAG_COMMON_CUDA_TAILNEIGHBORS_H

#include "Distance.hxx"
#include "KNN.hxx"

#include <cub/cub.cuh>
#include <chrono>
#include<thrust/execution_policy.h>
#include<thrust/sort.h>
//#include <parallel/algorithm>

using namespace SPTAG;
using namespace std;

// Structure used to re-order queries to improve thread divergence and access locality
class QueryGroup {
  public:
    int* sizes;
    int* offsets;
    int* query_ids;

  // Expects mem to already be allocated with N + 2*num_groups integers
  __device__ void init_mem(size_t N, int num_groups, int* mem) {
    sizes = mem;
    offsets = &mem[num_groups];
    query_ids = &mem[2*num_groups];
  }
};


/***********************************************************************************
 * Kernel to update the RNG list for each tail vector in the batch.  Assumes TPT 
 * is already created and populated based on the head vectors.
***********************************************************************************/
template<typename T, typename SUMTYPE, int Dim>
__device__ void findTailNeighbors(PointSet<T>* headPS, PointSet<T>* tailPS, TPtree* tptree, int KVAL, DistPair<SUMTYPE>* results, int metric, size_t numTails, int numHeads, QueryGroup* groups, DistPair<SUMTYPE>* threadList, SUMTYPE(*dist_comp)(T*,T*)) {

  T query[Dim];
  SUMTYPE max_dist = INFTY<SUMTYPE>();
  DistPair<SUMTYPE> temp;
  int read_id, write_id;

  for(int i=0; i<KVAL; i++) {
    threadList[i].dist = INFTY<SUMTYPE>();
    threadList[i].idx = -1;
  }

  DistPair<SUMTYPE> target;
  DistPair<SUMTYPE> candidate;
  int leafId;
  bool good;
  T* candidate_vec;

// If the queries were re-ordered, use the QueryGroup object to determine order to perform queries
// NOTE: Greatly improves locality and thread divergence

#if REORDER
  size_t tailId;
  for(size_t orderIdx = blockIdx.x*blockDim.x + threadIdx.x; orderIdx < numTails; orderIdx += gridDim.x*blockDim.x) {
    tailId = groups->query_ids[orderIdx];  
    for(int i=0; i<Dim; ++i) {
      query[i] = tailPS->getVec(tailId)[i];
    }
    leafId = searchForLeaf<T>(tptree, query);
#else
  for(size_t tailId = blockIdx.x*blockDim.x + threadIdx.x; tailId < numTails; tailId += gridDim.x*blockDim.x) {
    query = tailPS->getVec(tailId, true);
    leafId = searchForLeaf<T>(tptree, query);
#endif

    size_t leaf_offset = tptree->leafs[leafId].offset;

    // Load results from previous iterations into shared memory heap
    for(int j=0; j<KVAL; j++) {
      threadList[j] = results[(tailId*KVAL)+j];
    }
    max_dist = threadList[KVAL-1].dist;

    for(size_t j=0; j<tptree->leafs[leafId].size; ++j) {
      good = true;
      candidate.idx = tptree->leaf_points[leaf_offset+j];
      candidate_vec = headPS->getVec(candidate.idx);
      candidate.dist = dist_comp(query, candidate_vec);

      if(candidate.dist < max_dist && candidate.idx != tailId) { // If it is a candidate to be added to neighbor list

        for(read_id=0; candidate.dist > threadList[read_id].dist && good; read_id++) {
          if(violatesRNG<T, SUMTYPE>(candidate_vec, headPS->getVec(threadList[read_id].idx), candidate.dist, dist_comp)) {
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
            if(!violatesRNG<T, SUMTYPE>(candidate_vec, headPS->getVec(threadList[read_id].idx), candidate.dist, dist_comp)) {
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
    for(int j=0; j<KVAL; j++) {
      results[(size_t)(tailId*KVAL)+j] = threadList[j];
    }
  }
}

/***********************************************************************************
 * Kernel to update the RNG list for each tail vector in the batch.  Assumes TPT 
 * is already created and populated based on the head vectors.
 *          *** Uses Quantizer ***
***********************************************************************************/
template<int Dim>
__device__ void findTailNeighbors_PQ(PointSet<uint8_t>* headPS, PointSet<uint8_t>* tailPS, TPtree* tptree, int KVAL, DistPair<float>* results, size_t numTails, int numHeads, QueryGroup* groups, DistPair<float>* threadList, GPU_PQQuantizer* quantizer) {

  uint8_t query[Dim];
  float max_dist = INFTY<float>();
  DistPair<float> temp;
  int read_id, write_id;

  for(int i=0; i<KVAL; i++) {
    threadList[i].dist = INFTY<float>();
    threadList[i].idx = -1;
  }

  DistPair<float> target;
  DistPair<float> candidate;
  int leafId;
  bool good;
  uint8_t* candidate_vec;

// If the queries were re-ordered, use the QueryGroup object to determine order to perform queries
// NOTE: Greatly improves locality and thread divergence

#if REORDER
  size_t tailId;
  for(size_t orderIdx = blockIdx.x*blockDim.x + threadIdx.x; orderIdx < numTails; orderIdx += gridDim.x*blockDim.x) {
    tailId = groups->query_ids[orderIdx];  
    for(int i=0; i<Dim; ++i) {
      query[i] = tailPS->getVec(tailId)[i];
    }
    leafId = searchForLeaf<uint8_t>(tptree, query);
#else
  for(size_t tailId = blockIdx.x*blockDim.x + threadIdx.x; tailId < numTails; tailId += gridDim.x*blockDim.x) {
    query = tailPS->getVec(tailId, true);
    leafId = searchForLeaf<uint8_t>(tptree, query);
#endif

    size_t leaf_offset = tptree->leafs[leafId].offset;

    // Load results from previous iterations into shared memory heap
    for(int j=0; j<KVAL; j++) {
      threadList[j] = results[(tailId*KVAL)+j];
    }
    max_dist = threadList[KVAL-1].dist;

    for(size_t j=0; j<tptree->leafs[leafId].size; ++j) {
      good = true;
      candidate.idx = tptree->leaf_points[leaf_offset+j];
      candidate_vec = headPS->getVec(candidate.idx);
      candidate.dist = quantizer->dist(query, candidate_vec);

      if(candidate.dist < max_dist && candidate.idx != tailId) { // If it is a candidate to be added to neighbor list

        for(read_id=0; candidate.dist > threadList[read_id].dist && good; read_id++) {
          if(quantizer->violatesRNG(candidate_vec, headPS->getVec(threadList[read_id].idx), candidate.dist)) {
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
            if(!quantizer->violatesRNG(candidate_vec, headPS->getVec(threadList[read_id].idx), candidate.dist)) {
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
    for(int j=0; j<KVAL; j++) {
      results[(size_t)(tailId*KVAL)+j] = threadList[j];
    }
  }
}


__global__ void debug_warm_up_gpu() {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}


// Search each query into the TPTree to get the number of queries for each leaf node. This just generates a histogram
// for each leaf node so that we can compute offsets
template<typename T, typename SUMTYPE>
__global__ void compute_group_sizes(QueryGroup* groups, TPtree* tptree, PointSet<T>* queries, int N, int num_groups, int* queryMem, int dim) {
  groups->init_mem(N, num_groups, queryMem);

  extern __shared__ char sharememory[];
  T* query = (&((T*)sharememory)[threadIdx.x*dim]);
//  T query[MAX_DIM];
  int leafId;
  
  for(int qidx = blockDim.x*blockIdx.x + threadIdx.x; qidx < N; qidx += blockDim.x*gridDim.x) {
    for(int j=0; j<queries->dim;++j) {
      query[j] = queries->getVec(qidx)[j];
    }
    leafId = searchForLeaf<T>(tptree, query);
    atomicAdd(&(groups->sizes[leafId]), 1);   
  }
}

// Given the offsets and sizes of queries assigned to each leaf node, writes the list of queries assigned
// to each leaf.  The list of query_ids can then be used during neighborhood search to improve locality
template<typename T, typename SUMTYPE>
__global__ void assign_queries_to_group(QueryGroup* groups, TPtree* tptree, PointSet<T>* queries, int N, int num_groups, int dim) {

  extern __shared__ char sharememory[];
  T* query = (&((T*)sharememory)[threadIdx.x*dim]);

  int leafId;
  int idx_in_leaf;
  
  for(int qidx = blockDim.x*blockIdx.x + threadIdx.x; qidx < N; qidx += blockDim.x*gridDim.x) {
    for(int j=0; j<queries->dim;++j) {
      query[j] = queries->getVec(qidx)[j];
    }
    leafId = searchForLeaf<T>(tptree, query);
    idx_in_leaf = atomicAdd(&(groups->sizes[leafId]), 1);   
    groups->query_ids[groups->offsets[leafId]+idx_in_leaf] = qidx;
  }
}

// Gets the list of queries that are to be searched into each leaf.  The QueryGroup structure uses the memory
// provided with queryMem, which is required to be allocated with size N + 2*num_groups.  
template<typename T, typename SUMTYPE>
__host__ void get_query_groups(QueryGroup* groups, TPtree* tptree, PointSet<T>* queries, int N, int num_groups, int* queryMem, int NUM_BLOCKS, int NUM_THREADS, int dim) {

  CUDA_CHECK(cudaMemset(queryMem, 0, (N+2*num_groups)*sizeof(int)));

  compute_group_sizes<T,SUMTYPE><<<NUM_BLOCKS,NUM_THREADS, dim*NUM_THREADS*sizeof(T)>>>(groups, tptree, queries, N, num_groups, queryMem, dim);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Compute offsets based on sizes
  int* h_sizes = new int[num_groups];
  CUDA_CHECK(cudaMemcpy(h_sizes, queryMem, num_groups*sizeof(int), cudaMemcpyDeviceToHost));
  int* h_offsets = new int[num_groups];
  h_offsets[0]=0;
  for(int i=1; i<num_groups; ++i) h_offsets[i] = h_offsets[i-1] + h_sizes[i-1];

  CUDA_CHECK(cudaMemcpy(&queryMem[num_groups], h_offsets, num_groups*sizeof(int), cudaMemcpyHostToDevice));

  // Reset group sizes to use while assigning queires
  CUDA_CHECK(cudaMemset(queryMem, 0, num_groups*sizeof(int)));

  assign_queries_to_group<T,SUMTYPE><<<NUM_BLOCKS,NUM_THREADS, dim*NUM_THREADS*sizeof(T)>>>(groups, tptree, queries, N, num_groups, dim);
  CUDA_CHECK(cudaDeviceSynchronize());

  delete[] h_sizes;
  delete[] h_offsets;
}



#define MAX_SHAPE 1024

#define RUN_TAIL_KERNEL(size)                                     \
  if(dim <= size) {                                               \
    SUMTYPE (*dist_comp)(T*,T*);                                  \
    dist_comp = &dist<T,SUMTYPE,size,metric>;                     \
    findTailNeighbors<T,SUMTYPE,size>(headPS, tailPS, tptree,     \
                 KVAL, results, metric, curr_batch_size,          \
                 numHeads, groups, threadList, dist_comp); \
    return;                                                       \
  }                                                        

#define RUN_TAIL_KERNEL_PQ(size)                              \
  if(dim <= size) {                                        \
    findTailNeighbors_PQ<size>(headPS, tailPS, tptree,   \
                 KVAL, results, curr_batch_size,   \
                 numHeads, groups, threadList, quantizer);\
    return;                                                \
  }                                                        

template<typename T, typename SUMTYPE, int metric>
__global__ void findTailNeighbors_selector(PointSet<T>* headPS, PointSet<T>* tailPS, TPtree* tptree, int KVAL, DistPair<SUMTYPE>* results, size_t curr_batch_size, size_t numHeads, QueryGroup* groups, int dim) {

  extern __shared__ char sharememory[];
  DistPair<SUMTYPE>* threadList = (&((DistPair<SUMTYPE>*)sharememory)[KVAL*threadIdx.x]);
  
  RUN_TAIL_KERNEL(64)
  RUN_TAIL_KERNEL(100)
  RUN_TAIL_KERNEL(200)
  RUN_TAIL_KERNEL(768)
//  RUN_TAIL_KERNEL(MAX_SHAPE)

}

__global__ void findTailNeighbors_PQ_selector(PointSet<uint8_t>* headPS, PointSet<uint8_t>* tailPS, TPtree* tptree, int KVAL, DistPair<float>* results, size_t curr_batch_size, size_t numHeads, QueryGroup* groups, int dim, GPU_PQQuantizer* quantizer) {

  extern __shared__ char sharememory[];
  DistPair<float>* threadList = (&((DistPair<float>*)sharememory)[KVAL*threadIdx.x]);

  RUN_TAIL_KERNEL_PQ(50)
  RUN_TAIL_KERNEL_PQ(100)

}

    
#define COPY_BUFF_SIZE 100000

template<typename T>
void extractAndCopyHeadRaw_multi(T* dataBuffer, SPTAG::VectorIndex* headIndex, T** d_headRaw, size_t headRows, int dim, int NUM_GPUS) {
  T* vecPtr;
  size_t copy_size = COPY_BUFF_SIZE;
  for(size_t i=0; i<headRows; i+=COPY_BUFF_SIZE) {
    if(headRows-i < COPY_BUFF_SIZE) copy_size = headRows-i;

    for(int j=0; j<copy_size; ++j) {
      vecPtr = (T*)headIndex->GetSample(i+j);
      for(int k=0; k<dim; ++k) {
        dataBuffer[j*dim+k] = vecPtr[k];
      }
    }
    for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
      CUDA_CHECK(cudaSetDevice(gpuNum));
      CUDA_CHECK(cudaMemcpy((d_headRaw[gpuNum])+(i*dim), dataBuffer, copy_size*dim*sizeof(T), cudaMemcpyHostToDevice));
    }
  }
}


/***************
// TODO - Possible issue with ID of vectors in batch not having the offset!
***************/
template<typename T>
void extractAndCopyTailRaw_multi(T* dataBuffer, T* vectors, T** d_tailRaw, size_t tailRows, int dim, int NUM_GPUS) {
  T* vecPtr;
  size_t copy_size = COPY_BUFF_SIZE;
  size_t total_size=0;
  for(size_t i=0; i<tailRows; i+=COPY_BUFF_SIZE) {
    if(tailRows-i < COPY_BUFF_SIZE) copy_size = tailRows-i;

    vecPtr = &vectors[i*dim];

    for(int j=0; j<copy_size; ++j) {
      for(int k=0; k<dim; ++k) {
        dataBuffer[j*dim+k] = vecPtr[j*dim+k];
      }
    }

//    memcpy(dataBuffer, &vectors[i*dim], copy_size*dim);
    for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
      CUDA_CHECK(cudaSetDevice(gpuNum));
      CUDA_CHECK(cudaMemcpy((d_tailRaw[gpuNum])+(i*dim), dataBuffer, copy_size*dim*sizeof(T), cudaMemcpyHostToDevice));
    }
  }
}


template<typename T, typename SUMTYPE>
void getTailNeighborsTPT(T* vectors, SPTAG::SizeType N, SPTAG::VectorIndex* headIndex, std::unordered_set<int>& headVectorIDS, int dim, int RNG_SIZE, int numThreads, int NUM_TREES, int LEAF_SIZE, int metric, int NUM_GPUS, Edge* selections) {

    auto premem_t = std::chrono::high_resolution_clock::now();

    int numDevicesOnHost;
    CUDA_CHECK(cudaGetDeviceCount(&numDevicesOnHost));

    if(numDevicesOnHost < NUM_GPUS) {
      LOG(SPTAG::Helper::LogLevel::LL_Error, "NumGPUs parameter %d, but only %d devices available on system.  Exiting.\n", NUM_GPUS, numDevicesOnHost);
      exit(1);
    }

    LOG(SPTAG::Helper::LogLevel::LL_Info, "Building SSD index with %d GPUs...\n", NUM_GPUS);
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "Total of %d GPU devices on system, using %d of them.\n", numDevicesOnHost, NUM_GPUS);

/***** General variables *****/
    bool use_q = (headIndex->m_pQuantizer != NULL); // Using quantization?
    int resultErr;
    size_t headRows;
    headRows = headIndex->GetNumSamples();
    int TPTlevels = (int)std::log2(headRows/LEAF_SIZE);

    const int NUM_THREADS = 32;
    int NUM_BLOCKS;


/***** Host variables *****/
    // RawDataBuffer to shuttle raw data to GPU
    T* dataBuffer = new T[COPY_BUFF_SIZE*dim];
    PointSet<T> temp_ps; // Host structure to set before copying to GPU

    // Buffer to shuttle results back to CPU and save to selections structure
    DistPair<SUMTYPE>* results = new DistPair<SUMTYPE>[((size_t)COPY_BUFF_SIZE*RNG_SIZE)];

    std::vector<size_t> pointsPerGPU(NUM_GPUS);
    std::vector<size_t> GPUPointOffset(NUM_GPUS);
    for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
        pointsPerGPU[gpuNum] = N / NUM_GPUS;
        if(N % NUM_GPUS > gpuNum) pointsPerGPU[gpuNum]++;
    }
    GPUPointOffset[0] = 0;
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU 0: points:%lu, offset:%lu\n", pointsPerGPU[0], GPUPointOffset[0]);
    for(int gpuNum=1; gpuNum < NUM_GPUS; ++gpuNum) {
        GPUPointOffset[gpuNum] = GPUPointOffset[gpuNum-1] + pointsPerGPU[gpuNum-1];
        LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU %d: points:%lu, offset:%lu\n", gpuNum, pointsPerGPU[gpuNum], GPUPointOffset[gpuNum]);
    }
  
    // Streams and memory pointers for each GPU
    std::vector<cudaStream_t> streams(NUM_GPUS);
    std::vector<size_t> BATCH_SIZE(NUM_GPUS);

/***** Device variables *****/
    T** d_headRaw = new T*[NUM_GPUS];
    PointSet<T>** d_headPS = new PointSet<T>*[NUM_GPUS];
    T** d_tailRaw = new T*[NUM_GPUS];
    PointSet<T>** d_tailPS = new PointSet<T>*[NUM_GPUS];

    DistPair<SUMTYPE>** d_results = new DistPair<SUMTYPE>*[NUM_GPUS];
    TPtree** tptree = new TPtree*[NUM_GPUS];
    TPtree** d_tptree = new TPtree*[NUM_GPUS];

    // memory on each GPU for QuerySet reordering structures
    std::vector<int*> d_queryMem(NUM_GPUS);
    std::vector<QueryGroup*> d_queryGroups(NUM_GPUS);

    // Quantizer structures only used if quantization is enabled
    GPU_PQQuantizer* d_quantizer = NULL; 
    GPU_PQQuantizer* h_quantizer = NULL;

    if(use_q) {
      h_quantizer = new GPU_PQQuantizer(headIndex->m_pQuantizer, (DistMetric)metric);
      CUDA_CHECK(cudaMalloc(&d_quantizer, sizeof(GPU_PQQuantizer)));
      CUDA_CHECK(cudaMemcpy(d_quantizer, h_quantizer, sizeof(GPU_PQQuantizer), cudaMemcpyHostToDevice));
    }

    LOG(SPTAG::Helper::LogLevel::LL_Info, "Setting up each of the %d GPUs...\n", NUM_GPUS);

/*** Compute batch sizes and allocate all data on GPU ***/
    for(int gpuNum=0; gpuNum < NUM_GPUS; gpuNum++) {
        CUDA_CHECK(cudaSetDevice(gpuNum)); // Set current working GPU
        CUDA_CHECK(cudaStreamCreate(&streams[gpuNum])); // Create CDUA stream for each GPU

        debug_warm_up_gpu<<<1,32,0,streams[gpuNum]>>>();
        resultErr = cudaStreamSynchronize(streams[gpuNum]);
        LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU test/warmup complete - kernel status:%d\n", resultErr);

        // Get GPU info to compute batch sizes
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, gpuNum));
  
        LOG(SPTAG::Helper::LogLevel::LL_Info, "GPU %d - %s\n", gpuNum, prop.name);
        size_t freeMem, totalMem;
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
      
        // Auto-compute batch size based on available memory on the GPU
        size_t headVecSize = headRows*dim*sizeof(T) + sizeof(PointSet<T>);
        size_t randSize = (size_t)(min((int)headRows, (int)1024))*48; // Memory used for GPU random number generator
        size_t treeSize = 20*headRows + randSize;
        size_t tailMemAvail = (freeMem*0.9) - (headVecSize+treeSize); // Only use 90% of total memory to be safe

        // memory needed for raw data, pointset, results, and query reorder mem
        int maxEltsPerBatch = tailMemAvail / (dim*sizeof(T)+sizeof(PointSet<T>) + RNG_SIZE*sizeof(DistPair<SUMTYPE>) + 2*sizeof(int));
        BATCH_SIZE[gpuNum] = min(maxEltsPerBatch, (int)(pointsPerGPU[gpuNum]));

/*****************************************************
 * Batch size check and Debug information printing
 *****************************************************/
        if(BATCH_SIZE[gpuNum] == 0 || ((int)pointsPerGPU[gpuNum]) / BATCH_SIZE[gpuNum] > 10000) {
            LOG(SPTAG::Helper::LogLevel::LL_Error, "Insufficient GPU memory to build SSD index on GPU %d.  Available GPU memory:%lu MB, Head index requires:%lu MB, leaving a maximum batch size of %d elements, which is too small to run efficiently.\n", gpuNum, (freeMem)/1000000, (headVecSize+treeSize)/1000000, maxEltsPerBatch);
            exit(1);
        }
        LOG(SPTAG::Helper::LogLevel::LL_Info, "Memory for head vectors:%lu MiB, Memory for TP trees:%lu MiB, Memory left for tail vectors:%lu MiB, total tail vectors:%lu, batch size:%d, total batches:%d\n", headVecSize/1000000, treeSize/1000000, tailMemAvail/1000000, pointsPerGPU[gpuNum], BATCH_SIZE[gpuNum], (((BATCH_SIZE[gpuNum]-1)+pointsPerGPU[gpuNum]) / BATCH_SIZE[gpuNum]));
        LOG(SPTAG::Helper::LogLevel::LL_Debug, "Allocating GPU memory: tail points:%lu MiB, head points:%lu MiB, results:%lu MiB, TPT:%lu MiB, Total:%lu MiB\n", (BATCH_SIZE[gpuNum]*sizeof(T))/1000000, (headRows*sizeof(T))/1000000, (BATCH_SIZE[gpuNum]*RNG_SIZE*sizeof(DistPair<SUMTYPE>))/1000000, (sizeof(TPtree))/1000000, ((BATCH_SIZE[gpuNum]+headRows)*sizeof(T) + (BATCH_SIZE[gpuNum]*RNG_SIZE*sizeof(DistPair<SUMTYPE>)) + (sizeof(TPtree)))/1000000);

        // Allocate needed memory on the GPU
        CUDA_CHECK(cudaMalloc(&d_headRaw[gpuNum], headRows*dim*sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_headPS[gpuNum], sizeof(PointSet<T>)));
        CUDA_CHECK(cudaMalloc(&d_tailRaw[gpuNum], BATCH_SIZE[gpuNum]*dim*sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_tailPS[gpuNum], sizeof(PointSet<T>)));
        CUDA_CHECK(cudaMalloc(&d_results[gpuNum], BATCH_SIZE[gpuNum]*RNG_SIZE*sizeof(DistPair<SUMTYPE>)));

        // Prepare memory for TPTs
        CUDA_CHECK(cudaMalloc(&d_tptree[gpuNum], sizeof(TPtree)));
        tptree[gpuNum] = new TPtree;
        tptree[gpuNum]->initialize(headRows, TPTlevels, dim);

        LOG(SPTAG::Helper::LogLevel::LL_Debug, "tpt structure initialized for %lu head vectors, %d levels, leaf size:%d\n", headRows, TPTlevels, LEAF_SIZE);

        // Alloc memory for QuerySet structure
        CUDA_CHECK(cudaMalloc(&d_queryMem[gpuNum], BATCH_SIZE[gpuNum]*sizeof(int) + 2*tptree[gpuNum]->num_leaves*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_queryGroups[gpuNum], sizeof(QueryGroup)));

        // Copy head points to GPU

    } // End loop to set up memory on each GPU

    // Copy raw head vector data to each GPU
    extractAndCopyHeadRaw_multi<T>(dataBuffer, headIndex, d_headRaw, headRows, dim, NUM_GPUS);

    // Set and copy PointSet structure to each GPU
    temp_ps.dim = dim;
    for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
      temp_ps.data = d_headRaw[gpuNum];      
      CUDA_CHECK(cudaMemcpy(d_headPS[gpuNum], &temp_ps, sizeof(PointSet<T>), cudaMemcpyHostToDevice));
    }
    NUM_BLOCKS = min((int)(BATCH_SIZE[0]/NUM_THREADS), 10240);
    std::vector<size_t> curr_batch_size(NUM_GPUS);
    std::vector<size_t> offset(NUM_GPUS);
    for(int gpuNum=0; gpuNum<NUM_GPUS; ++gpuNum) {
        offset[gpuNum] = 0;
    }

    auto ssd_t1 = std::chrono::high_resolution_clock::now();


    bool done = false; 
    while(!done) { // Continue until all GPUs have completed all of their batches 

        // Prep next batch for each GPU
        for(int gpuNum=0; gpuNum<NUM_GPUS; ++gpuNum) {
            curr_batch_size[gpuNum] = BATCH_SIZE[gpuNum];
            // Check if final batch is smaller than previous
            if(offset[gpuNum]+BATCH_SIZE[gpuNum] > pointsPerGPU[gpuNum]) {
                curr_batch_size[gpuNum] = pointsPerGPU[gpuNum]-offset[gpuNum];
            }
            LOG(SPTAG::Helper::LogLevel::LL_Info, "GPU %d - starting batch with GPUOffset:%lu, offset:%lu, total offset:%lu, size:%lu, TailRows:%lld\n", gpuNum, GPUPointOffset[gpuNum], offset[gpuNum], GPUPointOffset[gpuNum]+offset[gpuNum], curr_batch_size[gpuNum], pointsPerGPU[gpuNum]);
    
            cudaSetDevice(gpuNum);

            // Copy next batch of tail vectors to corresponding GPU using many small copies to save memory
            extractAndCopyTailRaw_multi<T>(dataBuffer, &vectors[(GPUPointOffset[gpuNum]+offset[gpuNum])*dim], d_tailRaw, curr_batch_size[gpuNum], dim, NUM_GPUS);

            // Set tail pointset and copy to GPU
            temp_ps.data = d_tailRaw[gpuNum]; 
            CUDA_CHECK(cudaMemcpy(d_tailPS[gpuNum], &temp_ps, sizeof(PointSet<T>), cudaMemcpyHostToDevice));

            LOG(SPTAG::Helper::LogLevel::LL_Debug, "Copied %lu tail points to GPU - kernel status:%d\n", curr_batch_size[gpuNum], resultErr);

           int copy_size = COPY_BUFF_SIZE*RNG_SIZE;
           for(int i=0; i<curr_batch_size[gpuNum]*RNG_SIZE; i+=COPY_BUFF_SIZE*RNG_SIZE) {
                if(curr_batch_size[gpuNum]*RNG_SIZE - i < copy_size) copy_size = curr_batch_size[gpuNum]*RNG_SIZE - i;
                for(int j=0; j<copy_size; j++) {
                  results[j].idx=-1;
                  results[j].dist=INFTY<SUMTYPE>();
                }
                resultErr = cudaMemcpyAsync(d_results[gpuNum]+i, results, copy_size*sizeof(DistPair<SUMTYPE>), cudaMemcpyHostToDevice, streams[gpuNum]);
           }

            LOG(SPTAG::Helper::LogLevel::LL_Debug, "Copying initialized result list to GPU - batch size:%lu - replica count:%d, copy bytes:%lu, GPU offset:%lu, total result offset:%lu, - kernel status:%d\n", curr_batch_size[gpuNum], RNG_SIZE, curr_batch_size[gpuNum]*RNG_SIZE*sizeof(DistPair<SUMTYPE>), GPUPointOffset[gpuNum], (GPUPointOffset[gpuNum]+offset[gpuNum])*RNG_SIZE, resultErr);
        }

// OFFSET for batch of vector IDS: GPUPointOffset[gpuNum]+offset[gpuNum];
        // For each tree, create a new TPT and use it to refine tail neighbor list of batch
        for(int tree_id=0; tree_id < NUM_TREES; ++tree_id) {
            NUM_BLOCKS = min((int)(BATCH_SIZE[0]/NUM_THREADS), 10240);

auto t1 = std::chrono::high_resolution_clock::now();
            // Create TPT on each GPU
            create_tptree_multigpu<T>(tptree, d_headPS, headRows, TPTlevels, NUM_GPUS, streams.data(), 2);
            CUDA_CHECK(cudaDeviceSynchronize());
            LOG(SPTAG::Helper::LogLevel::LL_Debug, "TPT %d created on all GPUs\n", tree_id);

            // Copy TPTs to each GPU
            for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
                CUDA_CHECK(cudaSetDevice(gpuNum));
                CUDA_CHECK(cudaMemcpy(d_tptree[gpuNum], tptree[gpuNum], sizeof(TPtree), cudaMemcpyHostToDevice));
            }

auto t2 = std::chrono::high_resolution_clock::now();

#if REORDER
            // Compute QuerySet lists based on TPT, which can then be used to improve locality/divergence during RNG search
            for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
              CUDA_CHECK(cudaSetDevice(gpuNum));
              get_query_groups<T,SUMTYPE>(d_queryGroups[gpuNum], d_tptree[gpuNum], d_tailPS[gpuNum], (int)(curr_batch_size[gpuNum]), (int)tptree[gpuNum]->num_leaves, d_queryMem[gpuNum], NUM_BLOCKS, NUM_THREADS, dim);
            }
#endif

            for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
              CUDA_CHECK(cudaSetDevice(gpuNum));

              if(!use_q) {
                if(metric == (int)DistMetric::Cosine) {
                  findTailNeighbors_selector<T,SUMTYPE,(int)DistMetric::Cosine><<<NUM_BLOCKS, NUM_THREADS, sizeof(DistPair<SUMTYPE>)*RNG_SIZE*NUM_THREADS, streams[gpuNum]>>>(d_headPS[gpuNum], d_tailPS[gpuNum], d_tptree[gpuNum], RNG_SIZE, d_results[gpuNum], curr_batch_size[gpuNum], headRows, d_queryGroups[gpuNum], dim);
                }
                else {
                  findTailNeighbors_selector<T,SUMTYPE,(int)DistMetric::L2><<<NUM_BLOCKS, NUM_THREADS, sizeof(DistPair<SUMTYPE>)*RNG_SIZE*NUM_THREADS, streams[gpuNum]>>>(d_headPS[gpuNum], d_tailPS[gpuNum], d_tptree[gpuNum], RNG_SIZE, d_results[gpuNum], curr_batch_size[gpuNum], headRows, d_queryGroups[gpuNum], dim);
                }
              }
              else {
                findTailNeighbors_PQ_selector<<<NUM_BLOCKS, NUM_THREADS, sizeof(DistPair<float>)*RNG_SIZE*NUM_THREADS, streams[gpuNum]>>>((PointSet<uint8_t>*)d_headPS[gpuNum], (PointSet<uint8_t>*)d_tailPS[gpuNum], d_tptree[gpuNum], RNG_SIZE, (DistPair<float>*)d_results[gpuNum], curr_batch_size[gpuNum], headRows, d_queryGroups[gpuNum], dim, d_quantizer);
              }
            }

            CUDA_CHECK(cudaDeviceSynchronize());
            LOG(SPTAG::Helper::LogLevel::LL_Debug, "All GPUs finished finding neighbors of tails using TPT %d\n", tree_id);

auto t3 = std::chrono::high_resolution_clock::now();

LOG(SPTAG::Helper::LogLevel::LL_Debug, "Tree %d complete, time to build tree:%.2lf, time to compute tail neighbors:%.2lf\n", tree_id, GET_CHRONO_TIME(t1, t2), GET_CHRONO_TIME(t2, t3));

        } // TPT loop

        LOG(SPTAG::Helper::LogLevel::LL_Debug, "Batch complete on all GPUs, copying results back to CPU...\n");


        // Copy results of batch from each GPU to CPU result set
        for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
            CUDA_CHECK(cudaSetDevice(gpuNum));
LOG(SPTAG::Helper::LogLevel::LL_Debug, "gpu:%d, copying results size %lu to results offset:%d*%d=%d\n", gpuNum, curr_batch_size[gpuNum], GPUPointOffset[gpuNum]+offset[gpuNum],RNG_SIZE, (GPUPointOffset[gpuNum]+offset[gpuNum])*RNG_SIZE);

            size_t copy_size=COPY_BUFF_SIZE;
            for(int i=0; i<curr_batch_size[gpuNum]; i+=COPY_BUFF_SIZE) { 
                if(curr_batch_size[gpuNum] - i < copy_size) copy_size = curr_batch_size[gpuNum] - i;

                CUDA_CHECK(cudaMemcpy(results, d_results[gpuNum]+(i*RNG_SIZE), copy_size*RNG_SIZE*sizeof(DistPair<SUMTYPE>), cudaMemcpyDeviceToHost));

                size_t fullIdx = GPUPointOffset[gpuNum]+offset[gpuNum]+i;
                //#pragma omp parallel for
                for (size_t j = 0; j < copy_size; j++) {
                    size_t vecIdx = fullIdx+j;
                    if (headVectorIDS.count(vecIdx) == 0) {
                        size_t vecOffset = vecIdx * (size_t)RNG_SIZE;
                        size_t resOffset = j * (size_t)RNG_SIZE;
                        for (int resNum = 0; resNum < RNG_SIZE && results[resOffset + resNum].idx != -1; resNum++) {
                            selections[vecOffset + resNum].node = results[resOffset + resNum].idx;
                            selections[vecOffset + resNum].distance = (float)results[resOffset + resNum].dist;
                        }
                    }
                }
            }
        }

        LOG(SPTAG::Helper::LogLevel::LL_Debug, "Finished copying all batch results back to CPU\n");

        // Update all offsets and check if done
        done=true;
        for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
            offset[gpuNum] += curr_batch_size[gpuNum];
            if(offset[gpuNum] < pointsPerGPU[gpuNum]) {
                done=false;
            }
        }
    } // Batches loop (while !done)

    auto ssd_t2 = std::chrono::high_resolution_clock::now();


    LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU SSD build complete.  Freeing GPU memory...\n");

    for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
        CUDA_CHECK(cudaFree(d_headRaw[gpuNum]));
        CUDA_CHECK(cudaFree(d_tailRaw[gpuNum]));
        CUDA_CHECK(cudaFree(d_results[gpuNum]));
        CUDA_CHECK(cudaFree(d_tptree[gpuNum]));
        delete tptree[gpuNum];
    }
    delete dataBuffer;

    delete[] d_headRaw;
    delete[] d_tailRaw;
    delete[] d_results;
    delete[] tptree;
    delete[] d_tptree;
    delete[] results;

    auto ssd_t3 = std::chrono::high_resolution_clock::now();

    LOG(SPTAG::Helper::LogLevel::LL_Info, "Mam alloc time:%0.2lf, GPU time to build index:%.2lf, Memory free time:%.2lf\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(ssd_t1-premem_t).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(ssd_t1-premem_t).count())/1000, ((double)std::chrono::duration_cast<std::chrono::seconds>(ssd_t2-ssd_t1).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(ssd_t2-ssd_t1).count())/1000, ((double)std::chrono::duration_cast<std::chrono::seconds>(ssd_t3-ssd_t2).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(ssd_t3-ssd_t2).count())/1000);

}

__host__ __device__ struct GPUEdge
{
    SizeType node;
    float distance;
    SizeType tonode;
    __host__ __device__ GPUEdge() : node(MaxSize), distance(FLT_MAX/10.0), tonode(MaxSize) {}
};

struct GPU_EdgeCompare {
    bool operator()(const GPUEdge& a, int b) const
    {
        return a.node < b;
    };

    bool operator()(int a, const GPUEdge& b) const
    {
        return a < b.node;
    };

    __host__ __device__ bool operator()(const GPUEdge& a, const GPUEdge& b) {
        if (a.node == b.node)
        {
            if (a.distance == b.distance)
            {
                return a.tonode < b.tonode;
            }

            return a.distance < b.distance;
        }
        return a.node < b.node;
    }
} gpu_edgeComparer;


void GPU_SortSelections(std::vector<Edge>* selections) {

  size_t N = selections->size();

  size_t freeMem, totalMem;
  CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

// Maximum number of elements that can be sorted on GPU
  size_t sortBatchSize = (size_t)(freeMem*0.9 / sizeof(GPUEdge))/2; 

  std::vector<GPUEdge>* new_selections = reinterpret_cast<std::vector<GPUEdge>*>(selections);

  int num_batches = (N + (sortBatchSize-1)) / sortBatchSize;

  LOG(SPTAG::Helper::LogLevel::LL_Info, "Sorting final results. Size of result:%ld elements = %0.2lf GB, Available GPU memory:%0.2lf, sorting in %d batches\n", N, ((double)(N*sizeof(GPUEdge))/1000000000.0), ((double)freeMem)/1000000000.0, num_batches);

  int batchNum=0;

  GPUEdge* merge_mem;
  if(num_batches > 1) {
    merge_mem = new GPUEdge[N];
  }

  LOG(SPTAG::Helper::LogLevel::LL_Debug, "Allocating %ld bytes on GPU for sorting\n", sortBatchSize*sizeof(GPUEdge));
  GPUEdge* d_selections;
  CUDA_CHECK(cudaMalloc(&d_selections, sortBatchSize*sizeof(GPUEdge)));

  for(size_t startIdx = 0; startIdx < N; startIdx += sortBatchSize) {
    
    auto t1 = std::chrono::high_resolution_clock::now();

    size_t batchSize = sortBatchSize;
    if(startIdx + batchSize > N) {
      batchSize = N - startIdx;
    }
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "Sorting batch id:%ld, size:%ld\n", startIdx, batchSize);

    GPUEdge* batchPtr = &(new_selections->data()[startIdx]);
    
    CUDA_CHECK(cudaMemcpy(d_selections, batchPtr, batchSize*sizeof(GPUEdge), cudaMemcpyHostToDevice));
    try {
      thrust::sort(thrust::device, d_selections, d_selections+batchSize, gpu_edgeComparer);
    }
    catch (thrust::system_error &e){
      LOG(SPTAG::Helper::LogLevel::LL_Info, "Error: %s \n",e.what());
    }

    CUDA_CHECK(cudaMemcpy(batchPtr, d_selections, batchSize*sizeof(GPUEdge), cudaMemcpyDeviceToHost));

    auto t2 = std::chrono::high_resolution_clock::now();

    // For all batches after the first, merge into the final output
    if(startIdx > 0) {
      std::merge(new_selections->data(), batchPtr, batchPtr, &batchPtr[batchSize], merge_mem, gpu_edgeComparer);

// For faster merging on Linux systems, can use below instead of std::merge (above)
//      __gnu_parallel::merge(new_selections->data(), batchPtr, batchPtr, &batchPtr[batchSize], merge_mem, gpu_edgeComparer);
 
      memcpy(new_selections->data(), merge_mem, (startIdx+batchSize)*sizeof(GPUEdge));
    }
    
    auto t3 = std::chrono::high_resolution_clock::now();

    LOG(SPTAG::Helper::LogLevel::LL_Debug, "Sort batch %d - GPU transfer/sort time:%0.2lf, CPU merge time:%.2lf\n", batchNum, ((double)std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count())/1000, ((double)std::chrono::duration_cast<std::chrono::seconds>(t3-t2).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count())/1000);
    batchNum++;
  }
  if(num_batches > 1) {
    delete merge_mem;
  }
}



/*************************************************************************************************
 * Deprecated code from Hybrid CPU/GPU SSD Index builder
 *************************************************************************************************/

/*
// Function to extract head vectors from list and copy them to GPU, using only a small CPU buffer.
// This reduces the CPU memory usage needed to convert all vectors into the Point structure used by GPU
template<typename T, typename SUMTYPE, int MAX_DIM>
void extractAndCopyHeadPoints(Point<T,SUMTYPE,MAX_DIM>* headPointBuffer, T* vectors, SPTAG::VectorIndex* headIndex, Point<T,SUMTYPE,MAX_DIM>* d_headPoints, size_t headRows, int dim) {

  size_t copy_size = COPY_BUFF_SIZE;
  for(size_t i=0; i<headRows; i+=COPY_BUFF_SIZE) {
    if(headRows-i < COPY_BUFF_SIZE) copy_size = headRows-i; // Last copy may be smaller

    for(int j=0; j<copy_size; j++) {
      headPointBuffer[j].loadChunk((T*)headIndex->GetSample(i+j), dim);
      headPointBuffer[j].id = i+j;
    }
    
    CUDA_CHECK(cudaMemcpy(d_headPoints+i, headPointBuffer, copy_size*sizeof(Point<T,SUMTYPE,MAX_DIM>), cudaMemcpyHostToDevice));
  }
}

// Extracts tail vectors from vector list and copy them to GPU, using only a small CPU buffer for conversion.
// Returns number of tail vectors copied to GPU
template<typename T, typename SUMTYPE, int MAX_DIM>
size_t extractAndCopyTailPoints(Point<T,SUMTYPE,MAX_DIM>* pointBuffer, T* vectors, Point<T,SUMTYPE,MAX_DIM>* d_tailPoints, size_t size, std::unordered_set<int>& headVectorIDS, int dim, size_t batch_offset) {

  size_t copy_size = COPY_BUFF_SIZE;
  size_t write_idx=0;
  int tailIdx;

  for(size_t i=0; i<size; i+=COPY_BUFF_SIZE) {
    if(size-i < COPY_BUFF_SIZE) copy_size = size-i; // Last copy may be smaller
     
      for(size_t j=0; j<copy_size; ++j) {
        pointBuffer[j].loadChunk(&vectors[(i+j)*dim], dim);
        pointBuffer[j].id = i+j+batch_offset;
      }
      CUDA_CHECK(cudaMemcpy(d_tailPoints+i, pointBuffer, copy_size*sizeof(Point<T,SUMTYPE,MAX_DIM>), cudaMemcpyHostToDevice));
      write_idx+=copy_size;
  }
  return write_idx;
}
*/

/*
template <typename SUMTYPE>
class CandidateElt {
  public:
  int id;
  bool checked;

  __device__ CandidateElt& operator=(const CandidateElt& other) {
    id = other.id;
    checked = other.checked;
    return *this;
  }
};


template<typename T, typename SUMTYPE, int MAX_DIM, int NEAR_SIZE, int FAR_SIZE>
__global__ void compressToRNG(Point<T,SUMTYPE,MAX_DIM>* tailPoints, Point<T,SUMTYPE,MAX_DIM>* headPoints, CandidateElt<SUMTYPE>* near, CandidateElt<SUMTYPE>* far, DistPair<SUMTYPE>* results, int resultsPerVector, size_t batch_size, int metric) {

    Point<T,SUMTYPE,MAX_DIM> tail;
    int resultIdx, candidateIdx;

    extern __shared__ DistPair<SUMTYPE> RNGlist[];
    DistPair<SUMTYPE>* threadList = &RNGlist[threadIdx.x*resultsPerVector];
    bool good;

    for(size_t i=blockIdx.x*blockDim.x + threadIdx.x; i<batch_size; i+=gridDim.x*blockDim.x) {
        for(int j=0; j<resultsPerVector; j++) {
            threadList[j].idx=-1;
            threadList[j].dist=INFTY<SUMTYPE>();
        }

        if(near[NEAR_SIZE*i].id > -1) {  // TODO - see if can remove unnecessary checks like this?
            tail = tailPoints[i];
            threadList[0].idx = near[i*NEAR_SIZE].id;
            threadList[0].dist = tail.dist(&headPoints[threadList[0].idx], metric);

            resultIdx=1;
            candidateIdx=1;

            // First go through near list
            while(resultIdx < resultsPerVector && candidateIdx < NEAR_SIZE) {
                threadList[resultIdx].idx = near[i*NEAR_SIZE + candidateIdx].id;

                if(threadList[resultIdx].idx <= -1) {
                    candidateIdx++;
                    continue;
                }

                threadList[resultIdx].dist = tail.dist(&headPoints[threadList[resultIdx].idx], metric);
                good=true;
                for(int j=0; j<resultIdx; j++) {

                    if(violatesRNG(headPoints, threadList[resultIdx], threadList[j], metric)) {
                        good=false;
                        j=resultIdx;
                    }

                }
                if(good) {
                    resultIdx++;
                }
                candidateIdx++;
            }
            candidateIdx=0;
            // Then far list if needed
            while(resultIdx < resultsPerVector && candidateIdx < FAR_SIZE) {
                threadList[resultIdx].idx = far[i*FAR_SIZE + candidateIdx].id;

                if(threadList[resultIdx].idx <= -1) {
                    candidateIdx++;
                    continue;
                }

                threadList[resultIdx].dist = tail.dist(&headPoints[threadList[resultIdx].idx], metric);
                good=true;
                for(int j=0; j<resultIdx; j++) {

                    if(violatesRNG(headPoints, threadList[resultIdx], threadList[j], metric)) {
                        good=false;
                        j=resultIdx;
                    }

                }
                if(good) {
                    resultIdx++;
                }
                candidateIdx++;
            }
        }
        for(size_t j=0; j<resultsPerVector; j++) {
            results[i*resultsPerVector + j] = threadList[j];
        }
    }
}

template<typename SUMTYPE>
__device__ void markDuplicates(CandidateElt<SUMTYPE>* near, int NEAR_SIZE, CandidateElt<SUMTYPE>* far, int FAR_SIZE) {
    for(int i=threadIdx.x+1; i<NEAR_SIZE+FAR_SIZE; i+=blockDim.x) {
        if(i<NEAR_SIZE) {
            if(near[i].id == near[i-1].id) {
                near[i].id=-1;
                near[i].checked=true;
            }
        }
        else if(i==NEAR_SIZE) {
            if(far[0].id == near[NEAR_SIZE-1].id) {
                far[0].id = -1;
            }
        }
        else {
            if(far[i].id == far[i-1].id) {
                far[i].id=-1;
            }
        }
    }
}

template<typename T, typename SUMTYPE, int MAX_DIM, int NEAR_SIZE, int FAR_SIZE, int SORT_THREADS>
__global__ void sortCandidates(Point<T,SUMTYPE,MAX_DIM>* tailPoints, Point<T,SUMTYPE,MAX_DIM>* headPoints, CandidateElt<SUMTYPE>* near, CandidateElt<SUMTYPE>* far, int* nearest, size_t batch_size, int metric, int depth) {
    const int SORT_SIZE = NEAR_SIZE+FAR_SIZE;
    SUMTYPE dist[SORT_SIZE/SORT_THREADS];
    CandidateElt<SUMTYPE> sortVal[SORT_SIZE/SORT_THREADS];
    const int numNearest = FAR_SIZE/32;

    __shared__ bool print_debug;
    print_debug=false;

    typedef cub::BlockRadixSort<SUMTYPE, SORT_SIZE, SORT_SIZE/SORT_THREADS, CandidateElt<SUMTYPE>> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;
    Point<T,SUMTYPE,MAX_DIM> query;

    for(size_t i=blockIdx.x; i<batch_size; i+=gridDim.x) {
        query = tailPoints[i];
        if(query.id < 0) printf("query id:%d, i:%d\n", query.id, i);
        __syncthreads();

        for(int j=0; j<SORT_SIZE/SORT_THREADS; j++) {
            int readIdx = threadIdx.x + SORT_THREADS*j;
            if(readIdx < NEAR_SIZE) { // Fill registers with all values from NEAR and FAR lists
                sortVal[j] = near[i*NEAR_SIZE + readIdx];
            }
            else {
                bool dup=false;
                for(int k=0; k<NEAR_SIZE; k++) {
                    if(near[i*NEAR_SIZE+k].id == far[i*FAR_SIZE + (readIdx - NEAR_SIZE)].id) {
                        sortVal[j].id=-1;
                        sortVal[j].checked=true;
                        dup=true; 
                        break;
                    }
                }
                if(!dup) 
                    sortVal[j] = far[i*FAR_SIZE + (readIdx - NEAR_SIZE)];
            }
        }

            // Compute distances for all points to sort
        for(int j=0; j<SORT_SIZE/SORT_THREADS; j++) {
            if(sortVal[j].id <= -1) {
                dist[j] = INFTY<SUMTYPE>();
            }
            else {
                dist[j] = query.dist(&headPoints[sortVal[j].id], metric);
            }
        }

        __syncthreads();
        BlockRadixSort(temp_storage).Sort(dist, sortVal);
        __syncthreads();
        // Place sorted values back into near and far lists (closest in near)
        for(int j=0; j<SORT_SIZE/SORT_THREADS; j++) {
            int readIdx = threadIdx.x + SORT_THREADS*j;
            if(readIdx < NEAR_SIZE) { // Fill registers with all values from NEAR and FAR lists
                near[i*NEAR_SIZE + readIdx] = sortVal[j];
            }
            else {
                far[i*FAR_SIZE + (readIdx - NEAR_SIZE)] = sortVal[j];
            }
        }

//        __syncthreads();
//        markDuplicates<SUMTYPE>(&near[i*NEAR_SIZE], NEAR_SIZE, &far[i*FAR_SIZE], FAR_SIZE);
        __syncthreads();

        if(threadIdx.x==0) { // Set nearest to the first non-checked element
            int nearIdx=0;
            for(int j=0; j<numNearest; j++) {
                nearest[i*numNearest + j]=-1;
                for(; nearIdx<NEAR_SIZE && near[i*NEAR_SIZE + nearIdx].checked; nearIdx++); // Find next non-checked
                if(nearIdx < NEAR_SIZE) {
                    nearest[i*numNearest + j] = near[i*NEAR_SIZE + nearIdx].id;
                    near[i*NEAR_SIZE + nearIdx].checked=true;
                }
            }
        }
    }
}

#define RNG_BLOCKS 4096
#define RNG_THREADS 64
#define NEAR_SIZE 128
#define FAR_SIZE 32
#define SEEDS 128
#define SORT_THREADS 160

template<typename T, typename SUMTYPE, int MAX_DIM>
void getTailNeighborsGPU(T* vectors, SPTAG::SizeType N, std::shared_ptr<SPTAG::VectorIndex>& headIndex, std::unordered_set<int> headVectorIDS, int dim, DistPair<SUMTYPE>* results, int resultsPerVector, int BATCH_SIZE, int searchDepth, int numThreads, int metric) {

    auto t1 = std::chrono::high_resolution_clock::now();

    int NUM_THREADS = 32;
    int NUM_BLOCKS = min(BATCH_SIZE/NUM_THREADS, 128);

    Point<T,SUMTYPE,MAX_DIM>* headPoints = extractHeadPoints<T,SUMTYPE,MAX_DIM>(vectors, N, headVectorIDS, dim);
    Point<T,SUMTYPE,MAX_DIM>* tailPoints = extractTailPoints<T,SUMTYPE,MAX_DIM>(vectors, N, headVectorIDS, dim);

    Point<T,SUMTYPE,MAX_DIM>* d_tailPoints;
    cudaMalloc(&d_tailPoints, BATCH_SIZE*sizeof(Point<T,SUMTYPE,MAX_DIM>));

    Point<T,SUMTYPE,MAX_DIM>* d_headPoints;
    cudaMalloc(&d_headPoints, headVectorIDS.size()*sizeof(Point<T,SUMTYPE,MAX_DIM>));
    cudaMemcpy(d_headPoints, headPoints, headVectorIDS.size()*sizeof(Point<T,SUMTYPE,MAX_DIM>), cudaMemcpyHostToDevice);

// Memory to store batch of candidate IDs
    std::vector<CandidateElt<SUMTYPE>> candidates_near;
    candidates_near.reserve(BATCH_SIZE * NEAR_SIZE);

    std::vector<CandidateElt<SUMTYPE>> candidates_far;
    candidates_far.reserve(BATCH_SIZE * FAR_SIZE);

    const int numNearest = FAR_SIZE/32;
    std::vector<int> nearest;
    nearest.reserve(BATCH_SIZE*numNearest);

// Allocate temp memory for each batch on the GPU
    CandidateElt<SUMTYPE>* d_near;
    cudaMallocManaged(&d_near, BATCH_SIZE*NEAR_SIZE*sizeof(CandidateElt<SUMTYPE>));
    CandidateElt<SUMTYPE>* d_far;
    cudaMalloc(&d_far, BATCH_SIZE*FAR_SIZE*sizeof(CandidateElt<SUMTYPE>));
    int* d_nearest;
    cudaMallocManaged(&d_nearest, BATCH_SIZE*numNearest*sizeof(int));

    DistPair<SUMTYPE>* d_results;
    cudaMalloc(&d_results, BATCH_SIZE*resultsPerVector*sizeof(DistPair<SUMTYPE>));

    size_t curr_batch_size = BATCH_SIZE;

double sort_time=0;
double copy_search_time=0;
double compress_time=0;

    auto t1b = std::chrono::high_resolution_clock::now();
printf("Initialization time:%.2lf\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t1b-t1).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t1b-t1).count())/1000);

// Start of batch computation
    for(size_t offset=0; offset<(N-headVectorIDS.size()); offset+=BATCH_SIZE) {
        if(offset+BATCH_SIZE > (N-headVectorIDS.size())) {
            curr_batch_size = (N-headVectorIDS.size())-offset;
        }
printf("batch offset:%d, batch size:%d, total tail size:%d\n", offset, curr_batch_size, N-headVectorIDS.size());
    auto t2 = std::chrono::high_resolution_clock::now();

    // Copy tail vectors for batch to GPU
    cudaMemcpy(d_tailPoints, &tailPoints[offset], curr_batch_size*sizeof(Point<T,SUMTYPE,MAX_DIM>), cudaMemcpyHostToDevice);


// Get candidates from search on CPU
        #pragma omp parallel for num_threads(numThreads) schedule(dynamic,10)
        for(size_t fullID=0; fullID < curr_batch_size; fullID++) {

            SPTAG::COMMON::QueryResultSet<T> query((T*)&tailPoints[offset+fullID].coords[0], SEEDS);
            headIndex->SearchTree(query);

// Just use all seeds for initial candidate list
            if(NEAR_SIZE < SEEDS) {
                for(size_t i=0; i<NEAR_SIZE; i++) {
                    candidates_near[fullID*NEAR_SIZE+i].id = query.GetResult(i)->VID;
                    candidates_near[fullID*NEAR_SIZE+i].checked=false;
                }
                for(size_t i=0; i<(SEEDS-NEAR_SIZE); i++) {
                    candidates_far[fullID*FAR_SIZE+i].id = query.GetResult(NEAR_SIZE+i)->VID;
                    candidates_far[fullID*FAR_SIZE+i].checked=false;
                }
                for(size_t i=(SEEDS-NEAR_SIZE); i<FAR_SIZE; i++) {
                    candidates_far[fullID*FAR_SIZE+i].id = -1;
                    candidates_far[fullID*FAR_SIZE+i].checked=true;

                }
            }
            else {
                for(size_t i=0; i<SEEDS; i++) {
                    candidates_near[fullID*NEAR_SIZE+i].id = query.GetResult(i)->VID;
                    candidates_near[fullID*NEAR_SIZE+i].checked=false;
                }
                for(size_t i=SEEDS; i<NEAR_SIZE; i++) {
                    candidates_near[fullID*NEAR_SIZE+i].id = -1;
                    candidates_near[fullID*NEAR_SIZE+i].checked=true;
                }
                for(size_t i=0; i<FAR_SIZE; i++) {
                    candidates_far[fullID*FAR_SIZE+i].id = -1;
                    candidates_far[fullID*FAR_SIZE+i].checked=true;
                }
            }
        }
        // Copy initial far values
        cudaMemcpy(d_near, candidates_near.data(), curr_batch_size*NEAR_SIZE*sizeof(CandidateElt<SUMTYPE>), cudaMemcpyHostToDevice);
        cudaMemcpy(d_far, candidates_far.data(), curr_batch_size*FAR_SIZE*sizeof(CandidateElt<SUMTYPE>), cudaMemcpyHostToDevice);

    auto t3 = std::chrono::high_resolution_clock::now();
printf("Tree candidate time:%.2lf\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t3-t2).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count())/1000);


        // Continue searching graph to desired depth
        for(int depth=0; depth<searchDepth; depth++) {
auto l1 = std::chrono::high_resolution_clock::now();

// TODO - get rid of hard-coded values and have some max value or something, with error reporting when value is wrong...
            sortCandidates<T,SUMTYPE,MAX_DIM,NEAR_SIZE,FAR_SIZE,SORT_THREADS><<<NUM_BLOCKS, SORT_THREADS>>>(d_tailPoints, d_headPoints, d_near, d_far, d_nearest, curr_batch_size, metric, depth);

            cudaDeviceSynchronize();
auto l2 = std::chrono::high_resolution_clock::now();
sort_time += (double)std::chrono::duration_cast<std::chrono::seconds>(l2-l1).count();
sort_time += ((double)(std::chrono::duration_cast<std::chrono::milliseconds>(l2-l1).count())/1000.0);


            // Copy nearest neighbor back to CPU to find more candidates
            cudaMemcpy(nearest.data(), d_nearest, curr_batch_size*numNearest*sizeof(int), cudaMemcpyDeviceToHost);

            #pragma omp parallel for num_threads(numThreads)
            for(size_t fullID=0; fullID < curr_batch_size; fullID++) {
                for(size_t j=0; j<numNearest; j++) {
                    if (nearest[fullID*numNearest + j] > -1) {
// Get neighbors of nearest candidates
                        SizeType* neighborList = headIndex->GetNeighborList(nearest[fullID*numNearest + j]);
                        for(size_t i=0; i<32; i++) {
                            if(neighborList[i] > -1) {
                                candidates_far[fullID*FAR_SIZE + j*32 + i].id = neighborList[i];
                                candidates_far[fullID*FAR_SIZE + j*32 + i].checked=false;
                            }
                        }
                    }
                }
            }
            cudaMemcpy(d_far, candidates_far.data(), curr_batch_size*FAR_SIZE*sizeof(CandidateElt<SUMTYPE>), cudaMemcpyHostToDevice);
auto l3 = std::chrono::high_resolution_clock::now();

copy_search_time += (double)std::chrono::duration_cast<std::chrono::seconds>(l3-l2).count();
copy_search_time += ((double)(std::chrono::duration_cast<std::chrono::milliseconds>(l3-l2).count())/1000.0);
        }

auto l1 = std::chrono::high_resolution_clock::now();
        sortCandidates<T,SUMTYPE,MAX_DIM,NEAR_SIZE,FAR_SIZE,SORT_THREADS><<<NUM_BLOCKS, SORT_THREADS>>>(d_tailPoints, d_headPoints, d_near, d_far, d_nearest, curr_batch_size, metric, 0);
        printf("final sort kernel:%d\n", cudaDeviceSynchronize());
auto l2 = std::chrono::high_resolution_clock::now();
sort_time += (double)std::chrono::duration_cast<std::chrono::seconds>(l2-l1).count();
sort_time += ((double)(std::chrono::duration_cast<std::chrono::milliseconds>(l2-l1).count())/1000.0);

         
        compressToRNG<T,SUMTYPE,MAX_DIM,NEAR_SIZE,FAR_SIZE><<<NUM_BLOCKS, NUM_THREADS, NUM_THREADS*resultsPerVector*sizeof(DistPair<SUMTYPE>)>>>(d_tailPoints, d_headPoints, d_near, d_far, d_results, resultsPerVector, curr_batch_size, metric);
        printf("compress kernel:%d\n", cudaDeviceSynchronize());
auto l3 = std::chrono::high_resolution_clock::now();

        printf("memcpy:%d\n", cudaMemcpy(&results[offset*resultsPerVector], d_results, curr_batch_size*resultsPerVector*sizeof(DistPair<SUMTYPE>), cudaMemcpyDeviceToHost));
        printf("copied into offset:%d, size:%d\n", offset*resultsPerVector, curr_batch_size*resultsPerVector*sizeof(DistPair<SUMTYPE>));
compress_time += (double)std::chrono::duration_cast<std::chrono::seconds>(l3-l2).count();
compress_time += ((double)(std::chrono::duration_cast<std::chrono::milliseconds>(l3-l2).count())/1000.0);

    }

    printf("sort time:%.2lf, search time:%.2lf, compress time:%.2lf\n", sort_time, copy_search_time, compress_time);

    auto endt = std::chrono::high_resolution_clock::now();
    printf("Total time:%.2lf\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(endt-t1).count() + ((double)(std::chrono::duration_cast<std::chrono::milliseconds>(endt-t1).count()))/1000.0));

}
*/

#endif
