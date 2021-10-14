
#ifndef _SPTAG_COMMON_CUDA_TAILNEIGHBORS_H
#define _SPTAG_COMMON_CUDA_TAILNEIGHBORS_H

#include "Distance.hxx"
#include "KNN.hxx"

#include <cub/cub.cuh>
#include <chrono>

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
template<typename T, typename KEY_T, typename SUMTYPE, int Dim, int BLOCK_DIM>
__global__ void findTailRNG(Point<T,SUMTYPE,Dim>* headPoints, Point<T,SUMTYPE,Dim>* tailPoints, TPtree<T,KEY_T,SUMTYPE,Dim>* tptree, int KVAL, DistPair<SUMTYPE>* results, int metric, size_t numTails, int numHeads, QueryGroup* groups) {

  extern __shared__ char sharememory[];

  DistPair<SUMTYPE>* threadList = (&((DistPair<SUMTYPE>*)sharememory)[KVAL*threadIdx.x]);
  SUMTYPE max_dist = INFTY<SUMTYPE>();
  DistPair<SUMTYPE> temp;
  int read_id, write_id;

  for(int i=0; i<KVAL; i++) {
    threadList[i].dist = INFTY<SUMTYPE>();
    threadList[i].idx = -1;
  }

  Point<T,SUMTYPE,Dim> query;
  DistPair<SUMTYPE> target;
  DistPair<SUMTYPE> candidate;
  int leafId;
  bool good;

// If the queries were re-ordered, use the QueryGroup object to determine order to perform queries
// NOTE: Greatly improves locality and thread divergence
#if REORDER
  size_t tailId;
  for(size_t orderIdx = blockIdx.x*blockDim.x + threadIdx.x; orderIdx < numTails; orderIdx += gridDim.x*blockDim.x) {
    tailId = groups->query_ids[orderIdx];  
    query = tailPoints[tailId];
    leafId = searchForLeaf<T,KEY_T,SUMTYPE,Dim,Dim>(tptree, &query);
#else
  for(size_t tailId = blockIdx.x*blockDim.x + threadIdx.x; tailId < numTails; tailId += gridDim.x*blockDim.x) {
    query = tailPoints[tailId];
    leafId = searchForLeaf<T,KEY_T,SUMTYPE,Dim,Dim>(tptree, &query);
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
      candidate.dist = query.dist(&headPoints[candidate.idx], metric);

      if(candidate.dist < max_dist) { // If it is a candidate to be added to neighbor list

        for(read_id=0; candidate.dist > threadList[read_id].dist && good; read_id++) {
          if(violatesRNG<T, SUMTYPE, Dim>(headPoints, candidate, threadList[read_id], metric)) {
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
            if(!violatesRNG<T, SUMTYPE, Dim>(headPoints, threadList[read_id], candidate, metric)) {
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

__global__ void debug_warm_up_gpu() {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}


// Search each query into the TPTree to get the number of queries for each leaf node. This just generates a histogram
// for each leaf node so that we can compute offsets
template<typename T, typename KEY_T, typename SUMTYPE, int MAX_DIM>
__global__ void compute_group_sizes(QueryGroup* groups, TPtree<T,KEY_T,SUMTYPE,MAX_DIM>* tptree, Point<T,SUMTYPE,MAX_DIM>* queries, int N, int num_groups, int* queryMem) {
  groups->init_mem(N, num_groups, queryMem);

  Point<T,SUMTYPE,MAX_DIM> query;
  int leafId;
  
  for(int qidx = blockDim.x*blockIdx.x + threadIdx.x; qidx < N; qidx += blockDim.x*gridDim.x) {
    query = queries[qidx];
    leafId = searchForLeaf<T,KEY_T,SUMTYPE,MAX_DIM,MAX_DIM>(tptree, &query);
    atomicAdd(&(groups->sizes[leafId]), 1);   
  }
}

// Given the offsets and sizes of queries assigned to each leaf node, writes the list of queries assigned
// to each leaf.  The list of query_ids can then be used during neighborhood search to improve locality
template<typename T, typename KEY_T, typename SUMTYPE, int MAX_DIM>
__global__ void assign_queries_to_group(QueryGroup* groups, TPtree<T,KEY_T,SUMTYPE,MAX_DIM>* tptree, Point<T,SUMTYPE,MAX_DIM>* queries, int N, int num_groups) {
  Point<T,SUMTYPE,MAX_DIM> query;
  int leafId;
  int idx_in_leaf;
  
  for(int qidx = blockDim.x*blockIdx.x + threadIdx.x; qidx < N; qidx += blockDim.x*gridDim.x) {
    query = queries[qidx];
    leafId = searchForLeaf<T,KEY_T,SUMTYPE,MAX_DIM,MAX_DIM>(tptree, &query);
    idx_in_leaf = atomicAdd(&(groups->sizes[leafId]), 1);   
    groups->query_ids[groups->offsets[leafId]+idx_in_leaf] = qidx;
  }
  
}

// Gets the list of queries that are to be searched into each leaf.  The QueryGroup structure uses the memory
// provided with queryMem, which is required to be allocated with size N + 2*num_groups.  
template<typename T, typename KEY_T, typename SUMTYPE, int MAX_DIM>
__host__ void get_query_groups(QueryGroup* groups, TPtree<T,KEY_T,SUMTYPE,MAX_DIM>* tptree, Point<T,SUMTYPE,MAX_DIM>* queries, int N, int num_groups, int* queryMem, int NUM_BLOCKS, int NUM_THREADS) {

  CUDA_CHECK(cudaMemset(queryMem, 0, (N+2*num_groups)*sizeof(int)));

  compute_group_sizes<T,KEY_T,SUMTYPE,MAX_DIM><<<NUM_BLOCKS,NUM_THREADS>>>(groups, tptree, queries, N, num_groups, queryMem);
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

  assign_queries_to_group<T,KEY_T,SUMTYPE,MAX_DIM><<<NUM_BLOCKS,NUM_THREADS>>>(groups, tptree, queries, N, num_groups);
  CUDA_CHECK(cudaDeviceSynchronize());

  delete[] h_sizes;
  delete[] h_offsets;
}
    

template<typename T, typename KEY_T, typename SUMTYPE, int MAX_DIM>
__host__ void get_query_groups(QueryGroup* groups, TPtree<T,KEY_T,SUMTYPE,MAX_DIM>* tptree, Point<T,SUMTYPE,MAX_DIM>* queries, int N, int num_groups, int* queryMem, int NUM_BLOCKS, int NUM_THREADS) {

  CUDA_CHECK(cudaMemset(queryMem, 0, (N+2*num_groups)*sizeof(int)));

  compute_group_sizes<T,KEY_T,SUMTYPE,MAX_DIM><<<NUM_BLOCKS,NUM_THREADS>>>(groups, tptree, queries, N, num_groups, queryMem);
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

  assign_queries_to_group<T,KEY_T,SUMTYPE,MAX_DIM><<<NUM_BLOCKS,NUM_THREADS>>>(groups, tptree, queries, N, num_groups);
  CUDA_CHECK(cudaDeviceSynchronize());

  delete[] h_sizes;
  delete[] h_offsets;
}
    

template<typename T, typename KEY_T, typename SUMTYPE, int MAX_DIM>
void getTailNeighborsTPT(T* vectors, SPTAG::SizeType N, SPTAG::VectorIndex* headIndex, std::unordered_set<int>& headVectorIDS, int dim, DistPair<SUMTYPE>* results, int RNG_SIZE, int numThreads, int NUM_TREES, int LEAF_SIZE, int metric, int NUM_GPUS) {

    auto premem_t = std::chrono::high_resolution_clock::now();

    int numDevicesOnHost;
    CUDA_CHECK(cudaGetDeviceCount(&numDevicesOnHost));

    if(numDevicesOnHost < NUM_GPUS) {
      LOG(SPTAG::Helper::LogLevel::LL_Error, "NumGPUs parameter %d, but only %d devices available on system.  Exiting.\n", NUM_GPUS, numDevicesOnHost);
      exit(1);
    }

    LOG(SPTAG::Helper::LogLevel::LL_Info, "Building SSD index with %d GPUs...\n", NUM_GPUS);
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "Total of %d GPU devices on system, using %d of them.\n", numDevicesOnHost, NUM_GPUS);

    int resultErr;
    size_t headRows, tailRows;
    headRows = headIndex->GetNumSamples();
    if(headVectorIDS.size() == 0) { // If list of headVectors is not given, have to extract them from headIndex
        tailRows = N;
    }
    else {
        tailRows = N - headVectorIDS.size();
    }
    int TPTlevels = (int)std::log2(headRows/LEAF_SIZE);

    Point<T,SUMTYPE,MAX_DIM>* headPoints;
    Point<T,SUMTYPE,MAX_DIM>* tailPoints; 
    headPoints = new Point<T,SUMTYPE,MAX_DIM>[headRows];
    tailPoints = new Point<T,SUMTYPE,MAX_DIM>[tailRows];

    // If headVectors not given, extract from headIndex and use all vectors as tails
    extractHeadPointsFromIndex<T,SUMTYPE,MAX_DIM>(vectors, headIndex, headPoints, dim);
    if(headVectorIDS.size() == 0) {
        extractFullVectorPoints<T,SUMTYPE,MAX_DIM>(vectors, tailPoints, N, dim);
    }
    else {
        extractTailPoints<T,SUMTYPE,MAX_DIM>(vectors, tailPoints, N, headVectorIDS, dim);
    }
  
    // Get number and offset of tail vectors to be assigned to each GPU
    std::vector<size_t> tailsPerGPU(NUM_GPUS);
    std::vector<size_t> GPUOffset(NUM_GPUS);
    for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
        tailsPerGPU[gpuNum] = tailRows / NUM_GPUS;
        if(tailRows % NUM_GPUS > gpuNum) tailsPerGPU[gpuNum]++;
    }
    GPUOffset[0] = 0;
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU 0: tails:%lu, offset:%lu\n", tailsPerGPU[0], GPUOffset[0]);
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU 0: tails:%lu, offset:%lu\n", tailsPerGPU[0], GPUOffset[0]);
    for(int gpuNum=1; gpuNum < NUM_GPUS; ++gpuNum) {
        GPUOffset[gpuNum] = GPUOffset[gpuNum-1] + tailsPerGPU[gpuNum-1];
        LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU %d: tails:%lu, offset:%lu\n", gpuNum, tailsPerGPU[gpuNum], GPUOffset[gpuNum]);
    }

    // Streams and memory pointers for each GPU
    std::vector<cudaStream_t> streams(NUM_GPUS);
    std::vector<size_t> BATCH_SIZE(NUM_GPUS);

    Point<T,SUMTYPE,MAX_DIM>** d_tailPoints = new Point<T,SUMTYPE,MAX_DIM>*[NUM_GPUS];
    Point<T,SUMTYPE,MAX_DIM>** d_headPoints = new Point<T,SUMTYPE,MAX_DIM>*[NUM_GPUS];
    DistPair<SUMTYPE>** d_results = new DistPair<SUMTYPE>*[NUM_GPUS];
    TPtree<T,KEY_T,SUMTYPE,MAX_DIM>** tptree = new TPtree<T,KEY_T,SUMTYPE,MAX_DIM>*[NUM_GPUS];
    TPtree<T,KEY_T,SUMTYPE,MAX_DIM>** d_tptree = new TPtree<T,KEY_T,SUMTYPE,MAX_DIM>*[NUM_GPUS];

    // memory on each GPU for QuerySet reordering structures
    std::vector<int*> d_queryMem(NUM_GPUS);
    std::vector<QueryGroup*> d_queryGroups(NUM_GPUS);

    LOG(SPTAG::Helper::LogLevel::LL_Info, "Setting up each of the %d GPUs...\n", NUM_GPUS);

    // For each GPU, compute number of batches, allocate memory, copy head vectors to each
    for(int gpuNum=0; gpuNum < NUM_GPUS; gpuNum++) {

        CUDA_CHECK(cudaSetDevice(gpuNum)); // Set current working GPU
        CUDA_CHECK(cudaStreamCreate(&streams[gpuNum])); // Create CDUA stream for each GPU

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, gpuNum));
  
        LOG(SPTAG::Helper::LogLevel::LL_Info, "GPU %d - %s\n", gpuNum, prop.name);
  
        debug_warm_up_gpu<<<1,32,0,streams[gpuNum]>>>();
        resultErr = cudaStreamSynchronize(streams[gpuNum]);
        LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU test/warmup complete - kernel status:%d\n", resultErr);
  
        size_t freeMem, totalMem;
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
      
        // Auto-compute batch size based on available memory on the GPU
        size_t headVecSize = headRows*sizeof(Point<T,SUMTYPE,MAX_DIM>);

        int randSize = min((int)headRows, (int)1024)*48; // Memory used for GPU random number generator
        size_t treeSize = 20*headRows + (size_t)randSize;

        size_t tailMemAvail = (freeMem*0.9) - (headVecSize+treeSize); // Only use 90% of total memory to be safe

        // memory needed for points, results, and query reorder mem
        int maxEltsPerBatch = tailMemAvail / (sizeof(Point<T,SUMTYPE,MAX_DIM>) + RNG_SIZE*sizeof(DistPair<SUMTYPE>) + 2*sizeof(int));

        BATCH_SIZE[gpuNum] = min(maxEltsPerBatch, (int)(tailsPerGPU[gpuNum]));

       // If GPU memory is insufficient or so limited that we need so many batches it becomes inefficient, return error
        if(BATCH_SIZE[gpuNum] == 0 || ((int)tailsPerGPU[gpuNum]) / BATCH_SIZE[gpuNum] > 10000) {
            LOG(SPTAG::Helper::LogLevel::LL_Error, "Insufficient GPU memory to build SSD index on GPU %d.  Available GPU memory:%lu MB, Head index requires:%lu MB, leaving a maximum batch size of %d elements, which is too small to run efficiently.\n", gpuNum, (freeMem)/1000000, (headVecSize+treeSize)/1000000, maxEltsPerBatch);
            exit(1);
        }
  
        LOG(SPTAG::Helper::LogLevel::LL_Info, "Memory for head vectors:%lu MiB, Memory for TP trees:%lu MiB, Memory left for tail vectors:%lu MiB, total tail vectors:%lu, batch size:%d, total batches:%d\n", headVecSize/1000000, treeSize/1000000, tailMemAvail/1000000, tailsPerGPU[gpuNum], BATCH_SIZE[gpuNum], (((BATCH_SIZE[gpuNum]-1)+tailsPerGPU[gpuNum]) / BATCH_SIZE[gpuNum]));
  
        LOG(SPTAG::Helper::LogLevel::LL_Debug, "Allocating GPU memory: tail points:%lu MiB, head points:%lu MiB, results:%lu MiB, TPT:%lu MiB, Total:%lu MiB\n", (BATCH_SIZE[gpuNum]*sizeof(Point<T,SUMTYPE,MAX_DIM>))/1000000, (headRows*sizeof(Point<T,SUMTYPE,MAX_DIM>))/1000000, (BATCH_SIZE[gpuNum]*RNG_SIZE*sizeof(DistPair<SUMTYPE>))/1000000, (sizeof(TPtree<T,KEY_T,SUMTYPE, MAX_DIM>))/1000000, ((BATCH_SIZE[gpuNum]+headRows)*sizeof(Point<T,SUMTYPE,MAX_DIM>) + (BATCH_SIZE[gpuNum]*RNG_SIZE*sizeof(DistPair<SUMTYPE>)) + (sizeof(TPtree<T,KEY_T,SUMTYPE,MAX_DIM>)))/1000000);

        // Allocate needed memory on the GPU
        CUDA_CHECK(cudaMalloc(&d_tailPoints[gpuNum], BATCH_SIZE[gpuNum]*sizeof(Point<T,SUMTYPE,MAX_DIM>)));
        CUDA_CHECK(cudaMalloc(&d_headPoints[gpuNum], headRows*sizeof(Point<T,SUMTYPE,MAX_DIM>)));
        CUDA_CHECK(cudaMalloc(&d_results[gpuNum], BATCH_SIZE[gpuNum]*RNG_SIZE*sizeof(DistPair<SUMTYPE>)));

        // Prepare memory for TPTs
        CUDA_CHECK(cudaMalloc(&d_tptree[gpuNum], sizeof(TPtree<T,KEY_T,SUMTYPE, MAX_DIM>)));
        tptree[gpuNum] = new TPtree<T,KEY_T,SUMTYPE,MAX_DIM>;
        tptree[gpuNum]->initialize(headRows, TPTlevels);

        LOG(SPTAG::Helper::LogLevel::LL_Debug, "tpt structure initialized for %lu head vectors, %d levels, leaf size:%d\n", headRows, TPTlevels, LEAF_SIZE);

        // Alloc memory for QuerySet structure
        CUDA_CHECK(cudaMalloc(&d_queryMem[gpuNum], BATCH_SIZE[gpuNum]*sizeof(int) + 2*tptree[gpuNum]->num_leaves*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_queryGroups[gpuNum], sizeof(QueryGroup)));

        // Copy head points to GPU
        CUDA_CHECK(cudaMemcpy(d_headPoints[gpuNum], headPoints, headRows*sizeof(Point<T,SUMTYPE,MAX_DIM>), cudaMemcpyHostToDevice));

    } // End loop to set up memory on each GPU

    delete[] headPoints; // headPoints copied to GPU, no longer need on CPU

    const int NUM_THREADS = 32;
    int NUM_BLOCKS = min((int)(BATCH_SIZE[0]/NUM_THREADS), 10240);

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
            if(offset[gpuNum]+BATCH_SIZE[gpuNum] > tailsPerGPU[gpuNum]) {
                curr_batch_size[gpuNum] = tailsPerGPU[gpuNum]-offset[gpuNum];
            }
            LOG(SPTAG::Helper::LogLevel::LL_Info, "GPU %d - starting batch with offset:%lu, size:%lu, TailRows:%lld\n", gpuNum, offset[gpuNum], curr_batch_size[gpuNum], tailsPerGPU[gpuNum]);
    
            cudaSetDevice(gpuNum);

            // Copy next batch of tail vectors to corresponding GPU
            resultErr = cudaMemcpyAsync(d_tailPoints[gpuNum], &tailPoints[GPUOffset[gpuNum]+offset[gpuNum]], curr_batch_size[gpuNum]*sizeof(Point<T,SUMTYPE,MAX_DIM>), cudaMemcpyHostToDevice, streams[gpuNum]);
            LOG(SPTAG::Helper::LogLevel::LL_Debug, "Copied %lu tail points to GPU - kernel status:%d\n", curr_batch_size[gpuNum], resultErr);

            // Initialize and copy results for batch to each GPU
            for(int i=0; i<curr_batch_size[gpuNum]*RNG_SIZE; i++) {
                results[(GPUOffset[gpuNum] + offset[gpuNum])*RNG_SIZE + i].idx=-1;
                results[(GPUOffset[gpuNum] + offset[gpuNum])*RNG_SIZE + i].dist=INFTY<SUMTYPE>();
            }
            resultErr = cudaMemcpyAsync(d_results[gpuNum], &results[(GPUOffset[gpuNum]+offset[gpuNum])*RNG_SIZE], curr_batch_size[gpuNum]*RNG_SIZE*sizeof(DistPair<SUMTYPE>), cudaMemcpyHostToDevice, streams[gpuNum]);

            LOG(SPTAG::Helper::LogLevel::LL_Debug, "Copying initialized result list to GPU - batch size:%lu - replica count:%d, copy bytes:%lu, GPU offset:%lu, total result offset:%lu, - kernel status:%d\n", curr_batch_size[gpuNum], RNG_SIZE, curr_batch_size[gpuNum]*RNG_SIZE*sizeof(DistPair<SUMTYPE>), GPUOffset[gpuNum], (GPUOffset[gpuNum]+offset[gpuNum])*RNG_SIZE, resultErr);
        }


        // For each tree, create a new TPT and use it to refine tail neighbor list of batch
        for(int tree_id=0; tree_id < NUM_TREES; ++tree_id) {
            NUM_BLOCKS = min((int)(BATCH_SIZE[0]/NUM_THREADS), 10240);

auto t1 = std::chrono::high_resolution_clock::now();
            // Create TPT on each GPU
            create_tptree_multigpu<T, KEY_T, SUMTYPE, MAX_DIM>(tptree, d_headPoints, headRows, TPTlevels, NUM_GPUS, streams.data(), 2);
            CUDA_CHECK(cudaDeviceSynchronize());
            LOG(SPTAG::Helper::LogLevel::LL_Debug, "TPT %d created on all GPUs\n", tree_id);

            // Copy TPTs to each GPU
            for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
                CUDA_CHECK(cudaSetDevice(gpuNum));
                CUDA_CHECK(cudaMemcpy(d_tptree[gpuNum], tptree[gpuNum], sizeof(TPtree<T,KEY_T,SUMTYPE,MAX_DIM>), cudaMemcpyHostToDevice));
            }

#if REORDER
            // Compute QuerySet lists based on TPT, which can then be used to improve locality/divergence during RNG search
            for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
                CUDA_CHECK(cudaSetDevice(gpuNum));
                get_query_groups<T,KEY_T,SUMTYPE,MAX_DIM>(d_queryGroups[gpuNum], d_tptree[gpuNum], d_tailPoints[gpuNum], (int)(curr_batch_size[gpuNum]), (int)tptree[gpuNum]->num_leaves, d_queryMem[gpuNum], NUM_BLOCKS, NUM_THREADS);
            }
#endif

auto t2 = std::chrono::high_resolution_clock::now();

            // Call main kernel on each GPU
            for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
                CUDA_CHECK(cudaSetDevice(gpuNum));
                findTailRNG<T,KEY_T,SUMTYPE,MAX_DIM,NUM_THREADS><<<NUM_BLOCKS, NUM_THREADS, sizeof(DistPair<SUMTYPE>)*RNG_SIZE*NUM_THREADS, streams[gpuNum]>>>(d_headPoints[gpuNum], d_tailPoints[gpuNum], d_tptree[gpuNum], RNG_SIZE, d_results[gpuNum], metric, (size_t)curr_batch_size[gpuNum], headRows, d_queryGroups[gpuNum]);
            }

            CUDA_CHECK(cudaDeviceSynchronize());
            LOG(SPTAG::Helper::LogLevel::LL_Debug, "All GPUs finished finding neighbors of tails using TPT %d\n", tree_id);

auto t3 = std::chrono::high_resolution_clock::now();

LOG(SPTAG::Helper::LogLevel::LL_Debug, "Tree %d complete, time to build tree:%.2lf, time to compute tail neighbors:%.2lf\n", tree_id, ((double)std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count())/1000, ((double)std::chrono::duration_cast<std::chrono::seconds>(t3-t2).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count())/1000);
            
        } // TPT loop

        LOG(SPTAG::Helper::LogLevel::LL_Debug, "Batch complete on all GPUs, copying results back to CPU...\n");

        // Copy results of batch from each GPU to CPU result set
        for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
            CUDA_CHECK(cudaMemcpy(&results[(GPUOffset[gpuNum]+offset[gpuNum])*RNG_SIZE], d_results[gpuNum], curr_batch_size[gpuNum]*RNG_SIZE*sizeof(DistPair<SUMTYPE>), cudaMemcpyDeviceToHost));
        }
        LOG(SPTAG::Helper::LogLevel::LL_Debug, "Finished copying all batch results back to CPU\n");

        // Update all offsets and check if done
        done=true;
        for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
            offset[gpuNum] += curr_batch_size[gpuNum];
            if(offset[gpuNum] < tailsPerGPU[gpuNum]) {
                done=false;
            }
        }
    } // Batches loop (while !done)

    auto ssd_t2 = std::chrono::high_resolution_clock::now();


    LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU SSD build complete.  Freeing GPU memory...\n");

    for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
        CUDA_CHECK(cudaFree(d_headPoints[gpuNum]));
        CUDA_CHECK(cudaFree(d_tailPoints[gpuNum]));
        CUDA_CHECK(cudaFree(d_results[gpuNum]));
        CUDA_CHECK(cudaFree(d_tptree[gpuNum]));
        delete tptree[gpuNum];
    }
    delete[] tailPoints;

    delete[] d_headPoints;
    delete[] d_tailPoints;
    delete[] d_results;
    delete[] tptree;
    delete[] d_tptree;

    auto ssd_t3 = std::chrono::high_resolution_clock::now();

    LOG(SPTAG::Helper::LogLevel::LL_Info, "Mam alloc time:%0.2lf, GPU time to build index:%.2lf, Memory free time:%.2lf\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(ssd_t1-premem_t).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(ssd_t1-premem_t).count())/1000, ((double)std::chrono::duration_cast<std::chrono::seconds>(ssd_t2-ssd_t1).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(ssd_t2-ssd_t1).count())/1000, ((double)std::chrono::duration_cast<std::chrono::seconds>(ssd_t3-ssd_t2).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(ssd_t3-ssd_t2).count())/1000);

}

/*************************************************************************************************
 * Deprecated code from Hybrid CPU/GPU SSD Index builder
 *************************************************************************************************/
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
