
#ifndef _SPTAG_COMMON_CUDA_REFINE_H
#define _SPTAG_COMMON_CUDA_REFINE_H

#include "../../VectorIndex.h"
#include "Distance.hxx"

#include <cub/cub.cuh>


#define MAX_CHECK_COUNT 32 // Max number of neighbors checked during each step of refinement.

#define LISTCAP 2048 // Maximum size of buffer for each threadblock during refinemnt
#define LISTSIZE 1024 // Maximum size of nearest neighbors stored during refinement


using namespace SPTAG;

template <typename SUMTYPE>
class ListElt {
  public:
  int id;
  SUMTYPE dist;
  bool checkedFlag;

  __device__ ListElt& operator=(const ListElt& other) {
    id = other.id;
    dist = other.dist;
    checkedFlag = other.checkedFlag;
    return *this;
  }
};



template<typename T>
void getCandidates(SPTAG::VectorIndex* index, int numVectors, int candidatesPerVector, int* candidates) {

#pragma omp parallel for schedule(dynamic)
  for (SizeType i = 0; i < numVectors; i++)
  {
    SPTAG::COMMON::QueryResultSet<T> query((const T*)index->GetSample(i), candidatesPerVector);
     index->SearchTree(query);
     for (SPTAG::DimensionType j = 0; j < candidatesPerVector; j++) {
       candidates[i*candidatesPerVector+j] = query.GetResult(j)->VID;
     }
  }
}


template<typename SUMTYPE, int NUM_REGS, int NUM_THREADS>
__device__ void loadRegisters( ListElt<SUMTYPE>* regs,  ListElt<SUMTYPE>* listMem, int* listSize) {
  for(int i=0; i<NUM_REGS; i++) {
    if(i*NUM_THREADS + threadIdx.x < *listSize) {
      regs[i] = listMem[i*NUM_THREADS + threadIdx.x];
    }
    else {
      regs[i].id = INFTY<int>();
      regs[i].dist = INFTY<SUMTYPE>();
    }
  }
}

template<typename SUMTYPE, int NUM_REGS, int NUM_THREADS>
__device__ void storeRegisters( ListElt<SUMTYPE>* regs,  ListElt<SUMTYPE>* listMem, int* listSize) {
  for(int i=0; i<NUM_REGS; i++) {
    if(i*NUM_THREADS + threadIdx.x < *listSize) {
      listMem[i*NUM_THREADS + threadIdx.x] = regs[i];
    }
    else {
      listMem[i*NUM_THREADS+threadIdx.x].id = INFTY<int>();
      listMem[i*NUM_THREADS+threadIdx.x].dist = INFTY<int>();
    }
  }
}



template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__device__ void sortListById( ListElt<SUMTYPE>* listMem, int* listSize, void* temp_storage) {

   ListElt<SUMTYPE> sortMem[LISTCAP/NUM_THREADS];
  int sortKeys[LISTCAP/NUM_THREADS];

/* Sort list by ID to remove duplicates */
  /* Load list into registers to sort */
  loadRegisters<SUMTYPE,LISTCAP/NUM_THREADS, NUM_THREADS>(sortMem, listMem, listSize);

  for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
    sortKeys[i] = sortMem[i].id;
  }

  /* Sort by ID in registers */
  typedef cub::BlockRadixSort<int, NUM_THREADS, LISTCAP/NUM_THREADS, ListElt<SUMTYPE>> BlockRadixSort;
  BlockRadixSort(*(static_cast<typename BlockRadixSort::TempStorage*>(temp_storage))).SortBlockedToStriped(sortKeys, sortMem);

  __syncthreads();


  storeRegisters<SUMTYPE,LISTCAP/NUM_THREADS, NUM_THREADS>(sortMem, listMem, listSize);
  __syncthreads();

}



template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__device__ void sortListByDist( ListElt<SUMTYPE>* listMem, int* listSize, void* temp_storage) {

   ListElt<SUMTYPE> sortMem[LISTCAP/NUM_THREADS];
  SUMTYPE sortKeys[LISTCAP/NUM_THREADS];

/* Sort list by ID to remove duplicates */
  /* Load list into registers to sort */
  loadRegisters<SUMTYPE,LISTCAP/NUM_THREADS, NUM_THREADS>(sortMem, listMem, listSize);

  for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
    sortKeys[i] = sortMem[i].dist;
  }

  /* Sort by ID in registers */
  typedef cub::BlockRadixSort<SUMTYPE, NUM_THREADS, LISTCAP/NUM_THREADS, ListElt<SUMTYPE>> BlockRadixSort;
  BlockRadixSort(*(static_cast<typename BlockRadixSort::TempStorage*>(temp_storage))).SortBlockedToStriped(sortKeys, sortMem);
  __syncthreads();

  storeRegisters<SUMTYPE,LISTCAP/NUM_THREADS, NUM_THREADS>(sortMem, listMem, listSize);
  __syncthreads();
}


/* Remove duplicates and compact list with prefix sums */
template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__device__ void removeDuplicatesAndCompact( ListElt<SUMTYPE>* listMem, int* listSize, void *temp_storage, int src, int* /*borderVals*/) {

   ListElt<SUMTYPE> sortMem[LISTCAP/NUM_THREADS];
  int sortKeys[LISTCAP/NUM_THREADS];

  /* Copy weather is duplicate or not into registers */
  for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
    sortMem[i] = listMem[threadIdx.x*(LISTCAP/NUM_THREADS) + i];
    sortKeys[i] = 0;
    if(listMem[threadIdx.x*(LISTCAP/NUM_THREADS) + i].id == -1) {
      sortKeys[i] = 0;
    }
    else if(i==0 && threadIdx.x==0) {
      sortKeys[i] = 1;
    }
    else if(threadIdx.x*(LISTCAP/NUM_THREADS) + i < *listSize && listMem[threadIdx.x*(LISTCAP/NUM_THREADS) + i].id != INFTY<int>()) {
      sortKeys[i] = (sortMem[i].id != listMem[threadIdx.x*(LISTCAP/NUM_THREADS) + i - 1].id);
    }
  }

__syncthreads();
for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
        listMem[threadIdx.x*(LISTCAP/NUM_THREADS) + i] = sortMem[i];
}

  __syncthreads();
  typedef cub::BlockScan<int, NUM_THREADS> BlockScan;
  BlockScan(*(static_cast<typename BlockScan::TempStorage*>(temp_storage))).InclusiveSum(sortKeys, sortKeys);

  __syncthreads();
  for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
    listMem[threadIdx.x*(LISTCAP/NUM_THREADS)+i] = sortMem[i];
  }
  /* Share boarders of prefix sums */
  __shared__ int borderVals[NUM_THREADS];
  borderVals[threadIdx.x] = sortKeys[LISTCAP/NUM_THREADS - 1];

  __syncthreads();

  if(threadIdx.x==0 || borderVals[threadIdx.x-1] != sortKeys[0]) {
    listMem[sortKeys[0]-1] = sortMem[0];
  }
  for(int i=1; i<LISTCAP/NUM_THREADS; i++) {
    if(sortKeys[i] > sortKeys[i-1]) {
      listMem[sortKeys[i]-1] = sortMem[i];
    }
  }
  __syncthreads();
  *listSize = borderVals[NUM_THREADS-1];
  __syncthreads();

}



template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__device__ void checkClosestNeighbors(Point<T,SUMTYPE,MAX_DIM>* d_points, int src, int* d_graph,  ListElt<SUMTYPE>* listMem, int* listSize, int KVAL, int metric) {

/* Maximum number of vertices to check before list fills up*/
  int max_check = min(MAX_CHECK_COUNT, (LISTCAP-*listSize)/KVAL);

  int write_offset;
  __shared__ int check_count;
  check_count=0;
  __shared__ int new_listSize;
  new_listSize = *listSize;
  __syncthreads();


/* Fill up list with neighbors of nearest unchecked vectices */
  for(int i=threadIdx.x; i<LISTSIZE && check_count < max_check; i+=NUM_THREADS) {
    if(!listMem[i].checkedFlag && listMem[i].id != -1) {
      if(atomicAdd(&check_count, 1) < max_check) {
        write_offset = atomicAdd(&new_listSize, KVAL);
        listMem[i].checkedFlag=true;

        for(int j=0; j<KVAL; j++) {
          listMem[write_offset+j].id = d_graph[listMem[i].id*KVAL+j];
        }
      }
    }
  }
  __syncthreads();
/* Compute distance to all newly added vetrices */

  for(int i=(*listSize)+threadIdx.x; i<new_listSize; i+=NUM_THREADS) {
    if(listMem[i].id == src || listMem[i].id == -1) {
//      listMem[i].id = -1;
      listMem[i].dist = INFTY<SUMTYPE>();
    }
    else if (metric == 0){
      listMem[i].dist = d_points[src].l2(&d_points[listMem[i].id]);
    }
    else {
      listMem[i].dist = d_points[src].cosine(&d_points[listMem[i].id]);
    }
  }
  __syncthreads();
  *listSize = new_listSize;
  __syncthreads();


}


/* Sequential version for testing */
template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__device__ void shrinkListRNG_sequential(Point<T,SUMTYPE,MAX_DIM>* d_points, int* d_graph, int src, ListElt<SUMTYPE>* listMem, int listSize, int KVAL, int metric) {

  if(threadIdx.x==0) {
    int* nodes;
    int count=0;
    for(int j=0; j<listSize && j < KVAL; j++) {
      nodes = &d_graph[src*KVAL];
      ListElt<SUMTYPE> item = listMem[j];
      if(item.id < 0) break;
      bool good = true;
      if(item.id == src) good = false;

      for(int k=0; k<count && good; k++) {
        if(d_points[nodes[k]].l2(&d_points[item.id]) <= item.dist) {
	  good = false;
          break;
        }
      } 
      if(good) nodes[count++] = item.id;
    }
    for(int j=count; j<KVAL;j++) nodes[j] = -1;
  }
  __syncthreads();
}

template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__device__ void shrinkListRNG(Point<T,SUMTYPE,MAX_DIM>* d_points, int* d_graph, int src, ListElt<SUMTYPE>* listMem, int listSize, int KVAL, int metric) {

  volatile __shared__ bool good;
  int write_idx=0;
  int read_idx=0;

  for(read_idx=0; write_idx < KVAL && read_idx < listSize; read_idx++) {

        if(listMem[read_idx].id == -1 || listMem[read_idx].id == src) {
            good = false;
        }
        else {
            good = true;
        }
    __syncthreads();

    if(metric == 0) {
      for(int j=threadIdx.x; j<write_idx && good; j+=NUM_THREADS) {
        /* If it violates RNG */
        if(listMem[read_idx].dist >= d_points[listMem[read_idx].id].l2(&d_points[listMem[j].id])) {
          good=false;
        }
      }
    }
    else {
      for(int j=threadIdx.x; j<write_idx && good; j+=NUM_THREADS) {
        /* If it violates RNG */
        if(listMem[read_idx].dist >= d_points[listMem[read_idx].id].cosine(&d_points[listMem[j].id])) {
          good=false;
        }
      }
    }
    __syncthreads();
    if(good) {
      listMem[write_idx].id = listMem[read_idx].id;
      write_idx++;
    }
    __syncthreads();
  }
  for(int i=threadIdx.x; i<write_idx; i+=NUM_THREADS) {
    d_graph[src*KVAL+i] = listMem[i].id;
  }
  for(int i=write_idx+threadIdx.x; i<KVAL; i+=NUM_THREADS) {
      d_graph[src*KVAL+i] = -1;
  }
  __syncthreads();
}

#define TEST_THREADS 32
#define TEST_BLOCKS 1024
template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__global__ void refineBatch_kernel(Point<T,SUMTYPE,MAX_DIM>* d_points, int batchSize, int batchOffset, int* d_graph, int* candidates, ListElt<SUMTYPE>* listMemAll, int candidatesPerVector, int KVAL, int refineDepth, int metric) {

  // Offset to the memory allocated for this block's list
   ListElt<SUMTYPE>* listMem = &listMemAll[blockIdx.x*LISTCAP];

  __shared__ int listSize;
  __shared__ int borderVals[NUM_THREADS];
  typedef cub::BlockRadixSort<SUMTYPE, NUM_THREADS, LISTCAP/NUM_THREADS, ListElt<SUMTYPE>> BlockRadixSortT;
  __shared__ typename BlockRadixSortT::TempStorage temp_storage;

  for(int src=blockIdx.x+batchOffset; src<batchOffset+batchSize; src+=gridDim.x) {

// Place source vector in list
    listMem[0].id = src;
    listMem[0].dist = 0;
    listMem[0].checkedFlag = false;

    // Place all candidates into the list of neighbors to check during refinement
    for(int i=threadIdx.x; i<candidatesPerVector; i+=NUM_THREADS) {
      listMem[i+1].id = candidates[src*candidatesPerVector+i];
      if(listMem[i+1].id == -1) {
        listMem[i+1].dist = INFTY<SUMTYPE>();
        listMem[i+1].checkedFlag = true;
      } else {
        if(metric == 0) {
          listMem[i+1].dist = d_points[src].l2(&d_points[listMem[i+1].id]);
        }
        else {
          listMem[i+1].dist = d_points[src].cosine(&d_points[listMem[i+1].id]);
        }
        listMem[i+1].checkedFlag = false;
      }
    }
   listSize=candidatesPerVector+1;
    __syncthreads();
    sortListByDist<T,SUMTYPE,MAX_DIM,NUM_THREADS>(listMem, &listSize, &temp_storage);


// Perform DFS refinement up to refineDepth distance
    for(int iter=0; iter < refineDepth; iter++) {
      listSize = min(LISTSIZE, listSize);
      __syncthreads();
      checkClosestNeighbors<T,SUMTYPE,MAX_DIM,NUM_THREADS>(d_points, src, d_graph, listMem, &listSize, KVAL, metric);

      sortListById<T,SUMTYPE,MAX_DIM,NUM_THREADS>(listMem, &listSize, &temp_storage);
      removeDuplicatesAndCompact<T,SUMTYPE,MAX_DIM,NUM_THREADS>(listMem, &listSize, &temp_storage, src, borderVals);
      sortListByDist<T,SUMTYPE,MAX_DIM,NUM_THREADS>(listMem, &listSize, &temp_storage);
    }

// Place original graph neirhbors into list to compute the new RNG set of neighbors
/*
    if(listSize > LISTCAP - KVAL) listSize = LISTCAP-KVAL;
    __syncthreads();
    for(int i=threadIdx.x; i<KVAL; i+=NUM_THREADS) {
      listMem[listSize+i].id = d_graph[src*KVAL+i];
      if(listMem[listSize+i].id == -1 || listMem[listSize+i].id == src) {
        listMem[listSize+i].dist = INFTY<SUMTYPE>();
      }
      else {
      if(metric==0) {
        listMem[listSize+i].dist = d_points[src].l2(&d_points[listMem[listSize+i].id]);
      }
      else {
        listMem[listSize+i].dist = d_points[src].cosine(&d_points[listMem[listSize+i].id]);
      }
      }
    }
    __syncthreads();
    listSize += KVAL;
*/
    __syncthreads();

    sortListById<T,SUMTYPE,MAX_DIM,NUM_THREADS>(listMem, &listSize, &temp_storage);
    removeDuplicatesAndCompact<T,SUMTYPE,MAX_DIM,NUM_THREADS>(listMem, &listSize, &temp_storage, src, borderVals);
    __syncthreads();
    sortListByDist<T,SUMTYPE,MAX_DIM,NUM_THREADS>(listMem, &listSize, &temp_storage);
    __syncthreads();


//     Prune nearest RNG vectors and write them to d_graph
    shrinkListRNG_sequential<T,SUMTYPE,MAX_DIM,NUM_THREADS>(d_points, d_graph, src, listMem, listSize, KVAL, metric);

  }
}

/* Refine the KNN graph into a RNG graph using the GPU */
template<typename T, typename SUMTYPE, int MAX_DIM>
void refineGraphGPU(SPTAG::VectorIndex* index, Point<T,SUMTYPE,MAX_DIM>* d_points, int* d_graph, int dataSize, int KVAL, int candidatesPerVector, int refineDepth, int refines, int metric) {

  // Find candidates using the BKT on the CPU
  auto t1 = std::chrono::high_resolution_clock::now();
  int* candidates = (int*)malloc(dataSize*candidatesPerVector*sizeof(int));
  getCandidates<T>(index, dataSize, candidatesPerVector, candidates);

  // Copy candidate points to the GPU
  int* d_candidates;
  cudaMalloc(&d_candidates, dataSize*candidatesPerVector*sizeof(int));
  cudaMemcpy(d_candidates, candidates, dataSize*candidatesPerVector*sizeof(int), cudaMemcpyHostToDevice);

  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "find candidates time (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;

  // Use a number of batches of refinement to overlap CPU and GPU work (TODO)
  int NUM_BATCHES = 1;
  int batch_size = dataSize/NUM_BATCHES;

// Allocate scratch space memory to store large KNN lists before refining
  ListElt<SUMTYPE>* listMem;
  cudaMalloc(&listMem, TEST_BLOCKS*LISTCAP*sizeof(ListElt<SUMTYPE>));

  t1 = std::chrono::high_resolution_clock::now();
  for(int iter=0; iter < refines; iter++) {

    for(int i=0; i<NUM_BATCHES; i++) {
// Kernel that refines a batch of points' KNN neighbors into RNG neighbors
      refineBatch_kernel<T,SUMTYPE,MAX_DIM, TEST_THREADS><<<TEST_BLOCKS,TEST_THREADS>>>(d_points, batch_size, i*batch_size, d_graph, d_candidates, listMem, candidatesPerVector, KVAL, refineDepth, metric);
      cudaError_t status = cudaDeviceSynchronize();
      if(status != cudaSuccess) {
        printf("Refine error code:%d\n", status);
      }
    }
  }
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "GPU refine time (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;

  cudaFree(listMem);
  cudaFree(d_candidates);
  free(candidates);
}

#endif
