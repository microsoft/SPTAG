
#ifndef _SPTAG_COMMON_CUDA_REFINE_H
#define _SPTAG_COMMON_CUDA_REFINE_H

#include "../../VectorIndex.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "Distance.hxx"

#include <thrust/async/sort.h>
#include <cub/cub.cuh>
#include <chrono>


#define MAX_CHECK_COUNT 32 // Max number of neighbors checked during each step of refinement.

#define LISTCAP 2048 // Maximum size of buffer for each threadblock during refinemnt
#define LISTSIZE 1024 // Maximum size of nearest neighbors stored during refinement

#define REFINE_THREADS 64
#define REFINE_BLOCKS 1024

/*
using namespace SPTAG;
using namespace std;

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
      listMem[i*NUM_THREADS+threadIdx.x].dist = INFTY<SUMTYPE>();
    }
  }
}



template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__device__ void sortListById( ListElt<SUMTYPE>* listMem, int* listSize, void* temp_storage) {

   ListElt<SUMTYPE> sortMem[LISTCAP/NUM_THREADS];
  int sortKeys[LISTCAP/NUM_THREADS];

  loadRegisters<SUMTYPE,LISTCAP/NUM_THREADS, NUM_THREADS>(sortMem, listMem, listSize);

  for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
    sortKeys[i] = sortMem[i].id;
  }
  __syncthreads();
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

  loadRegisters<SUMTYPE,LISTCAP/NUM_THREADS, NUM_THREADS>(sortMem, listMem, listSize);

  for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
    sortKeys[i] = sortMem[i].dist;
  }

  typedef cub::BlockRadixSort<SUMTYPE, NUM_THREADS, LISTCAP/NUM_THREADS, ListElt<SUMTYPE>> BlockRadixSort;
  BlockRadixSort(*(static_cast<typename BlockRadixSort::TempStorage*>(temp_storage))).SortBlockedToStriped(sortKeys, sortMem);
  __syncthreads();

  storeRegisters<SUMTYPE,LISTCAP/NUM_THREADS, NUM_THREADS>(sortMem, listMem, listSize);
  __syncthreads();
}


template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__device__ void removeDuplicatesAndCompact( ListElt<SUMTYPE>* listMem, int* listSize, void *temp_storage, int* borderVals, int src) {

  ListElt<SUMTYPE> sortMem[LISTCAP/NUM_THREADS];
  int sortKeys[LISTCAP/NUM_THREADS];

  for(int i=threadIdx.x; i<*listSize-1 && listMem[i].id != INFTY<int>(); i+=NUM_THREADS) {
    for(int j=i+1; j < *listSize && listMem[i].id == listMem[j].id; j++) {
      listMem[i].checkedFlag = (listMem[i].checkedFlag || listMem[j].checkedFlag);
    }
  }

  __syncthreads();

  for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
    sortMem[i] = listMem[threadIdx.x*(LISTCAP/NUM_THREADS) + i];
  }

  if(threadIdx.x==0) {
    sortKeys[0] = 0;
  }
  else {
    sortKeys[0] = (sortMem[0].id != listMem[threadIdx.x*(LISTCAP/NUM_THREADS) - 1].id); 
  }
  for(int i=1; i<LISTCAP/NUM_THREADS; i++) {
    sortKeys[i] = (sortMem[i].id != sortMem[i-1].id);
  }

  typedef cub::BlockScan<int, NUM_THREADS> BlockScan;
  BlockScan(*(static_cast<typename BlockScan::TempStorage*>(temp_storage))).InclusiveSum(sortKeys, sortKeys);

  __syncthreads();
  for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
    listMem[threadIdx.x*(LISTCAP/NUM_THREADS)+i] = sortMem[i];
  }
  borderVals[threadIdx.x] = sortKeys[LISTCAP/NUM_THREADS - 1];

  __syncthreads();

  if(threadIdx.x==0 || borderVals[threadIdx.x-1] != sortKeys[0]) {
    listMem[sortKeys[0]] = sortMem[0];
  }
  for(int i=1; i<LISTCAP/NUM_THREADS; i++) {
    if(sortKeys[i] > sortKeys[i-1]) {
      listMem[sortKeys[i]] = sortMem[i];
    }
  }
  __syncthreads();
  *listSize = borderVals[NUM_THREADS-1]+1;
  __syncthreads();
}



template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__device__ void checkClosestNeighbors(Point<T,SUMTYPE,MAX_DIM>* d_points, int src, int* d_graph,  ListElt<SUMTYPE>* listMem, int* listSize, int KVAL, int metric) {

//  int max_check = min(MAX_CHECK_COUNT, (LISTCAP-*listSize)/KVAL);
  int max_check = (LISTCAP-*listSize)/KVAL;

  int check_count=0;
  int new_listSize = *listSize;

  for(int i=0; i<LISTSIZE && check_count < max_check; ++i) {
    if(!listMem[i].checkedFlag && listMem[i].id != INFTY<int>()) {

__syncthreads();
      for(int j=threadIdx.x; j<KVAL; j+=NUM_THREADS) {
        listMem[new_listSize + j].id = d_graph[listMem[i].id*KVAL + j];
        listMem[new_listSize + j].checkedFlag = false;

        listMem[new_listSize + j].dist = INFTY<SUMTYPE>();

        if(listMem[new_listSize+j].id == -1) {
          listMem[new_listSize + j].id = INFTY<int>();
          listMem[new_listSize + j].checkedFlag = true;
        }

      }
      check_count++;
      new_listSize+=KVAL;
      __syncthreads();
      listMem[i].checkedFlag=true;
    }
  }
  __syncthreads();
  *listSize = new_listSize;
  __syncthreads();
}


template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__device__ void shrinkListRNG_sequential(Point<T,SUMTYPE,MAX_DIM>* d_points, int* d_graph, int src, ListElt<SUMTYPE>* listMem, int listSize, int KVAL, int metric) {

  if(threadIdx.x==0) {
    int* nodes;
    int count=0;
    for(int j=0; j<listSize && j < KVAL; j++) {
      nodes = &d_graph[src*KVAL];
      ListElt<SUMTYPE> item = listMem[j];

      bool good = true;
      if(item.id == src || item.id == INFTY<int>()) good = false;

      for(int k=0; k<count && good; k++) {
        if(nodes[k] == -1) break;

	if(metric == 0) {
          if(d_points[nodes[k]].l2(&d_points[item.id]) <= item.dist) {
	    good = false;
            break;
	  }
        }
	if(metric == 1) {
          if(d_points[nodes[k]].cosine(&d_points[item.id]) <= item.dist) {
	    good = false;
            break;
	  }
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

    good = true;
    if(listMem[read_idx].id == INFTY<int>() || listMem[read_idx].id == src) {
        good = false;
    }

    __syncthreads();


    if(metric == 0) {
      for(int j=threadIdx.x; j<write_idx && good; j+=NUM_THREADS) {
        if(listMem[read_idx].dist >= d_points[listMem[read_idx].id].l2(&d_points[listMem[j].id])) {
          good=false;
        }
      }
    }
    else {
      for(int j=threadIdx.x; j<write_idx && good; j+=NUM_THREADS) {
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
//  __syncthreads();

  for(int i=threadIdx.x; i<write_idx; i+=NUM_THREADS) {
    d_graph[src*KVAL+i] = listMem[i].id;
    if(listMem[i].id == INFTY<int>())
      d_graph[src*KVAL+i] = -1;
  }
  for(int i=write_idx+threadIdx.x; i<KVAL; i+=NUM_THREADS) {
      d_graph[src*KVAL+i] = -1;
  }

  __syncthreads();
}


template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__global__ void refineBatch_kernel(Point<T,SUMTYPE,MAX_DIM>* d_points, int batchSize, int batchOffset, int* d_graph, int* candidates, ListElt<SUMTYPE>* listMemAll, int candidatesPerVector, int KVAL, int refineDepth, int metric) {

  // Offset to the memory allocated for this block's list
   ListElt<SUMTYPE>* listMem = &listMemAll[blockIdx.x*LISTCAP];

  __shared__ int listSize;
  __shared__ int borderVals[NUM_THREADS];
  typedef cub::BlockRadixSort<SUMTYPE, NUM_THREADS, LISTCAP/NUM_THREADS, ListElt<SUMTYPE>> BlockRadixSortT;
  __shared__ typename BlockRadixSortT::TempStorage temp_storage;

  for(int src=blockIdx.x+batchOffset; src<batchOffset+batchSize; src+=gridDim.x) {

//    src_point.id = d_points[src].id;
//    for(int i=threadIdx.x; i<MAX_DIM; i+=NUM_THREADS) {
//      src_point.coords[i] = d_points[src].coords[i];
//    }

    for(int i=threadIdx.x; i<LISTCAP;i+=blockDim.x)
      listMem[i].id = INFTY<int>();

    __syncthreads();

    // Place all candidates into the list of neighbors to check during refinement
    for(int i=threadIdx.x; i<candidatesPerVector; i+=NUM_THREADS) {
      listMem[i].id = candidates[src*candidatesPerVector+i];
      if(metric == 0) {
        listMem[i].dist = d_points[src].l2(&d_points[listMem[i].id]);
      }
      else {
        listMem[i].dist = d_points[src].cosine(&d_points[listMem[i].id]);
      }
      listMem[i].checkedFlag = false;
    }
    listSize=candidatesPerVector;
    __syncthreads();

    sortListByDist<T,SUMTYPE,MAX_DIM,NUM_THREADS>(listMem, &listSize, &temp_storage);

// Perform DFS refinement up to refineDepth distance
    for(int iter=0; iter < refineDepth; iter++) {
      listSize = min(LISTSIZE, listSize);
      __syncthreads();
      checkClosestNeighbors<T,SUMTYPE,MAX_DIM,NUM_THREADS>(d_points, src, d_graph, listMem, &listSize, KVAL, metric);
      sortListById<T,SUMTYPE,MAX_DIM,NUM_THREADS>(listMem, &listSize, &temp_storage);
      removeDuplicatesAndCompact<T,SUMTYPE,MAX_DIM,NUM_THREADS>(listMem, &listSize, &temp_storage, borderVals, src);

// Compute distance of all new unique neighbors
    if(metric == 0) {
      for(int i=threadIdx.x; i<listSize; i+=NUM_THREADS) {
        if(listMem[i].dist == INFTY<SUMTYPE>() && listMem[i].id != INFTY<int>()) {
          listMem[i].dist = d_points[src].l2(&d_points[listMem[i].id]);
        }
      }
    }
    else {
      for(int i=threadIdx.x; i<listSize; i+=NUM_THREADS) {
        if(listMem[i].dist == INFTY<SUMTYPE>() && listMem[i].id != INFTY<int>()) {
          listMem[i].dist = d_points[src].cosine(&d_points[listMem[i].id]);
        }
      }
    }

    __syncthreads();
      sortListByDist<T,SUMTYPE,MAX_DIM,NUM_THREADS>(listMem, &listSize, &temp_storage);
    }

//     Prune nearest RNG vectors and write them to d_graph
    shrinkListRNG<T,SUMTYPE,MAX_DIM,NUM_THREADS>(d_points, d_graph, src, listMem, listSize, KVAL, metric);
    __syncthreads();
  }

}

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
  LOG(SPTAG::Helper::LogLevel::LL_Info, "find candidates time (ms): %lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());

  // Use a number of batches of refinement to overlap CPU and GPU work (TODO)
  int NUM_BATCHES = 1;
  int batch_size = dataSize/NUM_BATCHES;

// Allocate scratch space memory to store large KNN lists before refining
  ListElt<SUMTYPE>* listMem;
  cudaMalloc(&listMem, REFINE_BLOCKS*LISTCAP*sizeof(ListElt<SUMTYPE>));

  t1 = std::chrono::high_resolution_clock::now();
  for(int iter=0; iter < refines; iter++) {

    for(int i=0; i<NUM_BATCHES; i++) {
// Kernel that refines a batch of points' KNN neighbors into RNG neighbors
      refineBatch_kernel<T,SUMTYPE,MAX_DIM, REFINE_THREADS><<<REFINE_BLOCKS,REFINE_THREADS>>>(d_points, batch_size, i*batch_size, d_graph, d_candidates, listMem, candidatesPerVector, KVAL, refineDepth, metric);
      cudaError_t status = cudaDeviceSynchronize();
      if(status != cudaSuccess) {
          LOG(SPTAG::Helper::LogLevel::LL_Error, "Refine error code:%d\n", status);
      }
    }

  }
  t2 = std::chrono::high_resolution_clock::now();
  LOG(SPTAG::Helper::LogLevel::LL_Info, "GPU refine time (ms): %lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());

  cudaFree(listMem);
  cudaFree(d_candidates);
  free(candidates);
}
*/

#endif
