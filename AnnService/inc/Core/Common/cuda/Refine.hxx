/*
#include "ThreadHeap.hxx"
#include "TPtree.hxx"
#include "Distance.hxx"

#include<cub/cub.cuh>

#define LISTCAP 128
#define LISTSIZE 64

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

template<typename SUMTYPE, int NUM_REGS, int NUM_THREADS>
__device__ void loadRegisters(ListElt<SUMTYPE>* regs, ListElt<SUMTYPE>* listMem, int* listSize) {
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
__device__ void storeRegisters(ListElt<SUMTYPE>* regs, ListElt<SUMTYPE>* listMem, int* listSize) {
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

template<typename T, typename SUMTYPE, int NUM_THREADS>
__device__ void sortListById(ListElt<SUMTYPE>* listMem, int listSize, void* temp_storage) {

  ListElt<SUMTYPE> sortMem[LISTCAP/NUM_THREADS];
  int sortKeys[LISTCAP/NUM_THREADS];

// Sort list by ID to remove duplicates
  // Load list into registers to sort
  loadRegisters<SUMTYPE,LISTCAP/NUM_THREADS, NUM_THREADS>(sortMem, listMem, &listSize);

  for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
    sortKeys[i] = sortMem[i].id;
  }

  // Sort by ID in registers
  typedef cub::BlockRadixSort<int, NUM_THREADS, LISTCAP/NUM_THREADS, ListElt<SUMTYPE>> BlockRadixSort;
  BlockRadixSort(*(static_cast<typename BlockRadixSort::TempStorage*>(temp_storage))).SortBlockedToStriped(sortKeys, sortMem);

  __syncthreads();


  // Write back to list
  storeRegisters<SUMTYPE,LISTCAP/NUM_THREADS, NUM_THREADS>(sortMem, listMem, &listSize);
  __syncthreads();

}

template<typename T, typename SUMTYPE, int NUM_THREADS>
__device__ void sortListByDist(ListElt<SUMTYPE>* listMem, int listSize, void* temp_storage) {

  if(threadIdx.x==0)
    printf("regs per thread:%d\n", LISTCAP/NUM_THREADS);

  ListElt<SUMTYPE> sortMem[LISTCAP/NUM_THREADS];
  SUMTYPE sortKeys[LISTCAP/NUM_THREADS];

// Sort list by ID to remove duplicates
  // Load list into registers to sort
  loadRegisters<SUMTYPE,LISTCAP/NUM_THREADS, NUM_THREADS>(sortMem, listMem, &listSize);

  for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
    sortKeys[i] = sortMem[i].dist;
  }

  // Sort by ID in registers
  typedef cub::BlockRadixSort<SUMTYPE, NUM_THREADS, LISTCAP/NUM_THREADS, ListElt<SUMTYPE>> BlockRadixSort;
  BlockRadixSort(*(static_cast<typename BlockRadixSort::TempStorage*>(temp_storage))).SortBlockedToStriped(sortKeys, sortMem);
  __syncthreads();

  // Write back to list
  storeRegisters<SUMTYPE,LISTCAP/NUM_THREADS, NUM_THREADS>(sortMem, listMem, &listSize);
  __syncthreads();
}

// Remove duplicates and compact list with prefix sums
template<typename T, typename SUMTYPE, int NUM_THREADS>
__device__ void removeDuplicatesAndCompact(ListElt<SUMTYPE>* listMem, int* listSize, void *temp_storage) {
  if(threadIdx.x==0) {
    printf("elts per thread:%d\n", LISTCAP/NUM_THREADS);
  }

  ListElt<SUMTYPE> sortMem[LISTCAP/NUM_THREADS];
  int sortKeys[LISTCAP/NUM_THREADS];

  // Copy weather is duplicate or not into registers
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

  printf("%d: (%d,%d,%d), (%d,%d,%d), (%d,%d,%d), (%d,%d,%d)\n", threadIdx.x, 
      sortMem[0].id, sortMem[0].dist, sortKeys[0],
      sortMem[1].id, sortMem[1].dist, sortKeys[1],
      sortMem[2].id, sortMem[2].dist, sortKeys[2],
      sortMem[3].id, sortMem[3].dist, sortKeys[3]);

for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
	listMem[threadIdx.x*(LISTCAP/NUM_THREADS) + i] = sortMem[i];
//	listMem[threadIdx.x*(LISTCAP/NUM_THREADS) + i].checkedFlag = sortKeys[i];
}

  __syncthreads();
  typedef cub::BlockScan<int, NUM_THREADS> BlockScan;
  BlockScan(*(static_cast<typename BlockScan::TempStorage*>(temp_storage))).InclusiveSum(sortKeys, sortKeys);

  __syncthreads();
  printf("%d: (%d,%d,%d), (%d,%d,%d), (%d,%d,%d), (%d,%d,%d)\n", threadIdx.x, 
      sortMem[0].id, sortMem[0].dist, sortKeys[0],
      sortMem[1].id, sortMem[1].dist, sortKeys[1],
      sortMem[2].id, sortMem[2].dist, sortKeys[2],
      sortMem[3].id, sortMem[3].dist, sortKeys[3]);
__syncthreads();

  for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
    listMem[threadIdx.x*(LISTCAP/NUM_THREADS)+i] = sortMem[i];
  }

  // Share boarders of prefix sums
  __shared__ int borderVals[NUM_THREADS];
  borderVals[threadIdx.x] = sortKeys[LISTCAP/NUM_THREADS - 1];

  __syncthreads();

  if(threadIdx.x==0) {
    for(int i=0; i<*listSize; i++) {
      printf("%d, %d\n", listMem[i].id, listMem[i].dist);
    }
    printf("\n");
  }
__syncthreads();
  printf("%d: (%d,%d,%d), (%d,%d,%d), (%d,%d,%d), (%d,%d,%d) - %d\n", threadIdx.x, 
      sortMem[0].id, sortMem[0].dist, sortKeys[0],
      sortMem[1].id, sortMem[1].dist, sortKeys[1],
      sortMem[2].id, sortMem[2].dist, sortKeys[2],
      sortMem[3].id, sortMem[3].dist, sortKeys[3], borderVals[threadIdx.x-1]);
__syncthreads();

  if(threadIdx.x==0 || borderVals[threadIdx.x-1] != sortKeys[0]) {
    listMem[sortKeys[0]-1] = sortMem[0];
    printf("thread:%d, printing %d to location %d\n", threadIdx.x, sortMem[0].id, sortKeys[0]-1);
  }
else {
  printf("thread:%d, borderVals:%d, sortMem[0].id:%d\n", threadIdx.x, borderVals[threadIdx.x-1], sortMem[0].id);
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

*/
