/*
#include "Refine.hxx"

#define T_BLOCKS 1
#define T_THREADS 64

__device__ void printList(ListElt<int>* list, int size) {
  if(threadIdx.x==0 && blockIdx.x==0) {
    printf("size:%d\n", size);
  for(int i=0; i<size; i++) {
    printf("%d, %d, %d\n", list[i].id, list[i].dist, list[i].checkedFlag);
  }
  printf("\n");
  }
  __syncthreads();
}

__global__ void test_removeDuplicates(ListElt<int>* listMem) {
  ListElt<int>* list = &listMem[blockIdx.x*LISTSIZE*2];

  typedef cub::BlockRadixSort<int, T_THREADS, LISTCAP/T_THREADS, ListElt<int>> BlockRadixSortT;
__shared__ typename BlockRadixSortT::TempStorage temp_storage;
__shared__ int border_vals[T_THREADS];


  for(int i=threadIdx.x; i<LISTSIZE; i+=blockDim.x) {
    list[i].id = i;
    list[i+LISTSIZE].id = i;
    list[i].dist = -i;
    list[i+LISTSIZE].dist = -i;
    list[i].checkedFlag=true;
    list[i+LISTSIZE].checkedFlag=false;
  }
  __syncthreads();


  int listSize = LISTSIZE*2;

  __syncthreads();
  printList(list, listSize);
  __syncthreads();

  sortListById<int, int, 0, T_THREADS>(list, &listSize, &temp_storage);

  removeDuplicatesAndCompact<int, int, 0, T_THREADS>(list, &listSize, &temp_storage, border_vals);


  printList(list, listSize);
}


int main(int argc, char* argv[]) {

  ListElt<int>* listMem;
  cudaMalloc(&listMem, T_BLOCKS*LISTSIZE*2*sizeof(ListElt<int>));

  test_removeDuplicates<<<T_BLOCKS, T_THREADS>>>(listMem);
  printf("Error: %d\n", cudaDeviceSynchronize());
}

*/
