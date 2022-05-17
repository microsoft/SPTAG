/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
 */

#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <assert.h>
#include<cuda.h>
//#include <float.h>
#include "gpu_params.h"
#include "Point.hxx"


// Object used to store id,distance combination for KNN graph
class DistPair {
  public:
    SUMTYPE dist;
    int idx;
};

// Swap the values of two DistPair objects
__forceinline__ __device__ void swap(DistPair* a, DistPair* b) {
        DistPair temp;
        temp.dist = a->dist;
        temp.idx = a->idx;
        a->dist = b->dist;
        a->idx=b->idx;
        b->dist=temp.dist;
        b->idx=temp.idx;

}


// Heap object.  Implements a max-heap of DistPair objects.  Stores a total of KVAL pairs
//   NOTE: Currently KVAL must be 2i-1 for some integer i since the heap must be complete
template<typename T, int Dim, int KVAL, int BLOCK_DIM>
class ThreadHeap {
  public:
    DistPair vals[KVAL];
    DistPair temp;
  
    // Enforce max-heap property on the entires
    __forceinline__ __device__ void heapify() {
      int i=0;
      int swapDest=0;

      while(2*i+2 < KVAL) {
        swapDest = 2*i;
        swapDest += (vals[i].dist < vals[2*i+1].dist && vals[2*i+2].dist <= vals[2*i+1].dist);
        swapDest += 2*(vals[i].dist < vals[2*i+2].dist && vals[2*i+1].dist < vals[2*i+2].dist);

        if(swapDest == 2*i) return;

        swap(&vals[i], &vals[swapDest]);

        i = swapDest;
      }

    }

    // Initialize all nodes of the heap to +infinity
    __device__ void initialize() {
      for(int i=0; i<KVAL; i++) {
        vals[i].dist = INFTY;
      }
    }


    // FOR DEBUGGING: Print heap out
    __device__ void printHeap() {

      int levelSize=0;
      int offset=0;
      for(int level=0; offset<=KVAL/2; level++) {
        levelSize = pow(2,level);
        for(int i=0; i<levelSize; i++) {
          printf("%0.3f   ", vals[offset+i].dist);
        }
        offset += levelSize;
        printf("\n");
      } 
    }


    // Replace the root of the heap with new pair
    __device__ void insert(float newDist, int newIdx) {
      vals[0].dist = newDist;
      vals[0].idx = newIdx;

      heapify();
    }

    __device__ void load_mem(Point<T, Dim>* data, int* mem, TransposePoint<T,Dim,BLOCK_DIM> query) {
      for(int i=0; i<=KVAL-1; i++) {
        vals[(KVAL-i)-1].idx = mem[i];
        if(vals[(KVAL-i)-1].idx == -1) {
          vals[(KVAL-i)-1].dist = INFTY;
        }
        else {
#if METRIC == 1
            vals[(KVAL-i)-1].dist = query.cosine(&data[mem[i]]);
#else
            //vals[(KVAL-i)-1].dist = query.l2_ILP(&data[mem[i]]);
            vals[(KVAL-i)-1].dist = query.l2(&data[mem[i]]);
#endif
        }
      }
    }

    __device__ void load_mem_regs(Point<T, Dim>* data, int* mem, T* query) {
      for(int i=0; i<=KVAL-1; i++) {
        vals[(KVAL-i)-1].idx = mem[i];
        if(vals[(KVAL-i)-1].idx == -1) {
          vals[(KVAL-i)-1].dist = INFTY;
        }
        else
          vals[(KVAL-i)-1].dist = l2_ILP(query, &data[mem[i]]);
      }
    }

    // Return value of the root of the heap (largest value)
    __device__ SUMTYPE top() {
      return vals[0].dist;
    }

};

/************************************************************************************************
* DEPRICATED BELOW - used for merging KNN graphs, not needed with reduced memory footprint code 
************************************************************************************************/
__device__ void cmp_swap(DistPair* a, DistPair* b) {
  bool check=(a->dist > b->dist);
  __syncthreads();
  if(check) {
    swap(a, b);
  }
}

template<int KVAL>
__device__ void merge_lists(DistPair* a, DistPair* b) {
  int th = threadIdx.x % 32;
  int id=th;

  cmp_swap(&a[id], &b[KVAL-id-1]);

  /*
  if(id >= 16) {
    a = b;
    th = th % 16;
  }
  */

  // Only need to sort the "a" list with 16 threads
  if(id < 16 ) {
  for(int dist=16; dist >=1; dist /= 2) {
    id = th + dist*(th / dist);
//    if(id > dist)
    cmp_swap(&a[id], &a[id+dist]);
  }
  }
  
}
