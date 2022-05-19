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
 * Licensed under the MIT License
 */

#ifndef _SPTAG_COMMON_CUDA_THREADHEAP_H_
#define _SPTAG_COMMON_CUDA_THREADHEAP_H_

#include <stdio.h>
#include "Distance.hxx"


// Object used to store id,distance combination for KNN graph
template<typename SUMTYPE>
class DistPair {
  public:
    SUMTYPE dist;
    int idx;

  __device__ __host__ DistPair& operator=( const DistPair& other ) {
    dist = other.dist;
    idx = other.idx;
    return *this;
  }
};

// Swap the values of two DistPair<SUMTYPE> objects
template<typename SUMTYPE>
__device__ void swap(DistPair<SUMTYPE>* a, DistPair<SUMTYPE>* b) {
        DistPair<SUMTYPE> temp;
        temp.dist = a->dist;
        temp.idx = a->idx;
        a->dist = b->dist;
        a->idx=b->idx;
        b->dist=temp.dist;
        b->idx=temp.idx;

}


// Heap object.  Implements a max-heap of DistPair<SUMTYPE> objects.  Stores a total of KVAL pairs
//   NOTE: Currently KVAL must be 2i-1 for some integer i since the heap must be complete
template<typename T, typename SUMTYPE, int Dim, int BLOCK_DIM>
class ThreadHeap {
  public:
    int KVAL;
    DistPair<SUMTYPE>* vals;
    DistPair<SUMTYPE> temp;
  
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

    __forceinline__ __device__ void heapifyAt(int idx) {
      int i=idx;
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

    __device__ void reset() {
      for(int i=0; i<KVAL; i++) {
        vals[i].dist = INFTY<SUMTYPE>();
      }
    }


    __device__ void initialize(DistPair<SUMTYPE>* v, int _kval) {
      vals = v;
      KVAL = _kval;
      reset();
    }



    // Initialize all nodes of the heap to +infinity
    __device__ void initialize() {
      for(int i=0; i<KVAL; i++) {
        vals[i].dist = INFTY<SUMTYPE>();
      }
    }

    __device__ void write_to_gmem(int* gmem) {
      for(int i=0; i<KVAL; i++) {
        gmem[i] = vals[i].idx;
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

    // Replace a specific element in the heap (and maintain heap properties)
    __device__ void insertAt(float newDist, int newIdx, int idx) {
      vals[idx].dist = newDist;
      vals[idx].idx = newIdx;

      heapifyAt(idx);
    }

/*
    // Load a sorted set of vectors into the heap
    __device__ void load_mem_sorted(Point<T, SUMTYPE,Dim>* data, int* mem, Point<T,SUMTYPE,Dim> query, int metric) {
      for(int i=0; i<=KVAL-1; i++) {
        vals[(KVAL-i)-1].idx = mem[i];
        if(vals[(KVAL-i)-1].idx == -1) {
          vals[(KVAL-i)-1].dist = INFTY<SUMTYPE>();
        }
        else {
          if(metric == 0) {
            vals[(KVAL-i)-1].dist = query.l2(&data[mem[i]]);
          }
          else if(metric == 1) {
            vals[(KVAL-i)-1].dist = query.cosine(&data[mem[i]]);
          }
        }
      }
    }

    // Return value of the root of the heap (largest value)
    __device__ SUMTYPE top() {
      return vals[0].dist;
    }
*/
};

#endif
