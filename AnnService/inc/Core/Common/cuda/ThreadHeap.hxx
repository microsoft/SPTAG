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

#ifndef _SPTAG_COMMON_CUDA_THREADHEAP_H_
#define _SPTAG_COMMON_CUDA_THREADHEAP_H_

#include <float.h>

// Object used to store id,distance combination for KNN graph
class DistPair {
public:
    float dist;
    int idx;
};

// Swap the values of two DistPair objects
__forceinline__ __device__ void swap(DistPair* a, DistPair* b) {
    DistPair temp;
    temp.dist = a->dist;
    temp.idx = a->idx;
    a->dist = b->dist;
    a->idx = b->idx;
    b->dist = temp.dist;
    b->idx = temp.idx;

}


// Heap object.  Implements a max-heap of DistPair objects.  Stores a total of KVAL pairs
//   NOTE: Currently KVAL must be 2i-1 for some integer i since the heap must be complete
template<typename T, int BLOCK_DIM>
class ThreadHeap {
public:
    int count;
    DistPair* vals;
    DistPair temp;

    // Enforce max-heap property on the entires
    __forceinline__ __device__ void heapify() {
        int i = 0;
        int swapDest = 0;

        while (2 * i + 2 < count) {
            swapDest = 2 * i;
            swapDest += (vals[i].dist < vals[2 * i + 1].dist && vals[2 * i + 2].dist <= vals[2 * i + 1].dist);
            swapDest += 2 * (vals[i].dist < vals[2 * i + 2].dist && vals[2 * i + 1].dist < vals[2 * i + 2].dist);

            if (swapDest == 2 * i) return;

            swap(&vals[i], &vals[swapDest]);

            i = swapDest;
        }

    }

    // Initialize all nodes of the heap to +infinity
    __device__ void initialize(int KVAL, DistPair* v) {
        count = KVAL;
        vals = v;
        reset();
    }

    __device__ void reset() {
        for (int i = 0; i < count; i++) {
            vals[i].dist = FLT_MAX;
        }
    }

    // FOR DEBUGGING: Print heap out
    __device__ void printHeap() {

        int levelSize = 0;
        int offset = 0;
        for (int level = 0; offset <= count / 2; level++) {
            levelSize = pow(2, level);
            for (int i = 0; i < levelSize; i++) {
                printf("%0.3f   ", vals[offset + i].dist);
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

    __device__ void load_mem(T* data, int dim, int* mem, T* query, float(*f)(T* a, T* b, int d)) {
        for (int i = 0; i <= count - 1; i++) {
            vals[(count - i) - 1].idx = mem[i];
            if (vals[(count - i) - 1].idx == -1) {
                vals[(count - i) - 1].dist = FLT_MAX;
            }
            else {
                vals[(count - i) - 1].dist = f(query, data + mem[i] * (long long int)dim, dim);
            }
        }
    }

    // Return value of the root of the heap (largest value)
    __device__ float top() {
        return vals[0].dist;
    }

};

#endif
