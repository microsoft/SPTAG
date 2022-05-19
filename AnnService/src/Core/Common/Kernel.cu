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

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <typeinfo>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <type_traits>

//#include "inc/Core/Common/TruthSet.h"
#include "inc/Core/Common/cuda/params.h"
#include "inc/Core/Common/cuda/TPtree.hxx"

#include "inc/Core/Common/cuda/ThreadHeap.hxx"
#include "inc/Core/Common/cuda/log.hxx"
//#include "inc/Core/Common/cuda/KNN.hxx"

/*****************************************************************************************
* Count the number of points assigned to each leaf
*****************************************************************************************/
__global__ void count_leaf_sizes(LeafNode* leafs, int* node_ids, int N, int internal_nodes) {
    int leaf_id;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x) {
        leaf_id = node_ids[i] - internal_nodes;
        atomicAdd(&leafs[leaf_id].size, 1);
    }
}

/*****************************************************************************************
* Assign each point to a leaf node (based on its node_id when creating the tptree).  Also
* computes the size and offset of each leaf node for easy permutation.
*****************************************************************************************/
__global__ void assign_leaf_points(LeafNode* leafs, int* leaf_points, int* node_ids, int N, int internal_nodes) {
    int leaf_id;
    int idx;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x) {
        leaf_id = node_ids[i] - internal_nodes;
        idx = atomicAdd(&leafs[leaf_id].size, 1);
        leaf_points[idx + leafs[leaf_id].offset] = i;
    }
}


__global__ void assign_leaf_points_in_batch(LeafNode* leafs, int* leaf_points, int* node_ids, int N, int internal_nodes, int min_id, int max_id) {
    int leaf_id;
    int idx;
    for (int i = min_id + blockIdx.x*blockDim.x + threadIdx.x; i < max_id; i += blockDim.x*gridDim.x) {
        leaf_id = node_ids[i] - internal_nodes;
        idx = atomicAdd(&leafs[leaf_id].size, 1);
        leaf_points[idx + leafs[leaf_id].offset] = i;
    }
}

__global__ void assign_leaf_points_out_batch(LeafNode* leafs, int* leaf_points, int* node_ids, int N, int internal_nodes, int min_id, int max_id) {
    int leaf_id;
    int idx;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < min_id; i += blockDim.x*gridDim.x) {
        leaf_id = node_ids[i] - internal_nodes;
        idx = atomicAdd(&leafs[leaf_id].size, 1);
        leaf_points[idx + leafs[leaf_id].offset] = i;
    }

    for (int i = max_id + blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x) {
        leaf_id = node_ids[i] - internal_nodes;
        idx = atomicAdd(&leafs[leaf_id].size, 1);
        leaf_points[idx + leafs[leaf_id].offset] = i;
    }
}


//#define BAL 2 // Balance factor - will only rebalance nodes that are at least 2x larger than their sibling

// Computes the fraction of points that need to be moved from each unbalanced node on the level
__global__ void check_for_imbalance(int* node_ids, int* node_sizes, int nodes_on_level, int node_start, float* frac_to_move, int bal_factor) {
  int neighborId;
  for(int i=node_start + blockIdx.x*blockDim.x + threadIdx.x; i<node_start+nodes_on_level; i+=blockDim.x*gridDim.x) {
    frac_to_move[i] = 0.0;
    neighborId = (i-1) + 2*(i&1); // neighbor is either left or right of current
    if(node_sizes[i] > bal_factor*node_sizes[neighborId]) {
      frac_to_move[i] = ((float)node_sizes[i] - (((float)(node_sizes[i]+node_sizes[neighborId]))/2.0)) / (float)node_sizes[i];
    }
  }
}

// Initialize random number generator for each thread
__global__ void initialize_rands(curandState* states, int iter) {
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  curand_init(1234, id, iter, &states[id]);
}

// Randomly move points to sibling nodes based on the fraction that need to be moved out of unbalanced nodes
__global__ void rebalance_nodes(int* node_ids, int N, float* frac_to_move, curandState* states) {
  int neighborId;
  int threadId = blockIdx.x*blockDim.x+threadIdx.x;

  for(int i=threadId; i<N; i+=blockDim.x*gridDim.x) {
    if((frac_to_move[node_ids[i]] > 0.0) && (curand_uniform(&states[threadId]) < frac_to_move[node_ids[i]])) {
      neighborId = (node_ids[i]-1) + 2*(node_ids[i]&1); // Compute idx of left or right neighbor
      node_ids[i] = neighborId;
    }
  }
}

/************************************************************************************
 * Brute-force K nearest neighbor kernel using shared memory (transposed to avoid conflicts)
 * Each thread keeps a heap of K elements to determine K smallest distances found
 * VAR data - linear matrix fo data
 * VAR queries - linear matrix of query vectors
 * RET results - linear vector of K pairs for each query vector
************************************************************************************/
template<typename T, int Dim, int KVAL, int BLOCK_DIM, typename SUMTYPE>
__global__ void query_KNN(Point<T, SUMTYPE, Dim>* querySet, Point<T, SUMTYPE, Dim>* data, int dataSize, int idx_offset, int numQueries, DistPair<SUMTYPE>* results) {
    // Memory for a heap for each thread
    __shared__ ThreadHeap<T, SUMTYPE, Dim, BLOCK_DIM> heapMem;

    heapMem.initialize(results);

    DistPair<SUMTYPE> extra; // extra variable to store the largest distance/id for all KNN of the point

    // Memory used to store a query point for each thread
    __shared__ T transpose_mem[Dim * BLOCK_DIM];
    TransposePoint<T, Dim, BLOCK_DIM, SUMTYPE> query;  // Stores in strided memory to avoid bank conflicts
    query.setMem(&transpose_mem[threadIdx.x]);

#if LOG_LEVEL >= 5
    int dSize = sizeof(T);
#endif
    DLOG_DEBUG("Shared memory per block - Queries:%d, Heaps:%d\n", Dim * BLOCK_DIM * dSize, BLOCK_DIM * KVAL * 4);

    heapMem[threadIdx.x].initialize();

    SUMTYPE dist;
    // Loop through all query points
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numQueries; i += blockDim.x * gridDim.x) {

        heapMem[threadIdx.x].initialize();
        extra.dist = INFTY;
        query.loadPoint(data[i]); // Load into shared memory

        // Compare with all points in the dataset
        for (int j = 0; j < dataSize; j++) {
#if METRIC == 1 
            dist = query.cosine(&data[j]);
#else
            dist = query.l2(&data[j]);
#endif
            if (dist < extra.dist) {
                if (dist < heapMem[threadIdx.x].top()) {
                    extra.dist = heapMem[threadIdx.x].vals[0].dist;
                    extra.idx = heapMem[threadIdx.x].vals[0].idx + idx_offset;
                    //When recording the index, remember the off_set to discriminate different subvectorSets
                    heapMem[threadIdx.x].insert(dist, j + idx_offset);
                }
                else {
                    extra.dist = dist;
                    extra.idx = j + idx_offset;
                }
            }
        }

        // Write KNN to result list in sorted order
        results[(i + 1) * KVAL - 1].idx = extra.idx;
        results[(i + 1) * KVAL - 1].dist = extra.dist;
        for (int j = KVAL - 2; j >= 0; j--) {
            results[i * KVAL + j].idx = heapMem[threadIdx.x].vals[0].idx;
            results[i * KVAL + j].dist = heapMem[threadIdx.x].vals[0].dist;
            heapMem[threadIdx.x].vals[0].dist = -1;
            heapMem[threadIdx.x].heapify();

        }
    }
}