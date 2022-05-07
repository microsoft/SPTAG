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

#include "inc/Core/Common/cuda/params.h"
#include "inc/Core/Common/cuda/TPtree.hxx"

/*****************************************************************************************
* Convert sums to means for each split key
*****************************************************************************************/
__global__ void compute_mean(KEYTYPE* split_keys, int* node_sizes, int num_nodes) {
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < num_nodes; i += blockDim.x*gridDim.x) {
        if (node_sizes[i] > 0) {
            split_keys[i] /= ((KEYTYPE)node_sizes[i]);
            node_sizes[i] = 0;
        }
    }
}


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


__global__ void print_level_device(int* node_sizes, float* split_keys, int level_size, LeafNode* leafs, int* leaf_points) {
for(int i=0; i<100; ++i) {
  printf("%d (%d), ", leafs[i].size, leafs[i].offset);
}
  printf("\n");
for(int i=0; i<10; ++i) {
  printf("%d, ", leaf_points[i]);
}

/*
  for(int i=0; i<level_size; i++) {
    printf("(%d) %0.2f, ", node_sizes[i], split_keys[i]);
  }
*/
  printf("\n");
}
