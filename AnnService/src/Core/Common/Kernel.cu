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

#include "inc/Core/Common/cuda/params.h"
#include "inc/Core/Common/cuda/TPtree.hxx"

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

