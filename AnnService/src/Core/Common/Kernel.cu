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

/*****************************************************************************************
* Convert sums to means for each split key
*****************************************************************************************/
__global__ void compute_mean(int dim, float* split_keys, int* node_sizes, int num_nodes) {
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < num_nodes; i += blockDim.x*gridDim.x) {
        if (node_sizes[i] > 0) {
            split_keys[i] /= ((float)node_sizes[i]);
            node_sizes[i] = 0;
        }
    }
}
