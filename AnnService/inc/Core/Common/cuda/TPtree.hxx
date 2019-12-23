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

#ifndef _SPTAG_COMMON_CUDA_TPTREE_H_
#define _SPTAG_COMMON_CUDA_TPTREE_H_

#include <queue>

#include "params.h"

/************************************************************************************
 * Structure that defines the memory locations where points/ids are for a leaf node
 ************************************************************************************/
class LeafNode {
public:
    int size;
    int offset;
};


/************************************************************************************
 * Determine the sizes (number of points in) each leaf node and sets leafs.size
 ************************************************************************************/
__global__ void count_leaf_sizes(LeafNode* leafs, int* node_ids, int N, int internal_nodes);


/************************************************************************************
 * Collect list of all point ids associated with a leaf and puts it in leaf_points array.
 * Also updates leafs.offset
 ************************************************************************************/
__global__ void assign_leaf_points(LeafNode* leafs, int* leaf_points, int* node_ids, int N, int internal_nodes);


/************************************************************************************
 * Set of functions to compute variances and mean to pick dividing hyperplanes
 ************************************************************************************/
__global__ void compute_mean(int dim, float* split_keys, int* node_sizes, int num_nodes);


/************************************************************************************
* Updates the node association for every points from one level to the next
* i.e., point associated with node k will become associated with 2k+1 or 2k+2
************************************************************************************/
template<typename T>
__global__ void update_node_assignments(T* points, int dim, int PART_DIMS, float* weights, int* partition_dims, int* node_ids, float* split_keys, int* node_sizes, int N);

template<typename T>
__global__ void find_level_sum(T* points, int dim, int PART_DIMS, float* weights, int* partition_dims, int* node_ids, float* split_keys, int* node_sizes, int N);

template<typename T>
void compute_variances(T* points, int dim, float* variances, int N);

template<typename T>
void find_high_var_dims(T* points, int dim, int PART_DIMS, float* variances, int* dim_ids, int N);

template<typename T, int RAND_ITERS>
void generate_weight_set(T* points, int dim, int PART_DIMS, T* keys, int* dim_ids, float* best_weights, int N);


/************************************************************************************
 * Definition of the GPU TPtree structure.
 * Only contains the nodes and hyperplane definitions that partition the data, as well
 * as indexes into the point array.  Does not contain the data itself.
 **********************************************************************************/
template<typename T, int RAND_ITERS>
class TPtree {
public:
    int PART_DIMS;

    // for each level of the tree, contains the dimensions and weights that defines the hyperplane
    int* partition_dims;
    float** weight_list;

    // for each node, defines the value of the partitioning hyperplane.  Laid out in breadth-first order
    float* split_keys;

    int* node_ids; // For each point, store which node it belongs to (ends at id of leaf)
    int* node_sizes; // Stores the size (number of points) in each node

    int num_nodes;
    int num_leaves;
    int levels;
    int N;

    LeafNode* leafs; // size and offset of each leaf node

    int* leaf_points; // IDs of points in each leaf. Only needed if we dont permute.

    /************************************************************************************
     * Initialize the structure and allocated enough memory for everything
     **********************************************************************************/
    __host__ void initialize(int N_, int levels_, int part_dims_) {

        long long int tree_mem = 0;

        PART_DIMS = part_dims_;

        N = N_;
        levels = levels_;
        num_leaves = pow(2, levels);

        cudaMallocManaged(&node_ids, sizeof(int)*N);
        tree_mem += sizeof(int)*N;

        for (int i = 0; i < N; ++i) {
            node_ids[i] = 0;
        }
        num_nodes = (2 * num_leaves - 1);

        int num_internals = num_nodes - num_leaves;

        cudaMallocManaged(&partition_dims, sizeof(int)*PART_DIMS);
        tree_mem += sizeof(int)*PART_DIMS;

        // Allocate memory for TOT_PART_DIMS weights at each level
        cudaMallocManaged(&weight_list, levels * sizeof(float*));
        for (int i = 0; i < levels; ++i) {
            cudaMallocManaged(&weight_list[i], sizeof(float)*PART_DIMS);
        }

        tree_mem += levels * sizeof(float*) + sizeof(float)*levels*PART_DIMS;

        tree_mem += N * sizeof(int);
        cudaMallocManaged(&node_sizes, num_nodes * sizeof(int));
        cudaMallocManaged(&split_keys, num_internals * sizeof(float));
        tree_mem += num_nodes * sizeof(int) + num_internals * sizeof(float);

        for (int i = 0; i < num_nodes; ++i)
            node_sizes[i] = 0;
        cudaMallocManaged(&leafs, num_leaves * sizeof(LeafNode));
        tree_mem += num_leaves * sizeof(LeafNode);

        cudaMallocManaged(&leaf_points, N * sizeof(int));
        tree_mem += N * sizeof(int);

        LOG("Total memory of TPtree:%lld, mem per elt:%lld\n", tree_mem, (tree_mem / N));
    }

    /***********************************************************
     *  Reset ids and sizes so that memory can be re-used for a new TPtree
     * *********************************************************/
    __host__ void reset() {
        for (long long int i = 0; i < N; ++i) {
            node_ids[i] = 0;
        }
        for (int i = 0; i < num_nodes; ++i) {
            node_sizes[i] = 0;
        }
        for (int i = 0; i < num_leaves; ++i) {
            leafs[i].size = 0;
        }
    }

    __host__ void destroy() {
        cudaFree(node_ids);
        cudaFree(partition_dims);
        for (int i = 0; i < levels; ++i) {
            cudaFree(weight_list[i]);
        }
        cudaFree(weight_list);
        cudaFree(node_sizes);
        cudaFree(split_keys);
        cudaFree(leafs);
        cudaFree(leaf_points);
    }

    /************************************************************************************
     * Construct the tree.  ** Assumes tree has been initialized and allocated **
     * For each level of the tree, compute the mean for each node and set it as the split_key,
     * then compute, for each element, which child node it belongs to (storing in node_ids)
    ************************************************************************************/
    __host__ void construct_tree(T* points, int dims) {


        for (int i = 0; i < levels; ++i) {

            find_level_sum<T><<<BLOCKS,THREADS>>>(points, dims, PART_DIMS, weight_list[i], partition_dims, node_ids, split_keys, node_sizes, N);
            cudaDeviceSynchronize();
            compute_mean<<<BLOCKS,THREADS>>>(dims, split_keys, node_sizes, num_nodes);
            cudaDeviceSynchronize();
            update_node_assignments<T><<<BLOCKS,THREADS>>>(points, dims, PART_DIMS, weight_list[i], partition_dims, node_ids, split_keys, node_sizes, N);
            cudaDeviceSynchronize();
        }
        count_leaf_sizes<<<BLOCKS, THREADS>>>(leafs, node_ids, N, num_nodes - num_leaves);
        cudaDeviceSynchronize();

        leafs[0].offset = 0;
        for (int i = 1; i < num_leaves; ++i) {
            leafs[i].offset = leafs[i - 1].offset + leafs[i - 1].size;
        }
        for (int i = 0; i < num_leaves; ++i)
            leafs[i].size = 0;

        assign_leaf_points<<<BLOCKS,THREADS>>>(leafs, leaf_points, node_ids, N, num_nodes-num_leaves);
    }


    /************************************************************************************
    // For debugging purposes
    ************************************************************************************/
    __host__ void print_tree(T* points, int dims) {
        printf("nodes:%d, leaves:%d, levels:%d\n", num_nodes, num_leaves, levels);
        for (int i = 0; i < levels; ++i) {
            for (int j = 0; j < pow(2, i); ++j) {
                printf("(%d) %0.2f, ", (int)pow(2, i) + j - 1, split_keys[(int)pow(2, i) + j - 1]);
            }
            printf("\n");
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dims; ++j) {
                printf("%0.2f, ", (float)points[i*dims + j]);
            }
            printf(" - %d\n", node_ids[i]);
        }
    }
};

/****************************************************************************************
 * Create and construct the TP-tree based on the point set, using random weights
 * on the PART_DIMS dimensions with highest variance.
 *
 * **Assumes TPtree is allocated and initialized**
 *
 * RET: A TP-tree defined in unified memory with all split keys and node offsets computed.
 *
 * RET: The leaf nodes are either full of points in leaf_points or ids are given in leaf_points.
 * depending on if PERMUTE is set or not.
*****************************************************************************************/

template<typename T, int RAND_ITERS>
__host__ void create_tptree_device(TPtree<T, RAND_ITERS>* d_tree, T* points, int dim, int PART_DIMS, int N, int MAX_LEVELS) {

    float* variances;
    if (PART_DIMS < dim) {
        // Find dimensions with highest variances
        cudaMallocManaged(&variances, dim * sizeof(float));
        for (int i = 0; i < dim; ++i)
            variances[i] = 0.0;

        // Find dimensions with highest variance
        compute_variances<T>(points, dim, variances, N);
        find_high_var_dims<T>(points, dim, PART_DIMS, variances, d_tree->partition_dims, N);
    }
    else {
        // If PART_DIMS == dim, then all dimensions are selected and we don't need to find highest
        // variance dimensions.
        for (int i = 0; i < dim; ++i) {
            d_tree->partition_dims[i] = i;
        }
    }

    T* temp_keys;
    cudaMallocManaged(&temp_keys, N * sizeof(T));

    // If TPT_ITERS == 1, then don't need to find random weights with best variance (just pick first set)
#if TPT_ITERS > 1
    for (int i = 0; i < d_tree->levels; ++i) {
        generate_weight_set<T, RAND_ITERS>(points, dim, PART_DIMS, temp_keys, d_tree->partition_dims, d_tree->weight_list[i], N);
    }
#else
    for (int i = 0; i < d_tree->levels; ++i) {
        for (int j = 0; j < PART_DIMS; ++j) {
            d_tree->weight_list[i][j] = ((float)(rand()) / RAND_MAX) * 2 - 1.0;
        }
    }
#endif

    d_tree->construct_tree(points, dim);
    cudaFree(temp_keys);

    if (PART_DIMS < dim) cudaFree(variances);
}


/*****************************************************************************************
 * Helper function to calculated the porjected value of point onto the partitioning hyperplane
 *****************************************************************************************/
template<typename T>
__device__ float weighted_val(T* point, int dim, int PART_DIMS, float* weights, int* dims) {
    float val = 0.0;
    for (int i = 0; i < PART_DIMS; ++i) {
        val += (weights[i] * point[dims[i]]);
    }
    return val;
}

/*****************************************************************************************
 * Compute the sum of all points assigned to each node at a level
 *****************************************************************************************/
template<typename T>
__global__ void find_level_sum(T* points, int dim, int PART_DIMS, float* weights, int* partition_dims, int* node_ids, float* split_keys, int* node_sizes, int N) {
    float val = 0;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x) {
        val = weighted_val<T>(points + i * dim, dim, PART_DIMS, weights, partition_dims);
        atomicAdd(&split_keys[node_ids[i]], val);
        atomicAdd(&node_sizes[node_ids[i]], 1);
    }
}

/*****************************************************************************************
 * Assign each point to a node of the next level of the tree (either left child or right).
 *****************************************************************************************/
template<typename T>
__global__ void update_node_assignments(T* points, int dim, int PART_DIMS, float* weights, int* partition_dims, int* node_ids, float* split_keys, int* node_sizes, int N) {

    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x) {
        node_ids[i] = (2 * node_ids[i]) + 1 + (weighted_val<T>(points + i * dim, dim, PART_DIMS, weights, partition_dims) > split_keys[node_ids[i]]);
    }
}

/*****************************************************************************************
 * Compute the sum of each all points in each dimension (used to compute mean)
 *****************************************************************************************/
template<typename T>
__global__ void d_collect_all_sums(T* points, int dim, T* means, int N) {
    // Each block deals with 1 vector at a time
    for (int i = blockIdx.x; i < N; i += gridDim.x) {
        //    atomicAdd(&means[threadIdx.x], points[i*dim+threadIdx.x]);
    }
}

/*****************************************************************************************
 * Compute the variance of on each dimension
 *****************************************************************************************/
template<typename T>
__global__ void d_compute_all_variances(T* points, int dim, T* means, float* variances, int N) {
    float pointVar;
    for (int i = blockIdx.x; i < N; i += gridDim.x) {
        pointVar = (means[threadIdx.x] - points[i*dim + threadIdx.x])*(means[threadIdx.x] - points[i*dim + threadIdx.x]);
        atomicAdd(&variances[threadIdx.x], pointVar);
    }
}

/*****************************************************************************************
 * Convert sum into mean
 *****************************************************************************************/
template<typename T>
__global__ void mean_fix(int dim, T* means, float* variances, int N) {
    means[threadIdx.x] /= N;
}

/*****************************************************************************************
 * Kernel to compute the variance of each dimension of a point set
 *****************************************************************************************/
template<typename T>
void compute_variances(T* points, int dim, float* variances, int N) {
    T* means;
    cudaMalloc(&means, dim * sizeof(T));
    cudaMemset(&means, 0, dim * sizeof(T));


    d_collect_all_sums<T> << <BLOCKS, dim >> > (points, dim, means, N);
    cudaDeviceSynchronize();

    mean_fix<T> << <1, dim >> > (dim, means, variances, N);
    cudaDeviceSynchronize();

    d_compute_all_variances<T> << <BLOCKS, dim >> > (points, dim, means, variances, N);
    cudaDeviceSynchronize();

    // Don't really need to divide by N, since we just compare variances to get best dimensions to use
    for (int i = 0; i < dim; ++i) {
        variances[i] /= (float)(N - 1);
    }

    cudaFree(means);

}


/*****************************************************************************************
 * Get list of dimensions with highest variance
 *****************************************************************************************/
template<typename T>
void find_high_var_dims(T* points, int dim, int PART_DIMS, float* variances, int* dim_ids, int N) {

    for (int i = 0; i < dim; ++i)
        variances[i] = 0.0;

    compute_variances<T>(points, dim, variances, N);

    std::priority_queue<std::pair<float, int>> q;
    for (int i = 0; i < dim; ++i) {
        q.push(std::pair<float, int>(variances[i], i));
    }
    for (int i = 0; i < PART_DIMS; ++i) {
        dim_ids[i] = q.top().second;
        q.pop();
    }
}

/*****************************************************************************************
 * Fill "keys" variable with projected value of each point using the given weights
 * also computes the mean.
 *****************************************************************************************/
template<typename T>
__global__ void fill_keys_from_weights(T* points, int dim, int PART_DIMS, T* keys, int* dim_ids, float* weights, float* mean, int N) {
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x) {
        keys[i] = 0.0;
        for (int j = 0; j < PART_DIMS; ++j) {
            keys[i] += (weights[j] * (points[i*dim + dim_ids[j]]));
        }
        atomicAdd(mean, keys[i]);
    }
}

/*****************************************************************************************
 * Given keys and mean already calculated, compute variance.
 *****************************************************************************************/
template<typename T>
__global__ void compute_weight_variance(T* keys, float* mean, float* variance, int N) {
    float val;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x) {
        val = (*mean - keys[i])*(*mean - keys[i]);
        atomicAdd(variance, val);
    }
}

/*****************************************************************************************
 * Generate a set of random weights with best variance.
 * RET: PART_DIMS number of weights where the linear combination yields the highest variance,
 * compared with RAND_ITERS other randomly selected weights.
 *****************************************************************************************/
template<typename T, int RAND_ITERS>
void generate_weight_set(T* points, int dim, int PART_DIMS, T* keys, int* dim_ids, float* best_weights, int N) {
    float best_variance = 0.0;
    best_variance = 0.0;

    float* temp_variance;
    cudaMallocManaged(&temp_variance, 1);
    cudaMallocManaged(&temp_variance, sizeof(float));
    float* temp_weights;
    cudaMallocManaged(&temp_weights, PART_DIMS * sizeof(float));
    float* mean;
    cudaMallocManaged(&mean, sizeof(float));


    for (int i = 0; i < RAND_ITERS; ++i) {
        *temp_variance = 0.0;
        *mean = 0.0;

        for (int j = 0; j < PART_DIMS; ++j) {
            temp_weights[j] = ((float)(rand()) / RAND_MAX) * 2 - 1.0;
        }

        fill_keys_from_weights<T> << <BLOCKS, THREADS >> > (points, dim, PART_DIMS, keys, dim_ids, temp_weights, mean, N);
        cudaDeviceSynchronize();

        *mean = *mean / (float)N;

        compute_weight_variance<T> << <BLOCKS, THREADS >> > (keys, mean, temp_variance, N);
        cudaDeviceSynchronize();
        *temp_variance = *temp_variance / (N - 1);

        if (*temp_variance > best_variance) {
            best_variance = *temp_variance;
            for (int j = 0; j < PART_DIMS; ++j) {
                best_weights[j] = temp_weights[j];
            }
        }
    }

    fill_keys_from_weights<T> << <BLOCKS, THREADS >> > (points, dim, PART_DIMS, keys, dim_ids, best_weights, mean, N);

    cudaFree(temp_variance);
    cudaFree(mean);
    cudaFree(temp_weights);
}

#endif