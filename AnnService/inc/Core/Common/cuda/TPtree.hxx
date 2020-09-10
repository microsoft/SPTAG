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


#ifndef _SPTAG_COMMON_CUDA_TPTREE_H_
#define _SPTAG_COMMON_CUDA_TPTREE_H_

#include<queue>
#include <cuda.h>
#include "params.h"
#include "Distance.hxx"

template<typename T, typename KEY_T,typename SUMTYPE, int Dim>
class TPtree;

/************************************************************************************
 * Structure that defines the memory locations where points/ids are for a leaf node
 ************************************************************************************/
class LeafNode {
  public:
    int size;
    int offset;
};


/************************************************************************************
 * Updates the node association for every points from one level to the next 
 * i.e., point associated with node k will become associated with 2k+1 or 2k+2
 ************************************************************************************/
template<typename T, typename KEY_T, typename SUMTYPE, int Dim, int PART_DIMS>
__global__ void update_node_assignments(Point<T,SUMTYPE,Dim>* points, KEY_T* weights, int* partition_dims, int* node_ids, KEY_T* split_keys, int* node_sizes, int N);

/************************************************************************************
 * Determine the sizes (number of points in) each leaf node and sets leafs.size
 ************************************************************************************/
__global__ void count_leaf_sizes(LeafNode* leafs, int* node_ids, int N, int internal_nodes);


/************************************************************************************
 * Collect list of all point ids associated with a leaf and puts it in leaf_points array.
 * Also updates leafs.offset
 ************************************************************************************/
__global__ void assign_leaf_points(LeafNode* leafs, int* leaf_points, int* node_ids, int N, int internal_nodes);

__global__ void assign_leaf_points_in_batch(LeafNode* leafs, int* leaf_points, int* node_ids, int N, int internal_nodes, int min_id, int max_id);
__global__ void assign_leaf_points_out_batch(LeafNode* leafs, int* leaf_points, int* node_ids, int N, int internal_nodes, int min_id, int max_id);


/************************************************************************************
 * Set of functions to compute mean to pick dividing hyperplanes
 ************************************************************************************/
template<typename T, typename KEY_T, typename SUMTYPE, int Dim, int PART_DIMS>
__global__ void find_level_sum(Point<T,SUMTYPE,Dim>* points, KEY_T* weights, int* partition_dims, int* node_ids, KEY_T* split_keys, int* node_sizes, int N, int nodes_on_level);

//template<typename KEY_T>
//__global__ void compute_mean(KEY_T* split_keys, int* node_sizes, int num_nodes);

/*****************************************************************************************
* Convert sums to means for each split key
*****************************************************************************************/
template<typename KEY_T>
__global__ void compute_mean(KEY_T* split_keys, int* node_sizes, int num_nodes) {
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < num_nodes; i += blockDim.x*gridDim.x) {
        if (node_sizes[i] > 0) {
            split_keys[i] /= ((KEY_T)node_sizes[i]);
            node_sizes[i] = 0;
        }
    }
}

namespace {
__global__ void clean_leaf_size(LeafNode* leafs) {
    leafs[blockIdx.x].size=0;
}

__global__ void accumulate_leaf(LeafNode* leafs, int num_leaves) {
    leafs[0].offset=0;
    for(int i=1; i<num_leaves; ++i)
        leafs[i].offset = leafs[i-1].offset+leafs[i-1].size;
}
}

/************************************************************************************
 * Definition of the GPU TPtree structure. 
 * Only contains the nodes and hyperplane definitions that partition the data, as well
 * as indexes into the point array.  Does not contain the data itself.
 **********************************************************************************/
template<typename T, typename KEY_T,typename SUMTYPE, int Dim>
class TPtree {
  public:
// for each level of the tree, contains the dimensions and weights that defines the hyperplane
    int* partition_dims;
    KEY_T** weight_list;

// for each node, defines the value of the partitioning hyperplane.  Laid out in breadth-first order
    KEY_T* split_keys; 

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
    __host__ void initialize(int N_, int levels_) {

      long long int tree_mem=0;

      N = N_;
      levels = levels_;
      num_leaves = pow(2,levels);

      cudaMalloc(&node_ids, (N)*sizeof(int));
      cudaMemset(node_ids, 0, N*sizeof(int));
      tree_mem+= N*sizeof(int);

      num_nodes = (2*num_leaves - 1);

      int num_internals = num_nodes - num_leaves;

      tree_mem+=Dim*sizeof(int);

      // Allocate memory for TOT_PART_DIMS weights at each level
      weight_list = new KEY_T*[levels];
      for(int i=0; i<levels; ++i) {
        cudaMalloc(&weight_list[i], Dim*sizeof(KEY_T));
      }

      tree_mem+= levels*sizeof(int*) + levels*Dim*sizeof(KEY_T);

      tree_mem+= N*sizeof(int);
      cudaMalloc(&node_sizes, num_nodes*sizeof(int));
      cudaMemset(node_sizes, 0, num_nodes*sizeof(int));

      cudaMalloc(&split_keys, num_internals*sizeof(KEY_T));
      tree_mem+= num_nodes*sizeof(int) + num_internals*sizeof(KEY_T);

      cudaMalloc(&leafs, num_leaves*sizeof(LeafNode));
      tree_mem+=num_leaves*sizeof(LeafNode);

      cudaMalloc(&leaf_points, N*sizeof(int));
      tree_mem+=N*sizeof(int);

    }

    /***********************************************************
     *  Reset ids and sizes so that memory can be re-used for a new TPtree
     * *********************************************************/
    __host__ void reset() {

      cudaMemset(node_ids, 0, N*sizeof(int));
      cudaMemset(node_sizes, 0, num_nodes*sizeof(int));
      cudaMemset(split_keys, 0.0, num_nodes*sizeof(float));
      clean_leaf_size<<<num_leaves, 1>>>(leafs);
    }

    __host__ void destroy() {
      cudaFree(node_ids);
      for(int i=0; i<levels; ++i) {
        cudaFree(weight_list[i]);
      }
      delete []weight_list;
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
    __host__ void construct_tree(Point<T,SUMTYPE,Dim>* points, int min_id, int max_id) {

      int nodes_on_level=1;
      for(int i=0; i<levels; ++i) {

        find_level_sum<T,KEY_T,SUMTYPE,Dim,Dim><<<BLOCKS,THREADS>>>(points, weight_list[i], partition_dims, node_ids, split_keys, node_sizes, N, nodes_on_level);
        cudaDeviceSynchronize();



        compute_mean<KEY_T><<<BLOCKS,THREADS>>>(split_keys, node_sizes, num_nodes);

        cudaDeviceSynchronize();

        update_node_assignments<T,KEY_T,SUMTYPE,Dim,Dim><<<BLOCKS,THREADS>>>(points, weight_list[i], partition_dims, node_ids, split_keys, node_sizes, N);
        cudaDeviceSynchronize();

        nodes_on_level*=2;
      }
      count_leaf_sizes<<<BLOCKS,THREADS>>>(leafs, node_ids, N, num_nodes-num_leaves);

      accumulate_leaf<<<1, 1>>>(leafs, num_leaves);
      clean_leaf_size<<<num_leaves, 1>>>(leafs);

//      assign_leaf_points<<<BLOCKS,THREADS>>>(leafs, leaf_points, node_ids, N, num_nodes-num_leaves);
      assign_leaf_points_in_batch<<<BLOCKS,THREADS>>>(leafs, leaf_points, node_ids, N, num_nodes-num_leaves, min_id, max_id);
      cudaDeviceSynchronize();
      assign_leaf_points_out_batch<<<BLOCKS,THREADS>>>(leafs, leaf_points, node_ids, N, num_nodes-num_leaves, min_id, max_id);
    }

    /************************************************************************************
    // For debugging purposes
    ************************************************************************************/
    __host__ void print_tree(Point<T,SUMTYPE,Dim>* points) {
      std::vector<int> h_node_sizes(num_nodes);
      std::vector<KEY_T> h_split_keys(num_nodes - num_leaves);
      cudaMemcpyAsync(h_node_sizes.data(), node_sizes, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost, 0);
      cudaMemcpyAsync(h_split_keys.data(), split_keys, sizeof(KEY_T) * h_split_keys.size(), cudaMemcpyDeviceToHost, 0);
      cudaDeviceSynchronize();
      printf("nodes:%d, leaves:%d, levels:%d\n", num_nodes, num_leaves, levels);
      for(int i=0; i<levels; ++i) {
        for(int j=0; j<pow(2,i); ++j) {
          printf("(%d) %0.2f, ", h_node_sizes[(int)pow(2,i)+j-1], h_split_keys[(int)pow(2,i)+j-1]);
        }
        printf("\n");
      }
    }
};

template<typename T, typename KEY_T, typename SUMTYPE, int Dim>
__host__ void create_tptree(TPtree<T,KEY_T,SUMTYPE,Dim>* d_tree, Point<T,SUMTYPE,Dim>* points, int N, int MAX_LEVELS, int min_id, int max_id) {
  std::vector<std::vector<KEY_T>> h_weight_list(d_tree->levels, std::vector<KEY_T>(Dim));
  for(int j=0; j<Dim; ++j) {
    h_weight_list[0][j] = ((rand()%2)*2)-1;
  }
  for(int i=1; i<d_tree->levels; ++i) {
    for(int j=0; j<Dim; ++j) {
      h_weight_list[i][j] = h_weight_list[0][j] * (((rand()%2)*2)-1);
    }
  }
  for (int i = 0; i < d_tree->levels; ++i)
    cudaMemcpyAsync(d_tree->weight_list[i], h_weight_list[i].data(), h_weight_list[i].size() * sizeof(KEY_T), cudaMemcpyHostToDevice, 0);
  cudaDeviceSynchronize();
  
  d_tree->construct_tree(points, min_id, max_id);
}



/*****************************************************************************************
 * Helper function to calculated the porjected value of point onto the partitioning hyperplane
 *****************************************************************************************/
template<typename T, typename KEY_T, typename SUMTYPE, int Dim, int PART_DIMS>
__device__ KEY_T weighted_val(Point<T,SUMTYPE,Dim> point, KEY_T* weights, int* dims) {
  KEY_T val=0.0;
  for(int i=0; i<PART_DIMS; ++i) {
    val += (weights[i] * (KEY_T)point.coords[i]);
  }
  return val;
}
template<typename T, typename KEY_T, typename SUMTYPE, int Dim, int PART_DIMS>
__device__ KEY_T weighted_val(Point<uint8_t,SUMTYPE,Dim> point, KEY_T* weights, int* dims) {
  KEY_T val=0.0;
  for(int i=0; i<PART_DIMS/4; ++i) {
    val += (weights[i*4] * (unsigned(point.coords[i]) & 0x000000FF));
    val += (weights[i*4+1] * ((unsigned(point.coords[i]) & 0x0000FF00) >> 8));
    val += (weights[i*4+2] * ((unsigned(point.coords[i]) & 0x00FF0000) >> 16));
    val += (weights[i*4+3] * ((unsigned(point.coords[i]) & 0xFF000000) >> 24));
  }
  return val;
}

template<typename T, typename KEY_T, typename SUMTYPE, int Dim, int PART_DIMS>
__device__ KEY_T weighted_val(Point<int8_t,SUMTYPE,Dim> point, KEY_T* weights, int* dims) {

  KEY_T val=0.0;
  for(int i=0; i<PART_DIMS/4; ++i) {
    val += (KEY_T)(weights[i*4] * (int8_t)(point.coords[i] & 0x000000FF));
    val += (KEY_T)(weights[i*4+1] *(int8_t)((point.coords[i] & 0x0000FF00) >> 8));
    val += (KEY_T)(weights[i*4+2] *(int8_t)((point.coords[i] & 0x00FF0000) >> 16));
    val += (KEY_T)(weights[i*4+3] *(int8_t)((point.coords[i] & 0xFF000000) >> 24));
  }
  return val;
}


template<typename T, typename KEY_T, typename SUMTYPE, int Dim, int PART_DIMS>
__global__ void find_level_sum(Point<T,SUMTYPE,Dim>* points, KEY_T* weights, int* partition_dims, int* node_ids, KEY_T* split_keys, int* node_sizes, int N, int nodes_on_level) {
  KEY_T val=0;
  int size = min(N, nodes_on_level*SAMPLES);
  for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<size; i+=blockDim.x*gridDim.x) {
    val = weighted_val<T,KEY_T,SUMTYPE,Dim,PART_DIMS>(points[i], weights, partition_dims);
    atomicAdd(&split_keys[node_ids[i]], val);
    atomicAdd(&node_sizes[node_ids[i]], 1);
  }
}

/*****************************************************************************************
 * Assign each point to a node of the next level of the tree (either left child or right).
 *****************************************************************************************/
template<typename T, typename KEY_T, typename SUMTYPE, int Dim, int PART_DIMS>
__global__ void update_node_assignments(Point<T,SUMTYPE,Dim>* points, KEY_T* weights, int* partition_dims, int* node_ids, KEY_T* split_keys, int* node_sizes, int N) {
  
  for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<N; i+=blockDim.x*gridDim.x) {
    node_ids[i] = (2*node_ids[i])+1 + (weighted_val<T,KEY_T,SUMTYPE,Dim,PART_DIMS>(points[i],weights,partition_dims) > split_keys[node_ids[i]]);
  }
}

#endif
