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

#include<vector>
#include<queue>

#include <cuda.h>
#include <thrust/sort.h>

#include "gpu_params.h"
#include "gpu_KNN.hxx"
#include "gpu_ThreadHeap.hxx"
#include "Point.hxx"

__forceinline__ __device__ void swap_result(DistPair* a, DistPair* b);

template<typename T, typename KEY_T, int Dim, int PART_DIMS, int RAND_ITERS>
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
template<typename T, typename KEY_T, int Dim, int PART_DIMS>
__global__ void update_node_assignments(Point<T,Dim>* points, float* weights, int* partition_dims, int* node_ids, KEY_T* split_keys, int* node_sizes, int N);

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
template<typename T, typename KEY_T, int Dim, int PART_DIMS>
__global__ void find_level_sum(Point<T,Dim>* points, float* weights, int* node_ids, T* split_keys, int* node_sizes, int N);

template<typename KEY_T, int Dim>
__global__ void compute_mean(KEY_T* split_keys, int* node_sizes, int num_nodes);

template<typename T, typename KEY_T, int Dim, int PART_DIMS>
__global__ void estimate_median(Point<T,Dim>* points, float* weights, int* node_ids, T* split_keys, int* node_sizes, int N);

template<typename T, int Dim>
void compute_variances(Point<T, Dim>* points, float* variances, int N);

template<typename T, int Dim, int PART_DIMS>
void find_high_var_dims(Point<T,Dim>* points, float* variances, int* dim_ids, int N);

template<typename T, int Dim, int PART_DIMS, int RAND_ITERS>
void generate_weight_set(Point<T,Dim>* points, T* keys, int* dim_ids, float* best_weights, int N);


/************************************************************************************
 * Definition of the GPU TPtree structure. 
 * Only contains the nodes and hyperplane definitions that partition the data, as well
 * as indexes into the point array.  Does not contain the data itself.
 **********************************************************************************/
template<typename T, typename KEY_T, int Dim, int PART_DIMS, int RAND_ITERS>
class TPtree {
  public:
// for each level of the tree, contains the dimensions and weights that defines the hyperplane
    int* partition_dims;
    float** weight_list;

// for each node, defines the value of the partitioning hyperplane.  Laid out in breadth-first order
    KEY_T* split_keys; 

    int* node_ids; // For each point, store which node it belongs to (ends at id of leaf)
    int* node_sizes; // Stores the size (number of points) in each node

    int num_nodes; 
    int num_leaves;
    int levels;
    int N;

    LeafNode* leafs; // size and offset of each leaf node

#if PERMUTE == 1
    Point<T,Dim>* leaf_points; // Only needed if we want to permute points into contigious lists at end
#else
    int* leaf_points; // IDs of points in each leaf. Only needed if we dont permute.
#endif


    /************************************************************************************
     * Initialize the structure and allocated enough memory for everything
     **********************************************************************************/
    __host__ void initialize(int N_, int levels_, Point<T,Dim>* points) {

      long long int tree_mem=0;

      N = N_;
      levels = levels_;
      num_leaves = pow(2,levels);

      cudaMallocManaged(&node_ids, (N)*sizeof(int));
      tree_mem+= N*sizeof(int);

      for(int i=0; i<N; ++i) {
        node_ids[i]=0;
      }
      num_nodes = (2*num_leaves - 1);

      int num_internals = num_nodes - num_leaves;

      cudaMallocManaged(&partition_dims, PART_DIMS*sizeof(int));

      tree_mem+=PART_DIMS*sizeof(int);

      // Allocate memory for TOT_PART_DIMS weights at each level
      cudaMallocManaged(&weight_list, levels*sizeof(float*));
      for(int i=0; i<levels; ++i) {
        cudaMallocManaged(&weight_list[i], PART_DIMS*sizeof(float));
      }

      tree_mem+= levels*sizeof(float*) + levels*PART_DIMS*sizeof(float);

      tree_mem+= N*sizeof(int);
      cudaMallocManaged(&node_sizes, num_nodes*sizeof(int));
      cudaMallocManaged(&split_keys, num_internals*sizeof(KEY_T));
      tree_mem+= num_nodes*sizeof(int) + num_internals*sizeof(KEY_T);

      for(int i=0; i<num_nodes; ++i)
        node_sizes[i]=0;
      cudaMallocManaged(&leafs, num_leaves*sizeof(LeafNode));
      tree_mem+=num_leaves*sizeof(LeafNode);

#if PERMUTE == 1
      cudaMallocManaged(&leaf_points, N*sizeof(Point<T,Dim>));
      tree_mem+=N*sizeof(Point<T,Dim>);
#else 
      cudaMallocManaged(&leaf_points, N*sizeof(int));
      tree_mem+=N*sizeof(int);
#endif

      LOG_INFO("Total memory of TPtree:%lld, mem per elt:%lld\n", tree_mem, tree_mem/N);
    }

    /***********************************************************
     *  Reset ids and sizes so that memory can be re-used for a new TPtree
     * *********************************************************/
    __host__ void reset() {
      for(long long int i=0; i<N; ++i) {
        node_ids[i]=0;
      }
      for(int i=0; i<num_nodes; ++i) {
        node_sizes[i]=0;
      }
      for(int i=0; i<num_leaves; ++i) {
        leafs[i].size=0;
      }
    }

    __host__ void destroy() {
      cudaFree(node_ids);
      cudaFree(partition_dims);
      for(int i=0; i<levels; ++i) {
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
    __host__ void construct_tree(Point<T,Dim>* points) {


      for(int i=0; i<levels; ++i) {
        
        find_level_sum<T,KEY_T,Dim,PART_DIMS><<<BLOCKS,THREADS>>>(points, weight_list[i], partition_dims, node_ids, split_keys, node_sizes, N);
        cudaDeviceSynchronize();
        compute_mean<KEY_T,Dim><<<BLOCKS,THREADS>>>(split_keys, node_sizes, num_nodes);
        
//        estimate_median<KEY_T,Dim><<<BLOCKS,THREADS>>>(points, weight_list[i], partition_dims, node_ids, split_keys, node_sizes, N);
        cudaDeviceSynchronize();
        update_node_assignments<T,KEY_T,Dim,PART_DIMS><<<BLOCKS,THREADS>>>(points, weight_list[i], partition_dims, node_ids, split_keys, node_sizes, N);
        cudaDeviceSynchronize();
      }
      count_leaf_sizes<<<BLOCKS,THREADS>>>(leafs, node_ids, N, num_nodes-num_leaves);
      cudaDeviceSynchronize();

      leafs[0].offset=0;
      for(int i=1; i<num_leaves; ++i) {
        leafs[i].offset = leafs[i-1].offset+leafs[i-1].size;
      } 
      for(int i=0; i<num_leaves; ++i)
        leafs[i].size=0;


#if PERMUTE == 1
      // If we want to have leafs permuted into contigious order
        permute_leaf_points<<<BLOCKS,THREADS>>>(leafs, points, leaf_points, node_ids, N, num_nodes-num_leaves);
#else
      assign_leaf_points<<<BLOCKS,THREADS>>>(leafs, leaf_points, node_ids, N, num_nodes-num_leaves);
#endif
    }


    /************************************************************************************
    // For debugging purposes
    ************************************************************************************/
    __host__ void print_tree(Point<T,Dim>* points) {
      printf("nodes:%d, leaves:%d, levels:%d\n", num_nodes, num_leaves, levels);
      for(int i=0; i<levels; ++i) {
//        printf("dim:%d - ", dim_list[i]);
        for(int j=0; j<pow(2,i); ++j) {
          printf("(%d) %0.2f, ", (int)pow(2,i)+j-1, split_keys[(int)pow(2,i)+j-1]);
        }
        printf("\n");
      }
      
      for(int i=0; i<N; ++i) {
        for(int j=0; j<Dim; ++j) {
          printf("%0.2f, ", points[i].coords[j]);
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

template<typename T, typename KEY_T, int Dim, int PART_DIMS, int RAND_ITERS>
__host__ void create_tptree_device(TPtree<T,KEY_T,Dim,PART_DIMS,RAND_ITERS>* d_tree, Point<T,Dim>* points, int N, int MAX_LEVELS) {

#if TPT_PART_DIMS < D
  // Find dimensions with highest variances
  float* variances;
  cudaMallocManaged(&variances, Dim*sizeof(float));
  for(int i=0; i<Dim; ++i)
    variances[i]=0.0;

  // Find dimensions with highest variance
  compute_variances<T, Dim>(points, variances, N);
  find_high_var_dims<T,Dim,PART_DIMS>(points, variances, d_tree->partition_dims, N);
#else
  // If TPT_PART_DIMS == D, then all dimensions are selected and we don't need to find highest
  // variance dimensions.
  for(int i=0; i<Dim; ++i) {
    d_tree->partition_dims[i]=i;
  }
#endif

  T* temp_keys;
  cudaMallocManaged(&temp_keys, N*sizeof(T));

  // If TPT_ITERS == 1, then don't need to find random weights with best variance (just pick first set)
#if TPT_ITERS > 1
  for(int i=0; i<d_tree->levels; ++i) {
    generate_weight_set<T,Dim,PART_DIMS,RAND_ITERS>(points, temp_keys, d_tree->partition_dims, d_tree->weight_list[i], N);
  }
#else

  for(int i=0; i<d_tree->levels; ++i) {
    for(int j=0; j<PART_DIMS; ++j) {
      d_tree->weight_list[i][j] = ((float) (rand()) / RAND_MAX)*2 - 1.0;
    }
  }
#endif

  d_tree->construct_tree(points);
  cudaFree(temp_keys);
  
#if TPT_PART_DIMS < D
  cudaFree(variances);
#endif
}


/*****************************************************************************************
 * Helper function to calculated the porjected value of point onto the partitioning hyperplane
 *****************************************************************************************/
template<typename T, int Dim, int PART_DIMS>
__device__ float weighted_val(Point<T,Dim> point, float* weights, int* dims) {
  float val=0.0;
  for(int i=0; i<PART_DIMS; ++i) {
    val += (weights[i] * point.coords[dims[i]]);
  }
  return val;
}

/*****************************************************************************************
 * Compute the sum of all points assigned to each node at a level
 *****************************************************************************************/
template<typename T, typename KEY_T, int Dim, int PART_DIMS>
__global__ void find_level_sum(Point<T,Dim>* points, float* weights, int* partition_dims, int* node_ids, KEY_T* split_keys, int* node_sizes, int N) {
  KEY_T val=0;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<N; i+=blockDim.x*gridDim.x) {
    val = weighted_val<T,Dim,PART_DIMS>(points[i], weights, partition_dims);
    atomicAdd(&split_keys[node_ids[i]], val);
    atomicAdd(&node_sizes[node_ids[i]], 1);
  }
}

/*****************************************************************************************
 * Convert sums to means for each split key
 *****************************************************************************************/
template<typename KEY_T, int Dim>
__global__ void compute_mean(KEY_T* split_keys, int* node_sizes, int num_nodes) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<num_nodes; i+=blockDim.x*gridDim.x) {
    if(node_sizes[i]>0) {
      split_keys[i] /= ((KEY_T)node_sizes[i]);
      node_sizes[i]=0;
    }
  }
}

/*****************************************************************************************
 * Convert sums to means for each split key
 *****************************************************************************************/
template<typename T, typename KEY_T, int Dim, int PART_DIMS>
__global__ void estimate_median(Point<T,Dim>* points, float* weights, int* node_ids, T* split_keys, int* node_sizes, int N) {
  
/*
  for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<num_nodes; i+=blockDim.x*gridDim.x) {
    if(node_sizes[i]>0) {
      split_keys[i] /= ((KEY_T)node_sizes[i]*MEAN_SKEW);
      node_sizes[i]=0;
    }
  }
  */
}

/*****************************************************************************************
 * Assign each point to a node of the next level of the tree (either left child or right).
 *****************************************************************************************/
template<typename T, typename KEY_T, int Dim, int PART_DIMS>
__global__ void update_node_assignments(Point<T,Dim>* points, float* weights, int* partition_dims, int* node_ids, KEY_T* split_keys, int* node_sizes, int N) {
  
  for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<N; i+=blockDim.x*gridDim.x) {
    node_ids[i] = (2*node_ids[i])+1 + (weighted_val<T,Dim,PART_DIMS>(points[i],weights,partition_dims) > split_keys[node_ids[i]]);
//    atomicAdd(&node_sizes[node_ids[i]], 1);
  }
}

/*****************************************************************************************
 * Count the number of points assigned to each leaf
 *****************************************************************************************/
__global__ void count_leaf_sizes(LeafNode* leafs, int* node_ids, int N, int internal_nodes) {
  int leaf_id;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<N; i+=blockDim.x*gridDim.x) {
    leaf_id = node_ids[i] - internal_nodes;
    atomicAdd(&leafs[leaf_id].size,1);
  }
}

/*****************************************************************************************
 * Copy all points from point array into tptree->leafs arrays, based on the leaf_id of each point
 *****************************************************************************************/
template<typename T, int Dim>
__global__ void permute_leaf_points(LeafNode* leafs, Point<T,Dim>* points, Point<T,Dim>* leaf_points, int* node_ids, int N, int internal_nodes) {
  int leaf_id;
  int idx;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<N; i+=blockDim.x*gridDim.x) {
    leaf_id = node_ids[i] - internal_nodes;
//    printf("i:%d, leaf_id:%d, offset:%d\n", i, leaf_id, leafs[leaf_id].offset);
    idx = atomicAdd(&leafs[leaf_id].size,1);
    leaf_points[idx+leafs[leaf_id].offset] = points[i];
  }
}


/*****************************************************************************************
 * Assign each point to a leaf node (based on its node_id when creating the tptree).  Also
 * computes the size and offset of each leaf node for easy permutation.
 *****************************************************************************************/
__global__ void assign_leaf_points(LeafNode* leafs, int* leaf_points, int* node_ids, int N, int internal_nodes) {
  int leaf_id;
  int idx;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<N; i+=blockDim.x*gridDim.x) {
    leaf_id = node_ids[i] - internal_nodes;
    idx = atomicAdd(&leafs[leaf_id].size,1);
    leaf_points[idx+leafs[leaf_id].offset] = i;
  }
}

/*****************************************************************************************
 * Compute the sum of each all points in each dimension (used to compute mean)
 *****************************************************************************************/
template<typename T, int Dim>
__global__ void d_collect_all_sums(Point<T,Dim>* points, T* means, int N) {
  // Each block deals with 1 vector at a time
  for(int i=blockIdx.x; i<N; i+=gridDim.x) {
//    atomicAdd(&means[threadIdx.x], points[i].coords[threadIdx.x]);
  }
}

/*****************************************************************************************
 * Compute the variance of on each dimension
 *****************************************************************************************/
template<typename T, int Dim>
__global__ void d_compute_all_variances(Point<T,Dim>* points, T* means, float* variances, int N) {
  float pointVar;
  for(int i=blockIdx.x; i<N; i+=gridDim.x) {
    pointVar = (means[threadIdx.x]-points[i].coords[threadIdx.x])*(means[threadIdx.x]-points[i].coords[threadIdx.x]); 
    atomicAdd(&variances[threadIdx.x], pointVar);
  } 
}


/*****************************************************************************************
 * Convert sum into mean
 *****************************************************************************************/
template<typename T, int Dim>
__global__ void mean_fix(T* means, float* variances, int N) {
  means[threadIdx.x] /= N;
}

/*****************************************************************************************
 * Kernel to compute the variance of each dimension of a point set
 *****************************************************************************************/
template<typename T, int Dim>
void compute_variances(Point<T, Dim>* points, float* variances, int N) {
  T* means;
  cudaMalloc(&means, Dim*sizeof(T));
  cudaMemset(&means, 0, Dim*sizeof(T));


  d_collect_all_sums<T,Dim><<<BLOCKS,Dim>>>(points, means, N);
  cudaDeviceSynchronize();

  mean_fix<T,Dim><<<1, Dim>>>(means, variances, N);
  cudaDeviceSynchronize();

  d_compute_all_variances<T,Dim><<<BLOCKS,Dim>>>(points, means, variances, N);
  cudaDeviceSynchronize(); 

   // Don't really need to divide by N, since we just compare variances to get best dimensions to use
  for(int i=0; i<Dim; ++i) {
    variances[i] /= (float)(N-1);
  }

//  cudaFree(means);

}


/*****************************************************************************************
 * Get list of dimensions with highest variance
 *****************************************************************************************/
template<typename T, int Dim, int PART_DIMS>
void find_high_var_dims(Point<T,Dim>* points, float* variances, int* dim_ids, int N) {

  for(int i=0; i<Dim; ++i)
    variances[i]=0.0;

  compute_variances<T,Dim>(points, variances, N);

  std::priority_queue<std::pair<float, int>> q;
  for(int i=0; i<Dim; ++i) {
    q.push(std::pair<float, int>(variances[i],i));
  }
  for(int i=0; i<PART_DIMS; ++i) {
    dim_ids[i]=q.top().second;
    q.pop();
  }
}

/*****************************************************************************************
 * Fill "keys" variable with projected value of each point using the given weights
 * also computes the mean.
 *****************************************************************************************/
template<typename T, int Dim, int PART_DIMS>
__global__ void fill_keys_from_weights(Point<T,Dim>* points, T* keys, int* dim_ids, float* weights, float* mean, int N) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<N; i+=blockDim.x*gridDim.x) {
    keys[i]=0.0;
    for(int j=0; j<PART_DIMS; ++j) {
      keys[i] += (weights[j]*(points[i].coords[dim_ids[j]]));
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
  for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<N; i+=blockDim.x*gridDim.x) {
    val = (*mean - keys[i])*(*mean - keys[i]);
    atomicAdd(variance, val);
  }
}

/*****************************************************************************************
 * Generate a set of random weights with best variance.  
 * RET: PART_DIMS number of weights where the linear combination yields the highest variance,
 * compared with RAND_ITERS other randomly selected weights.
 *****************************************************************************************/
template<typename T, int Dim, int PART_DIMS, int RAND_ITERS>
void generate_weight_set(Point<T,Dim>* points, T* keys, int* dim_ids, float* best_weights, int N) {
  float best_variance = 0.0;
  best_variance=0.0;

  float* temp_variance;
  cudaMallocManaged(&temp_variance, 1);
  cudaMallocManaged(&temp_variance, sizeof(float));
  float* temp_weights;
  cudaMallocManaged(&temp_weights, PART_DIMS*sizeof(float));
  float* mean;
  cudaMallocManaged(&mean, sizeof(float));


  for(int i=0; i < RAND_ITERS; ++i) {
    *temp_variance=0.0;
    *mean = 0.0;

    for(int j=0; j<PART_DIMS; ++j) {
      temp_weights[j] = ((float) (rand()) / RAND_MAX)*2 - 1.0;
    }

    fill_keys_from_weights<T,Dim,PART_DIMS><<<BLOCKS,THREADS>>>(points, keys, dim_ids, temp_weights,mean, N); 
    cudaDeviceSynchronize();

    *mean = *mean / (float)N;

    compute_weight_variance<T><<<BLOCKS,THREADS>>>(keys, mean, temp_variance, N);
    cudaDeviceSynchronize();
    *temp_variance = *temp_variance / (N-1);

    if(*temp_variance > best_variance) {
      best_variance = *temp_variance;
      for(int j=0; j<PART_DIMS; ++j) {
        best_weights[j] = temp_weights[j];
      }
    }
  }

  fill_keys_from_weights<T,Dim,PART_DIMS><<<BLOCKS,THREADS>>>(points, keys, dim_ids, best_weights, mean, N);

  cudaFree(temp_variance);
  cudaFree(mean);
  cudaFree(temp_weights);
}

