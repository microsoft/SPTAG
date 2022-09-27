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

#include "Point.hxx"
#include <cuda.h>
#include <thrust/sort.h>
#include<cub/cub.cuh>



/************************************************************************************
 * Sorts point set based on the dimension sort_dim, segmented based on leaf_offsets.
 * Results in each segment defined by leaf_offsets within "points" being sorted
 * by sort_dim. 
 * Uses Thrust key-value pair sorting to accomplish this.
************************************************************************************/
template<typename T, int Dim>
void sort_by_leaf_nodes(Point<T,Dim>* points, T* key_mem, int* leaf_offsets, int sort_dim, int N);

/************************************************************************************
 * Sorts point set based on the dimension sort_dim, segmented based on leaf_offsets.
 * Results in each segment defined by leaf_offsets within "points" being sorted
 * by sort_dim. 
 * Uses CUB segmented sort on key-value pairs.
************************************************************************************/
template<typename T, int Dim>
bool sort_by_leaf_nodes_CUB(Point<T,Dim>* points, Point<T,Dim>* point_buf, T* key_mem, T* key_buf, int num_leafs, int* leaf_offsets, int sort_dim, int N);


template<typename T, int Dim>
class Kdtree;

template<typename T, int Dim>
__global__ void set_level(Kdtree<T, Dim>* tree, Point<T,Dim>* points, int level, int level_nodes, int LEAF_SIZE);


/************************************************************************************
 * Definition of the GPU Kdtree structure. 
 * Only contains the nodes and hyperplane definitions that partition the data, as well
 * as indexes into the point array.  Does not contain the data itself.
 **********************************************************************************/
template<typename T, int Dim>
class Kdtree {
  public:
// for each level of the tree, contains the dimension that defines the hyperplane
    int* dim_list; 

// for each node, defines the value of the partitioning hyperplane.  Laid out in breadth-first order
    T* split_keys; 

// defines the indexed into the point set of each leaf node.
    int* node_offsets;

    int num_nodes;
    int num_leaves;
    int levels;
    int N;


    /************************************************************************************
     * Initialize the structure and allocated enough memory for everything
     **********************************************************************************/
    __host__ void initialize(int N_, int LEAF_SIZE) {
      N = N_;
      levels = (int)ceil(log2((float)N/(float)LEAF_SIZE));
      num_leaves = pow(2,levels);
      cudaMallocManaged(&dim_list, levels*sizeof(int));
      cudaMallocManaged(&node_offsets, (num_leaves+1)*sizeof(int));
      num_nodes = (2*num_leaves - 1);
      cudaMallocManaged(&split_keys, num_nodes*sizeof(T));
      node_offsets[0]=0;
      node_offsets[1]=N;
    }

    // For debugging purposes
    __host__ void print_tree() {
      printf("nodes:%d, leaves:%d, levels:%d\n", num_nodes, num_leaves, levels);
      for(int i=0; i<levels; i++) {
        printf("dim:%d - ", dim_list[i]);
        for(int j=0; j<pow(2,i); j++) {
          printf("(%d) %0.2f, ", (int)pow(2,i)+j-1, split_keys[(int)pow(2,i)+j-1]);
        }
        printf("\n");
      }
      printf("offsets: ");
      for(int i=0; i<num_leaves+1; i++) {
        printf("%d, ", node_offsets[i]);
      }
      printf("\n");
    }


    /************************************************************************************
     * Construct the tree 
     * For each level of the tree, it uses segmented sort to sort each node by the
     * desired dimension.  It then picks the median value as splitting key and updates
     * the node offsets for the next level of the tree.
    ************************************************************************************/
    __host__ void construct_tree(Point<T,Dim>* points, T* key_mem, int LEAF_SIZE) {

      Point<T,Dim>* point_ptr; 
      bool swap;

      // Copy buffers needed by CUB
      Point<T, Dim>* point_buf; 
      T* key_buf;
      cudaMallocManaged(&point_buf, N*sizeof(Point<T,Dim>));
      cudaMallocManaged(&key_buf, N*sizeof(T));

      for(int i=0; i<levels; i++) {
        // Sort nodes by the desired dimension
        swap = sort_by_leaf_nodes_CUB<T,Dim>(points, point_buf, key_mem, key_buf, (int)pow(2,i), node_offsets, dim_list[i], N);
        // Swap pointers if needed since CUB may give result in either buffer
        if(swap) {
          point_ptr = points;
          points = point_buf;
          point_buf = point_ptr;
        }

//        sort_by_leaf_nodes<T,Dim>(points, key_mem, node_offsets, dim_list[i], N); 
        
        // Call kernel to set the offsets and split keys
        // Use only 1 warp to avoid syncs (not much work done)
        set_level<<<32, 1>>>(this, points, i, (int)pow(2,i), LEAF_SIZE);
        cudaDeviceSynchronize();
      }
      cudaFree(point_buf);
      cudaFree(key_buf);
    }
};



/************************************************************************************
 * Copy the dimension into the key array for sorting.
 * This can be replaced by a more complex calculation using multiple dimensions
 * to create a TP tree instead of Kd-tree.
************************************************************************************/
template<typename T, int Dim>
__global__ void copy_to_keys(Point<T,Dim>* points, T* keys, int sort_dim, int N) {
  for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<N; i+=gridDim.x*blockDim.x) {
    keys[i] = points[i].coords[sort_dim];
  }
}


/************************************************************************************
 * Set the split key and node offsets for all nodes of one level of the tree, after
 * the point list has been segmented-sorted.
************************************************************************************/
template<typename T, int Dim>
__global__ void set_level(Kdtree<T, Dim>* tree, Point<T,Dim>* points, int level, int level_nodes, int LEAF_SIZE) {
  int level_offset = level_nodes-1;

  for(int j=level_nodes-threadIdx.x; j>0; j-=blockDim.x) {
    tree->node_offsets[2*j] = tree->node_offsets[j];
    tree->node_offsets[2*j-1] = (tree->node_offsets[j]+tree->node_offsets[j-1])/2;
    tree->split_keys[level_offset+j-1] = (points[tree->node_offsets[2*j-1]].coords[tree->dim_list[level]] + points[tree->node_offsets[2*j-1]-1].coords[tree->dim_list[level]])/2;
  }
}

/************************************************************************************
 * Perform a segmented sort, where each segment is a one node of the kd-tree.
 *    DEPRICATED - CUB segmented sorting is faster, use that instead.
************************************************************************************/
template<typename T, int Dim>
void sort_by_leaf_nodes(Point<T,Dim>* points, T* key_mem, int* leaf_offsets, int sort_dim, int N) {
  copy_to_keys<T,Dim><<<BLOCKS,THREADS>>>(points, key_mem, sort_dim, N);
  cudaDeviceSynchronize();

  int endIdx=0;
  int start=0;
  int end=0;
  do {
    start = end;
    end = leaf_offsets[++endIdx];
    thrust::sort_by_key(key_mem+start, key_mem+end, points+start);
  } while(end < N);
  cudaDeviceSynchronize();
}


/************************************************************************************
 * Perform a segmented sort, where each segment is a one node of the kd-tree.
 * Uses CUB segmented sort.  Requires an addition key and point buffer memory to be
 * given as well.
************************************************************************************/
template<typename T, int Dim>
bool sort_by_leaf_nodes_CUB(Point<T,Dim>* points, Point<T,Dim>* point_buf, T* key_mem, T* key_buf, int num_leafs, int* leaf_offsets, int sort_dim, int N) {


  copy_to_keys<T,Dim><<<BLOCKS,THREADS>>>(points, key_mem, sort_dim, N);
  cudaDeviceSynchronize();

    cub::DoubleBuffer<T> d_keys(key_mem, key_buf);
    cub::DoubleBuffer<Point<T,Dim>> d_values(points, point_buf);

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, N, num_leafs, leaf_offsets, leaf_offsets + 1);
            
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, N, num_leafs, leaf_offsets, leaf_offsets + 1);
    cudaDeviceSynchronize();

    if(d_keys.Current() == key_buf)
      return true;
    else
      return false;
}



/************************************************************************************
 * Create, allocate, and construct the Kd-tree based on the point set and dimension list.
 * RET: A Kd-tree defined in unified memory with all split keys and node offsets computed.
************************************************************************************/
template<typename T, int Dim>
__host__ Kdtree<T,Dim>* create_kdtree_device(Point<T,Dim>* points, T* key_mem, int* dim_list, int N, int LEAF_SIZE) {
  Kdtree<T, Dim>* d_tree;
  cudaMallocManaged(&d_tree, sizeof(Kdtree<T, Dim>));
  d_tree->initialize(N, LEAF_SIZE);

  for(int i=0; i<d_tree->levels; i++) {
    d_tree->dim_list[i] = dim_list[i];
  }

  d_tree->construct_tree(points, key_mem, LEAF_SIZE);

  return d_tree;
}


