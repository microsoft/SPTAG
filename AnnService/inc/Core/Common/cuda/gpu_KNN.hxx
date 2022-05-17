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

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <assert.h>
#include <typeinfo>
#include<cuda.h>
#include<cuda_fp16.h>

#include "log.hxx"
#include "gpu_params.h"

#include "gpu_ThreadHeap.hxx"
#include "gpu_TPtree.hxx"
#include "Point.hxx"


/************************************************************************************
 * Brute-force K nearest neighbor kernel using shared memory (transposed to avoid conflicts)
 * Each thread keeps a heap of K elements to determine K smallest distances found
 * VAR data - linear matrix fo data
 * VAR queries - linear matrix of query vectors
 * RET results - linear vector of K pairs for each query vector
************************************************************************************/
template<typename T, int Dim, int KVAL, int BLOCK_DIM>
__global__ void query_KNN(Point<T, Dim>* querySet, Point<T, Dim>* data, int dataSize, int idx_offset, int numQueries, DistPair* results) {
    // Memory for a heap for each thread
    __shared__ ThreadHeap<T, Dim, KVAL - 1, BLOCK_DIM> heapMem[BLOCK_DIM];

    DistPair extra; // extra variable to store the largest distance/id for all KNN of the point

    // Memory used to store a query point for each thread
    __shared__ T transpose_mem[Dim * BLOCK_DIM];
    TransposePoint<T, Dim, BLOCK_DIM> query;  // Stores in strided memory to avoid bank conflicts
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

/************************************************************************************
 * Brute-force K nearest neighbor kernel using shared memory (transposed to avoid conflicts)
 * Each thread keeps a heap of K elements to determine K smallest distances found
 * VAR data - linear matrix fo data
 * VAR queries - linear matrix of query vectors
 * RET results - linear vector of K pairs for each query vector
************************************************************************************/
template<typename T, int Dim, int KVAL, int BLOCK_DIM>
__global__ void findKNN_SMEM_transpose(Point<T,Dim>* data, int dataSize, int query_offset, int numQueries, DistPair* results) {


  // Memory for a heap for each thread
  __shared__ ThreadHeap<T, Dim, KVAL-1, BLOCK_DIM> heapMem[BLOCK_DIM];

  DistPair extra; // extra variable to store the largest distance/id for all KNN of the point

  // Memory used to store a query point for each thread
  __shared__ T transpose_mem[Dim*BLOCK_DIM]; 
  TransposePoint<T,Dim,BLOCK_DIM> query;  // Stores in strided memory to avoid bank conflicts
  query.setMem(&transpose_mem[threadIdx.x]);

#if LOG_LEVEL >= 5
  int dSize = sizeof(T);
#endif
  DLOG_DEBUG("Shared memory per block - Queries:%d, Heaps:%d\n", Dim*BLOCK_DIM*dSize, BLOCK_DIM*KVAL*4);
  
  heapMem[threadIdx.x].initialize();

  SUMTYPE dist;
  // Loop through all query points
  for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<numQueries; i+=blockDim.x*gridDim.x) {

    heapMem[threadIdx.x].initialize();
    extra.dist=INFTY;
    query.loadPoint(data[query_offset+i]); // Load into shared memory

    // Compare with all points in the dataset
    for(int j=0; j<dataSize; j++) {
      if(j!=(i+query_offset)) {
#if METRIC == 1 
          dist = query.cosine(&data[j]);
#else
          dist = query.l2(&data[j]);
#endif
        if(dist < extra.dist) {
          if(dist < heapMem[threadIdx.x].top()) {
            extra.dist = heapMem[threadIdx.x].vals[0].dist;
            extra.idx = heapMem[threadIdx.x].vals[0].idx;
            
            heapMem[threadIdx.x].insert(dist, j);
          }
          else {
            extra.dist = dist;
            extra.idx = j;
          }
        }
      }
    }

    // Write KNN to result list in sorted order
    results[(i+1)*KVAL-1].idx = extra.idx;
    results[(i+1)*KVAL-1].dist = extra.dist;
    for(int j=KVAL-2; j>=0; j--) {
      results[i*KVAL+j].idx = heapMem[threadIdx.x].vals[0].idx;
      results[i*KVAL+j].dist = heapMem[threadIdx.x].vals[0].dist; 
      heapMem[threadIdx.x].vals[0].dist=-1;
      heapMem[threadIdx.x].heapify();

    }
  }
}


/*****************************************************************************************
 * Perform brute-force KNN on each leaf node, where only list of point ids is stored as leafs.
 * Returns for each point: the K nearest neighbors within the leaf node containing it.
 *      ** Memory footprint reduced compared with brute-force approach (above) **
 *****************************************************************************************/
template<typename T, typename KEY_T, int Dim, int KVAL, int BLOCK_DIM, int PART_DIMS, int RAND_ITERS>
__global__ void findKNN_leaf_nodes_transpose(Point<T,Dim>* data, TPtree<T,KEY_T,Dim,PART_DIMS, RAND_ITERS>* tptree, int dataSize, int* results, long long int N) {

  __shared__ ThreadHeap<T, Dim, KVAL-1, BLOCK_DIM> heapMem[BLOCK_DIM];
  __shared__ T transpose_mem[Dim*BLOCK_DIM];

  TransposePoint<T,Dim,BLOCK_DIM> query;
  query.setMem(&transpose_mem[threadIdx.x]);

  DistPair max_K; // Stores largest of the K nearest neighbor of the query point
  bool dup; // Is the target already in the KNN list?

  DistPair target;
  long long int src_id; // Id of query vector

  int blocks_per_leaf = gridDim.x / tptree->num_leaves;
  int threads_per_leaf = blocks_per_leaf*blockDim.x;
  int thread_id_in_leaf = blockIdx.x % blocks_per_leaf * blockDim.x + threadIdx.x;
  int leafIdx= blockIdx.x / blocks_per_leaf;
  long long int leaf_offset = tptree->leafs[leafIdx].offset;

  // Each point in the leaf is handled by a separate thread
  for(int i=thread_id_in_leaf; i<tptree->leafs[leafIdx].size; i+=threads_per_leaf) {
    query.loadPoint(data[tptree->leaf_points[leaf_offset + i]]);

    heapMem[threadIdx.x].initialize();
    src_id = tptree->leaf_points[leaf_offset + i];

    // Load results from previous iterations into shared memory heap
    // and re-compute distances since they are not stored in result set
    heapMem[threadIdx.x].load_mem(data, &results[src_id*KVAL], query);

    max_K.idx = results[(src_id+1)*KVAL-1]; 
    if(max_K.idx == -1) {
      max_K.dist = INFTY;
    }
    else {
#if METRIC == 1 // cosine distance metric
        max_K.dist = query.cosine(&data[max_K.idx]);
#else // L2 distance metric
        max_K.dist = query.l2(&data[max_K.idx]);
#endif
    }

    // Compare source query with all points in the leaf
    for(long long int j=0; j<tptree->leafs[leafIdx].size; ++j) {
      if(j!=i) {
#if METRIC == 1 
        target.dist = query.cosine(&data[tptree->leaf_points[leaf_offset+j]]);
#else 
        target.dist = query.l2(&data[tptree->leaf_points[leaf_offset+j]]);
#endif
        if(target.dist < max_K.dist){
          target.idx = tptree->leaf_points[leaf_offset+j];
          dup=false;
          for(int dup_id=0; dup_id < KVAL-1; ++dup_id) {
            if(heapMem[threadIdx.x].vals[dup_id].idx == target.idx) {
              dup = true;
              dup_id=KVAL;
            }
          }
          if(!dup) { // Only consider it if not already in the KNN list
            if(target.dist < heapMem[threadIdx.x].top()) {
              max_K.dist = heapMem[threadIdx.x].vals[0].dist;
              max_K.idx = heapMem[threadIdx.x].vals[0].idx;
              heapMem[threadIdx.x].insert(target.dist, target.idx);
            }
            else {
              max_K.dist = target.dist;
              max_K.idx = target.idx;
            }
          }
        }
      }
    }

    // Write KNN to result list in sorted order
    results[(src_id+1)*KVAL-1] = max_K.idx;
    for(int j=KVAL-2; j>=0; j--) {
      results[src_id*KVAL+j] = heapMem[threadIdx.x].vals[0].idx;
      heapMem[threadIdx.x].vals[0].dist=-1;
      heapMem[threadIdx.x].heapify();
    }

  }
}

/*****************************************************************************************
 * Determines if @target is a nearer neighbor than the current contents of heap @heapMem.
 * RET: true if @target is nearer than the current KNN, false if @target is more distant,
 *      or if @target.idx is already in the heap (i.e., it is a duplicate)
 *****************************************************************************************/
template<typename T, int Dim, int KVAL, int BLOCK_DIM>
__forceinline__ __device__ bool is_NN(ThreadHeap<T,Dim,KVAL-1,BLOCK_DIM>* heapMem, DistPair* max_K, DistPair target) {
  bool ret_val = false;;
  if(target.dist < max_K->dist) {
    ret_val = true;
    for(int dup_id=0; dup_id < KVAL-1; ++dup_id) {
      if(heapMem[threadIdx.x].vals[dup_id].idx == target.idx) {
        ret_val=false;
        dup_id = KVAL;
      }
    }
  }
  return ret_val;
}

/*****************************************************************************************
 * For a given point, @src_id, looks at all neighbors' neighbors to refine KNN if any nearer
 * neighbors are found.  Recursively continues based on @DEPTH macro value.
 *****************************************************************************************/
template<typename T, int Dim, int KVAL, int BLOCK_DIM>
__device__ void check_neighbors(Point<T,Dim>* data, int* results, ThreadHeap<T, Dim, KVAL-1, BLOCK_DIM>* heapMem, DistPair* max_K, long long int neighbor, long long int src_id, TransposePoint<T,Dim,BLOCK_DIM> query, int dfs_level) {
  DistPair target;

  for(long long int j=0; j<KVAL; ++j) { // Check each neighbor of this neighbor
    target.idx = results[neighbor*KVAL + j];
    if(target.idx != src_id) { // Don't include the source itself

#if METRIC == 1
        target.dist = query.cosine(&data[target.idx]);
#else
        target.dist = query.l2(&data[target.idx]);
#endif

      if(is_NN<T,Dim,KVAL,BLOCK_DIM>(heapMem, max_K, target)) {
        if(target.dist < heapMem[threadIdx.x].top()) {
          max_K->dist = heapMem[threadIdx.x].vals[0].dist;
          max_K->idx = heapMem[threadIdx.x].vals[0].idx;
          heapMem[threadIdx.x].insert(target.dist, target.idx);
        }
        else {
          max_K->dist = target.dist;
          max_K->idx = target.idx;
        }
        if(dfs_level < REFINE_DEPTH) {
          check_neighbors<T,Dim,KVAL,BLOCK_DIM>(data, results, heapMem, max_K, target.idx, src_id, query, dfs_level+1);
        }
      }
    }
  }
}

/*****************************************************************************************
 * Refine KNN graph using neighbors' neighbors lookup process
 * Significantly improves accuracy once the approximate KNN is created.
 * DEPTH macro controls the depth of refinement that is performed.
 *****************************************************************************************/
template<typename T, int Dim, int KVAL, int BLOCK_DIM>
__global__ void refine_KNN(Point<T,Dim>* data, int* results, long long int N) {

  __shared__ ThreadHeap<T, Dim, KVAL-1, BLOCK_DIM> heapMem[BLOCK_DIM];
  __shared__ T transpose_mem[Dim*BLOCK_DIM];

  TransposePoint<T,Dim,BLOCK_DIM> query;
  query.setMem(&transpose_mem[threadIdx.x]);

  DistPair max_K;

  int neighbors[KVAL];

  for(long long int src_id=blockIdx.x*blockDim.x + threadIdx.x; src_id<N; src_id+= blockDim.x*gridDim.x) {
    query.loadPoint(data[src_id]); // Load query into shared memory

    // Load current result set into heap
    heapMem[threadIdx.x].initialize();
    heapMem[threadIdx.x].load_mem(data, &results[src_id*KVAL], query);
    max_K.idx = results[(src_id+1)*KVAL-1];
#if METRIC == 1
    max_K.dist = query.cosine(&data[max_K.idx]);
#else
    max_K.dist = query.l2(&data[max_K.idx]);
#endif

    neighbors[0] = max_K.idx;
    // Load all neighbor ids
    for(int i=1; i<KVAL; ++i) {
      neighbors[i] = heapMem[threadIdx.x].vals[i-1].idx;
    }
#pragma unroll
    for(int i=1; i<KVAL; ++i) {
      check_neighbors<T, Dim, KVAL, BLOCK_DIM>(data, results, heapMem, &max_K, neighbors[i], src_id, query, 1);
    }

    results[(src_id+1)*KVAL-1] = max_K.idx;
    // Write KNN to result list in sorted order
    for(int j=KVAL-2; j>=0; j--) {
      results[src_id*KVAL+j] = heapMem[threadIdx.x].vals[0].idx;
      heapMem[threadIdx.x].vals[0].dist=-1;
      heapMem[threadIdx.x].heapify();
    }
  }
}


/*****************************************************************************************
*****************************************************************************************
 * DEPRICATED / UNUSED CODE BELOW - Contains code used to try and improve performance
 *                  that did not work out, or code that is currently no longer needed.
 *****************************************************************************************
 *****************************************************************************************/

/*****************************************************************************************
 * Swap to DistPair objects, used by merge routine below.
 *****************************************************************************************/
__device__ void swap_result(DistPair* a, DistPair* b) {
  DistPair temp;
  temp.idx = a->idx;
  temp.dist = a->dist;
  a->idx = b->idx;
  a->dist = b->dist;
  b->idx = temp.idx;
  b->dist = temp.dist;
}

/*****************************************************************************************
 * Merge two KNN graphs, keeping only the K entries with the smallest distnace.
 * Resulting KNN graph is returned in "a" (first argument).
 * DEPRICATED - No longer need to merge now that heap handles duplicates
 *****************************************************************************************/
template<typename T, int Dim, int KVAL>
__global__ void merge_KNN_results(DistPair* a, DistPair* b, int N) {

  DistPair temp[KVAL];

  int aPtr;
  int bPtr;

  for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<N; i+=blockDim.x*gridDim.x) {
    aPtr = i*KVAL;
    bPtr = i*KVAL;

    if(i==0) {
      for(int j=0; j<KVAL; ++j) {
        printf("%0.3f, ", a[i*KVAL+j].dist);
      }
      printf("\n");
      for(int j=0; j<KVAL; ++j) {
        printf("%0.3f, ", b[i*KVAL+j].dist);
      }
      printf("\n\n");
    }

    for(int j=0; j<KVAL; ++j) {
      if(a[aPtr].idx == b[bPtr].idx && j < KVAL-1) {
        bPtr++;
      }
      if(a[aPtr].dist <= b[bPtr].dist) {
        temp[j].idx = a[aPtr].idx;
        temp[j].dist = a[aPtr].dist;
        aPtr++;
      }
      else {
        temp[j].idx = b[bPtr].idx;
        temp[j].dist = b[bPtr].dist;
        bPtr++;
      }
    }
    for(int j=0; j<KVAL; ++j) {
      a[i*KVAL+j].idx = temp[j].idx;
      a[i*KVAL+j].dist = temp[j].dist;
    }

    if(i==0) {
      for(int j=0; j<KVAL; ++j) {
        printf("%0.3f, ", a[i*KVAL+j].dist);
      }
      printf("\n");
    }
  }

}


/*
#define WARP_SIZE 32
#define BUFF_SIZE 512

template<typename T, typename KEY_T, int Dim, int KVAL, int BLOCK_DIM, int PART_DIMS, int RAND_ITERS>
__global__ void block_KNN(Point<T,Dim>* d_vectors, TPtree<T,KEY_T,Dim,PART_DIMS,RAND_ITERS>* tptree, int* results, int size) {

  __shared__ Point<T,Dim> query;
  const int WARPS_PER_BLOCK = BLOCK_DIM/WARP_SIZE;
  int laneId = threadIdx.x % WARP_SIZE;
  int warpId = threadIdx.x / WARP_SIZE;
  const int BUFF_PER_THREAD = (BUFF_SIZE) / BLOCK_DIM;

  typedef cub::WarpReduce<SUMTYPE> distReduce;
  __shared__ typename distReduce::TempStorage tempDist[WARPS_PER_BLOCK];

  typedef cub::BlockRadixSort<SUMTYPE, BLOCK_DIM, BUFF_PER_THREAD, int> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  int blocks_per_leaf = gridDim.x / tptree->num_leaves;
  int leaf_idx = blockIdx.x / blocks_per_leaf;
  int blockId_in_leaf = blockIdx.x % blocks_per_leaf;

  long long int leaf_offset = tptree->leafs[leaf_idx].offset;

  __shared__ int candidate_id[BUFF_SIZE];
  __shared__ SUMTYPE candidate_dist[BUFF_SIZE];

  int reg_id[BUFF_PER_THREAD];
  SUMTYPE reg_dist[BUFF_PER_THREAD];

//  for(int i = 0; i < tptree->leafs[blockIdx.x].size; i++) {
  for(int i = blockId_in_leaf; i < tptree->leafs[leaf_idx].size; i+=blocks_per_leaf) {

    // Load query point into shared memory
    query.id = tptree->leaf_points[leaf_offset + i];
    __syncthreads();

    if(threadIdx.x < Dim) {
      query.coords[threadIdx.x] = d_vectors[query.id].coords[threadIdx.x];
    }
    __syncthreads();

    // Fill KNN list from previous KNN results
    if(threadIdx.x < KVAL) {
      candidate_id[threadIdx.x] = results[query.id*KVAL+threadIdx.x];
      if(candidate_id[threadIdx.x] == -1)
        candidate_dist[threadIdx.x] = INFTY;
      else
        candidate_dist[threadIdx.x] = query.l2_dist_sq(&d_vectors[candidate_id[threadIdx.x]]);
    }
//    __syncthreads();
    // How many buffer fills & sorts do we need to do
    for(int round_offset=0; round_offset < tptree->leafs[leaf_idx].size; round_offset += (BUFF_SIZE-KVAL)) {

      for(int j=warpId; j<BUFF_SIZE-KVAL; j+=WARPS_PER_BLOCK) { // Fill buffers
        if(round_offset + j < tptree->leafs[leaf_idx].size) {
          reg_id[0] = tptree->leaf_points[leaf_offset + round_offset + j];
          if(reg_id[0] == query.id)
            reg_dist[0]=INFTY/WARP_SIZE;
          else
            reg_dist[0] = query.l2_partial(&d_vectors[reg_id[0]], laneId);

          // Check for duplicates
#pragma unroll
          for(int k=laneId; k<Dim; k+=WARP_SIZE) {
            if(reg_id[0] == candidate_id[k])
              reg_dist[0]=INFTY/WARP_SIZE;
          }
        }
        else {
          reg_id[0] = -1;
          reg_dist[0] = INFTY/WARP_SIZE;
        }
        __syncthreads();
          reg_dist[0] = distReduce(tempDist[warpId]).Sum(reg_dist[0]);
        __syncthreads();

        if(laneId == 0) {
          candidate_dist[KVAL+j] = reg_dist[0];
          candidate_id[KVAL+j] = reg_id[0];
        }
      }

      // Sort candidates
      __syncthreads();
      for(int j=0; j<BUFF_PER_THREAD; j++) {
        reg_id[j] = candidate_id[j*BLOCK_DIM + threadIdx.x];
        reg_dist[j] = candidate_dist[j*BLOCK_DIM + threadIdx.x];
      }
      __syncthreads();

      BlockRadixSort(temp_storage).SortBlockedToStriped(reg_dist, reg_id);
      __syncthreads();


      if(threadIdx.x < KVAL) {
        candidate_dist[threadIdx.x] = reg_dist[0];
        candidate_id[threadIdx.x] = reg_id[0];
      }

    }
    if(threadIdx.x < KVAL) {
      results[query.id*KVAL+threadIdx.x] = candidate_id[threadIdx.x];
    }
    __syncthreads();
  }
}
*/

/*****************************************************************************************
 * Perform brute-force KNN on each leaf node, where only list of point ids is stored as leafs.
 * Uses registers to store K, NOT shared memory heap structure
 * SLOWER THAN SHARED MEMORY VERSION - Back-burner to see if this can be improved
 * Returns for each point: the K nearest neighbors within the leaf node containing it.
 *****************************************************************************************/
template<typename T, typename KEY_T, int Dim, int KVAL, int BLOCK_DIM, int PART_DIMS, int RAND_ITERS>
__global__ void findKNN_leaf_nodes_regs(Point<T,Dim>* data, TPtree<T,KEY_T,Dim,PART_DIMS, RAND_ITERS>* tptree, int dataSize, int* results, long long int N) {
  __shared__ ThreadHeap<T, Dim, KVAL-1, BLOCK_DIM> heapMem[BLOCK_DIM];
//  __shared__ T transpose_mem[Dim*BLOCK_DIM];

//  TransposePoint<T,Dim,BLOCK_DIM> query;
//  query.setMem(&transpose_mem[threadIdx.x]);

  T query[Dim];

  DistPair max_K;
  bool dup;

  DistPair target;
  long long int src_id;

  int blocks_per_leaf = gridDim.x / tptree->num_leaves;
  int threads_per_leaf = blocks_per_leaf*blockDim.x;
  int thread_id_in_leaf = blockIdx.x % blocks_per_leaf * blockDim.x + threadIdx.x;
  int leafIdx= blockIdx.x / blocks_per_leaf;
  long long int leaf_offset = tptree->leafs[leafIdx].offset;

  // Each point in the leaf is handled by a separate thread
  for(int i=thread_id_in_leaf; i<tptree->leafs[leafIdx].size; i+=threads_per_leaf) {
//    query.loadPoint(data[tptree->leaf_points[leaf_offset + i]]);
    for(int j=0; j<Dim; ++j) {
      query[j] = data[tptree->leaf_points[leaf_offset+i]].coords[j];
    }

    heapMem[threadIdx.x].initialize();
    src_id = tptree->leaf_points[leaf_offset + i];

//    max_K.dist = INFTY;

    // Load results from previous iterations into shared memory heap
    // and re-compute distances since they are not stored in result set
    heapMem[threadIdx.x].load_mem_regs(data, &results[src_id*KVAL], query);

    max_K.idx = results[(src_id+1)*KVAL-1];
    if(max_K.idx == -1) {
      max_K.dist = INFTY;
    }
    else
      max_K.dist = l2_ILP(query, &data[max_K.idx]);
//    max_K.dist = results[(src_id+1)*KVAL-1].dist;

    for(long long int j=0; j<tptree->leafs[leafIdx].size; ++j) {
      if(j!=i) {
        target.dist = l2_ILP(query, &data[tptree->leaf_points[leaf_offset+j]]);

        if(target.dist < max_K.dist) {
          target.idx = tptree->leaf_points[leaf_offset+j];
          dup=false;
          for(int dup_id=0; dup_id < KVAL-1; ++dup_id) {
            if(heapMem[threadIdx.x].vals[dup_id].idx == target.idx) {
              dup = true;
              dup_id=KVAL;
            }
          }
          if(!dup) {
            if(target.dist < heapMem[threadIdx.x].top()) {
              max_K.dist = heapMem[threadIdx.x].vals[0].dist;
              max_K.idx = heapMem[threadIdx.x].vals[0].idx;
              heapMem[threadIdx.x].insert(target.dist, target.idx);
            }
            else {
              max_K.dist = target.dist;
              max_K.idx = target.idx;
            }
          }
        }
      }
    }

      results[(src_id+1)*KVAL-1] = max_K.idx;
//      results[(src_id+1)*KVAL-1].dist = max_K.dist;
    // Write KNN to result list in sorted order
    for(int j=KVAL-2; j>=0; j--) {
      results[src_id*KVAL+j] = heapMem[threadIdx.x].vals[0].idx;
//      results[src_id*KVAL+j].dist = heapMem[threadIdx.x].vals[0].dist;
      // Use sqrt only if using "l2_check" that shortcircuits without sqrts
//      results[i*KVAL + j].dist = sqrt((float)heapMem[threadIdx.x].vals[0].dist);
      heapMem[threadIdx.x].vals[0].dist=0;
      heapMem[threadIdx.x].heapify();
    }
  }
}
