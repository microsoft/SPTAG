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

#ifndef _SPTAG_COMMON_CUDA_KNN_H_
#define _SPTAG_COMMON_CUDA_KNN_H_

#include "Refine.hxx"
#include "ThreadHeap.hxx"
#include "TPtree.hxx"

#include <chrono>

template<typename T, typename SUMTYPE, int Dim>
__device__ bool violatesRNG(Point<T,SUMTYPE,Dim>* data, DistPair<SUMTYPE> farther, DistPair<SUMTYPE> closer, int metric) {
  SUMTYPE between;
  if(metric == 0) {
    between = data[closer.idx].l2(&data[farther.idx]);
  }
  else {
    between = data[closer.idx].cosine(&data[farther.idx]);
  }
  return between <= farther.dist;
}

/*****************************************************************************************
 * Perform brute-force KNN on each leaf node, where only list of point ids is stored as leafs.
 * Returns for each point: the K nearest neighbors within the leaf node containing it.
 *****************************************************************************************/
template<typename T, typename KEY_T, typename SUMTYPE, int Dim, int BLOCK_DIM>
__global__ void findKNN_leaf_nodes(Point<T,SUMTYPE,Dim>* data, TPtree<T,KEY_T,SUMTYPE,Dim>* tptree, int KVAL, int* results, int metric) {

  extern __shared__ char sharememory[];
  ThreadHeap<T, SUMTYPE, Dim, BLOCK_DIM> heapMem;

  heapMem.initialize(&((DistPair<SUMTYPE>*)sharememory)[(KVAL-1) * threadIdx.x], KVAL-1);

  Point<T,SUMTYPE,Dim> query;

  DistPair<SUMTYPE> max_K; // Stores largest of the K nearest neighbor of the query point
  bool dup; // Is the target already in the KNN list?

  DistPair<SUMTYPE> target;

  int blocks_per_leaf = gridDim.x / tptree->num_leaves;
  int threads_per_leaf = blocks_per_leaf*blockDim.x;
  int thread_id_in_leaf = blockIdx.x % blocks_per_leaf * blockDim.x + threadIdx.x;
  int leafIdx= blockIdx.x / blocks_per_leaf;
  long long int leaf_offset = tptree->leafs[leafIdx].offset;

  // Each point in the leaf is handled by a separate thread
  for(int i=thread_id_in_leaf; i<tptree->leafs[leafIdx].size; i+=threads_per_leaf) {
    query = data[tptree->leaf_points[leaf_offset + i]];

    heapMem.reset();

    // Load results from previous iterations into shared memory heap
    // and re-compute distances since they are not stored in result set
    heapMem.load_mem_sorted(data, &results[(long long int)query.id*KVAL], query, metric);

    max_K.idx = results[((long long int)query.id+1)*KVAL-1];
    if(max_K.idx == -1) {
      max_K.dist = INFTY<SUMTYPE>();
    }
    else {
      if(metric == 0) {
        max_K.dist = query.l2(&data[max_K.idx]);
      }
      else if(metric == 1) {
        max_K.dist = query.cosine(&data[max_K.idx]);
      }
    }

    // Compare source query with all points in the leaf
    for(long long int j=0; j<tptree->leafs[leafIdx].size; ++j) {
      if(j!=i) {
        if(metric == 0) {
          target.dist = query.l2(&data[tptree->leaf_points[leaf_offset+j]]);
        }
        else if (metric == 1) {
          target.dist = query.cosine(&data[tptree->leaf_points[leaf_offset+j]]);
        }
        if(target.dist < max_K.dist) {
          target.idx = tptree->leaf_points[leaf_offset+j];

          if(target.dist <= heapMem.top()) {
            dup=false;
            for(int dup_id=0; dup_id < KVAL-1; ++dup_id) {
              if(heapMem.vals[dup_id].idx == target.idx) {
                dup = true;
                dup_id=KVAL;
              }
            }
            if(!dup) { // Only consider it if not already in the KNN list
              max_K.dist = heapMem.vals[0].dist;
              max_K.idx = heapMem.vals[0].idx;
              heapMem.insert(target.dist, target.idx);
            }
          }
          else {
            max_K.dist = target.dist;
            max_K.idx = target.idx;
          }
        }
      }
    }
    // Write KNN to result list in sorted order
    results[((long long int)query.id+1)*KVAL-1] = max_K.idx;
    for(int j=KVAL-2; j>=0; j--) {
      results[(long long int)query.id*KVAL+j] = heapMem.vals[0].idx;
      heapMem.vals[0].dist=-1;
      heapMem.heapify();
    }
  }
}

/*****************************************************************************************
 * For a given point, @src_id, looks at all neighbors' neighbors to refine KNN if any nearer
 * neighbors are found.  Recursively continues based on @DEPTH macro value.
 *****************************************************************************************/
template<typename T, typename SUMTYPE, int Dim, int BLOCK_DIM>
__device__ void check_neighbors(Point<T,SUMTYPE,Dim>* data, int* results, int KVAL, ThreadHeap<T, SUMTYPE, Dim, BLOCK_DIM>* heapMem, DistPair<SUMTYPE>* max_K, int neighbor, int src_id, Point<T,SUMTYPE,Dim>* query, int metric) {

  DistPair<SUMTYPE> target;
  bool dup;

  for(long long int j=0; j<KVAL; ++j) { // Check each neighbor of this neighbor
    target.idx = results[neighbor*KVAL + j];
    if(target.idx != src_id && target.idx != -1) { // Don't include the source itself

      dup = (target.idx == max_K->idx);
      for(int dup_id=0; dup_id < KVAL-1 && dup==false; ++dup_id) {
        dup = (heapMem->vals[dup_id].idx == target.idx);
      }
      target.dist = INFTY<SUMTYPE>();
      if(!dup) {
        if(metric == 0) {
          target.dist = query->l2(&data[target.idx]);
        }
        else if(metric == 1) {
          target.dist = query->cosine(&data[target.idx]);
        }
      }
      if(target.dist < max_K->dist) {
        if(target.dist < heapMem->top()) {
          max_K->dist = heapMem->vals[0].dist;
          max_K->idx = heapMem->vals[0].idx;
          heapMem->insert(target.dist, target.idx);
        }
        else {
          max_K->dist = target.dist;
          max_K->idx = target.idx;
        }
      }
    }
  }
}

/*****************************************************************************************
 * Improve KNN graph using neighbors' neighbors lookup process
 * Significantly improves accuracy once the approximate KNN is created.
 *****************************************************************************************/
template<typename T, typename SUMTYPE, int Dim, int BLOCK_DIM>
__global__ void neighbors_KNN(Point<T,SUMTYPE,Dim>* data, int* results, int N, int KVAL, int metric) {

  extern __shared__ char sharememory[];
  ThreadHeap<T, SUMTYPE, Dim, BLOCK_DIM> heapMem;

  heapMem.initialize(&((DistPair<SUMTYPE>*)sharememory)[(KVAL-1) * threadIdx.x], KVAL-1);

  Point<T,SUMTYPE,Dim> query;
  DistPair<SUMTYPE> max_K;

  int neighbor;

  for(int src_id=blockIdx.x*blockDim.x + threadIdx.x; src_id<N; src_id+= blockDim.x*gridDim.x) {
    query = data[src_id]; // Load query into registers

    heapMem.reset();
    // Load current result set into heap
    heapMem.load_mem_sorted(data, &results[src_id*KVAL], query, metric);

    max_K.dist = INFTY<SUMTYPE>();
    max_K.idx = results[(src_id+1)*KVAL-1];

    if(metric == 0) {
      if(max_K.idx != -1) {
        max_K.dist = query.l2(&data[max_K.idx]);
      }
    }
    else if (metric == 1) {
      if(max_K.idx != -1) {
        max_K.dist = query.cosine(&data[max_K.idx]);
      }
    }

    // check nearest neighbor first
    neighbor = max_K.idx;
    check_neighbors<T, SUMTYPE, Dim, BLOCK_DIM>(data, results, KVAL, &heapMem, &max_K, neighbor, src_id, &query, metric);

    for(int i=0; i<KVAL-1 && neighbor != -1; ++i) {
      neighbor = heapMem.vals[i].idx;
      if(neighbor != -1) {
        check_neighbors<T, SUMTYPE, Dim, BLOCK_DIM>(data, results, KVAL, &heapMem, &max_K, neighbor, src_id, &query, metric);
      }
    }

    results[(src_id+1)*KVAL-1] = max_K.idx;
    // Write KNN to result list in sorted order
    for(int j=KVAL-2; j>=0; j--) {
      results[src_id*KVAL+j] = heapMem.vals[0].idx;
      heapMem.vals[0].dist=-1;
      heapMem.heapify();
    }
  }
}

/*****************************************************************************************
 * Compute the "accessibility score" of a given point (target) from a source (id).
 * This is computed as the number of neighbors that have an edge to the target.
 * Costly operation but useful to improve the accuracy of the graph by increasing the
 * connectivity.
 * DEPRICATED AND NO LONGER USED
 e****************************************************************************************/
template<typename T>
__device__ int compute_accessibility(int* results, int id, int target, int KVAL) {
  int access=0;
  int* ptr;
  for(int i=0; i<KVAL; i++) {
    access += (results[id*KVAL+i] == target);
    if(results[id*KVAL+i] != -1) {
      ptr = &results[results[id*KVAL+i]*KVAL];
      for(int j=0; j<KVAL; j++) {
        access += (ptr[j] == target);
      }
    }
  }
  return access;
}

/*****************************************************************************************
 * Perform the brute-force graph construction on each leaf node, while STRICTLY maintaining RNG properties.  May end up with less than K neighbors per vector.
 *****************************************************************************************/
template<typename T, typename KEY_T, typename SUMTYPE, int Dim, int BLOCK_DIM>
__global__ void findRNG_strict(Point<T,SUMTYPE,Dim>* data, TPtree<T,KEY_T,SUMTYPE,Dim>* tptree, int KVAL, int* results, int metric, size_t min_id, size_t max_id) {

  extern __shared__ char sharememory[];

  DistPair<SUMTYPE>* threadList = (&((DistPair<SUMTYPE>*)sharememory)[KVAL*threadIdx.x]);
  SUMTYPE max_dist = INFTY<SUMTYPE>();
  DistPair<SUMTYPE> temp;
  int read_id, write_id;

  for(int i=0; i<KVAL; i++) {
    threadList[i].dist = INFTY<SUMTYPE>();
  }

  Point<T,SUMTYPE,Dim> query;
  DistPair<SUMTYPE> target;
  DistPair<SUMTYPE> candidate;

  int blocks_per_leaf = gridDim.x / tptree->num_leaves;
  int threads_per_leaf = blocks_per_leaf*blockDim.x;
  int thread_id_in_leaf = blockIdx.x % blocks_per_leaf * blockDim.x + threadIdx.x;
  int leafIdx= blockIdx.x / blocks_per_leaf;
  size_t leaf_offset = tptree->leafs[leafIdx].offset;

  bool good;

  // Each point in the leaf is handled by a separate thread
  for(int i=thread_id_in_leaf; leafIdx < tptree->num_leaves && i<tptree->leafs[leafIdx].size; i+=threads_per_leaf) {
    if(tptree->leaf_points[leaf_offset+i] >= min_id && tptree->leaf_points[leaf_offset+i] < max_id) {
      query = data[tptree->leaf_points[leaf_offset + i]];


      // Load results from previous iterations into shared memory heap
      // and re-compute distances since they are not stored in result set
      for(int j=0; j<KVAL; j++) {
        threadList[j].idx = results[(((long long int)(query.id-min_id))*(long long int)(KVAL))+j];
        if(threadList[j].idx != -1) {
          if(metric == 0) {
            threadList[j].dist = query.l2(&data[threadList[j].idx]);
	  }
	  else {
            threadList[j].dist = query.cosine(&data[threadList[j].idx]);
	  }
        }
        else {
          threadList[j].dist = INFTY<SUMTYPE>();
        }
      }
      max_dist = threadList[KVAL-1].dist;

      // Compare source query with all points in the leaf
      for(size_t j=0; j<tptree->leafs[leafIdx].size; ++j) {
        if(j!=i) {
          good = true;
	  candidate.idx = tptree->leaf_points[leaf_offset+j];
          if(metric == 0) {
            candidate.dist = query.l2(&data[candidate.idx]);
          }
          else if(metric == 1) {
            candidate.dist = query.cosine(&data[candidate.idx]);
         }

         if(candidate.dist < max_dist){ // If it is a candidate to be added to neighbor list

  // TODO: handle if two different points have same dist
	    for(read_id=0; candidate.dist > threadList[read_id].dist && good; read_id++) {
              if(violatesRNG<T, SUMTYPE, Dim>(data, candidate, threadList[read_id], metric)) {
                good = false;
	      }
	    }
	    if(candidate.idx == threadList[read_id].idx)  // Ignore duplicates
              good = false;

	    if(good) { // candidate should be in RNG list
              target = threadList[read_id];
	      threadList[read_id] = candidate;
	      read_id++;
              for(write_id = read_id; read_id < KVAL && threadList[read_id].idx != -1; read_id++) {
                if(!violatesRNG<T, SUMTYPE, Dim>(data, threadList[read_id], candidate, metric)) {
                  if(read_id == write_id) {
                    temp = threadList[read_id];
		    threadList[write_id] = target;
		    target = temp;
		  }
		  else {
                    threadList[write_id] = target;
		    target = threadList[read_id];
		  }
		  write_id++;
                }
	      }
	      if(write_id < KVAL) {
                threadList[write_id] = target;
                write_id++;
              }
	      for(int k=write_id; k<KVAL && threadList[k].idx != -1; k++) {
                threadList[k].dist = INFTY<SUMTYPE>();
	        threadList[k].idx = -1;
	      }
              max_dist = threadList[KVAL-1].dist;
	    }
          }
        }
      }
      for(size_t j=0; j<KVAL; j++) {
        results[(size_t)(query.id-min_id)*KVAL+j] = threadList[j].idx;
      }

    } // End if within batch
  } // End leaf node loop
}

/*****************************************************************************************
 * Perform the brute-force graph construction on each leaf node, but tries to maintain 
 * RNG properties when adding/removing vectors from the neighbor list.  Also only
 * inserts new vectors into the neighbor list when the "accessibility score" is 0 (i.e.,
 * it is not already accessible by any neighbors.
 *****************************************************************************************/
template<typename T, typename KEY_T, typename SUMTYPE, int Dim, int BLOCK_DIM>
__global__ void findRNG_leaf_nodes(Point<T,SUMTYPE,Dim>* data, TPtree<T,KEY_T,SUMTYPE,Dim>* tptree, int KVAL, int* results, int metric) {

  extern __shared__ char sharememory[];
  ThreadHeap<T, SUMTYPE, Dim, BLOCK_DIM> heapMem; // Stored in registers but contains a pointer 

  // Assigns the memory pointers in the heap to shared memory
  heapMem.initialize(&((DistPair<SUMTYPE>*)sharememory)[(KVAL-1) * threadIdx.x], KVAL-1);

  SUMTYPE nearest_dist;
  Point<T,SUMTYPE,Dim> query;

  DistPair<SUMTYPE> max_K; // Stores largest of the K nearest neighbor of the query point
  bool dup; // Is the target already in the KNN list?

  DistPair<SUMTYPE> target;

  int blocks_per_leaf = gridDim.x / tptree->num_leaves;
  int threads_per_leaf = blocks_per_leaf*blockDim.x;
  int thread_id_in_leaf = blockIdx.x % blocks_per_leaf * blockDim.x + threadIdx.x;
  int leafIdx= blockIdx.x / blocks_per_leaf;
  long long int leaf_offset = tptree->leafs[leafIdx].offset;

  int write_id;
  SUMTYPE write_dist;

  // Each point in the leaf is handled by a separate thread
  for(int i=thread_id_in_leaf; i<tptree->leafs[leafIdx].size; i+=threads_per_leaf) {
    query = data[tptree->leaf_points[leaf_offset + i]];
    nearest_dist = INFTY<SUMTYPE>();

    heapMem.reset();

    // Load results from previous iterations into shared memory heap
    // and re-compute distances since they are not stored in result set
    heapMem.load_mem_sorted(data, &results[(long long int)query.id*KVAL], query, metric);

    for(int k=0; k<KVAL-1; k++) {
      if(heapMem.vals[k].dist < nearest_dist) {
        nearest_dist = heapMem.vals[k].dist;
      }
    }

    max_K.idx = results[((long long int)query.id+1)*KVAL-1];
    if(max_K.idx == -1) {
      max_K.dist = INFTY<SUMTYPE>();
    }
    else {
      if(metric == 0) {
        max_K.dist = query.l2(&data[max_K.idx]);
      }
      else if(metric == 1) {
        max_K.dist = query.cosine(&data[max_K.idx]);
      }
    }

    // Compare source query with all points in the leaf
    for(long long int j=0; j<tptree->leafs[leafIdx].size; ++j) {
      if(j!=i) {
        if(metric == 0) {
          target.dist = query.l2(&data[tptree->leaf_points[leaf_offset+j]]);
        }
        else if(metric == 1) {
          target.dist = query.cosine(&data[tptree->leaf_points[leaf_offset+j]]);
       }

        if(target.dist < max_K.dist){ // If it is to be added as a new near neighbor
          target.idx = tptree->leaf_points[leaf_offset+j];

          if(target.dist <= heapMem.top()) {
            dup=false;
            for(int dup_id=0; dup_id < KVAL-1; ++dup_id) {
              if(heapMem.vals[dup_id].idx == target.idx) {
                dup = true;
                dup_id=KVAL;
              }
            }
// Only consider it if not already in the KNN list and it is not already accessible
            if(!dup) {
              write_dist = INFTY<SUMTYPE>();
              write_id=0;

              if(metric == 0) { // L2
                for(int k=0; k<KVAL-1 && heapMem.vals[k].dist > target.dist; k++) {
                  if(heapMem.vals[k].idx != -1 && (data[target.idx].l2(&data[heapMem.vals[k].idx]) < write_dist)) {
                    write_id = k;
                    write_dist = data[target.idx].l2(&data[heapMem.vals[k].idx]);
                  }
                }
              }
              else if(metric == 1){ // Cosine
                for(int k=0; k<KVAL-1 && heapMem.vals[k].dist > target.dist; k++) {
                  if(heapMem.vals[k].idx != -1 && (data[target.idx].cosine(&data[heapMem.vals[k].idx]) < write_dist)) {
                    write_id = k;
                    write_dist = data[target.idx].cosine(&data[heapMem.vals[k].idx]);
                  }
                }
              }

              if(max_K.idx == -1) {
                max_K.dist = heapMem.vals[0].dist;
                max_K.idx = heapMem.vals[0].idx;
                heapMem.insert(target.dist, target.idx);
              }
              else {
                heapMem.insertAt(target.dist, target.idx, write_id); // Replace RNG violating vector
              }
              if(target.dist < nearest_dist)
                nearest_dist = target.dist;
            }
          }
          else {
            max_K.dist = target.dist;
            max_K.idx = target.idx;
          }
        }
      }
    }

    // Write KNN to result list in sorted order
    results[((long long int)query.id+1)*KVAL-1] = max_K.idx;
    for(int j=KVAL-2; j>=0; j--) {
      results[(long long int)query.id*KVAL+j] = heapMem.vals[0].idx;
      heapMem.vals[0].dist=-1;
      heapMem.heapify();
    }

  }
}

/*****************************************************************************************
 * Compare distance of all neighbor's points to see if they are nearer neighbors
 * RNG properties when adding/removing vectors from the neighbor list.  Also only
 * inserts new vectors into the neighbor list when the "accessibility score" is 0 (i.e.,
 * it is not already accessible by any neighbors.
 *****************************************************************************************/
template<typename T, typename SUMTYPE, int Dim, int BLOCK_DIM>
__device__ void check_neighbors_RNG(Point<T,SUMTYPE,Dim>* data, int* results, int KVAL, int* RNG_id, int* RNG_dist, long long int neighbor, long long int src_id, Point<T,SUMTYPE,Dim>* query, int metric) {
  DistPair<SUMTYPE> target;
  bool dup;
  int write_id;
  SUMTYPE write_dist;

  for(long long int j=0; j<KVAL; ++j) { // Check each neighbor of this neighbor
    target.idx = results[neighbor*KVAL + j];
    if(target.idx != src_id && target.idx != -1) { // Don't include the source itself
      dup=false;
      for(int dup_id=0; dup_id < KVAL && dup==false; ++dup_id) {
        dup = (RNG_id[dup_id] == target.idx);
      }
      target.dist = INFTY<SUMTYPE>();
      if(!dup) {
        if(metric == 0) {
          target.dist = query->l2(&data[target.idx]);
        }
        else if(metric == 1) {
          target.dist = query->cosine(&data[target.idx]);
        }
      }
      if(target.dist < RNG_dist[KVAL-1]) {

        if(RNG_dist[KVAL-1] != -1) {
          for(int k=0; target.dist > RNG_dist[k] && dup==false; k++) { // check to enforce RNG property
            if(metric == 0) {
              dup = (data[target.idx].l2(&data[RNG_id[k]]) < target.dist);
            }
            else if(metric == 1) {
              dup = (data[target.idx].cosine(&data[RNG_id[k]]) < target.dist);
            }
          }
        }
        if(!dup) { // If new point doesn't violate RNG property
            // Clean up RNG list

          write_id=KVAL-1;
          write_dist=INFTY<SUMTYPE>();
          if(RNG_id[KVAL-1] != -1) {
            if(metric == 0) {
              for(int k=KVAL-1; RNG_dist[k] > target.dist && k>=0; k--) { // Remove farthest neighbor that will now violate RNG
                if(RNG_id[k] != -1 && (write_dist > data[target.idx].l2(&data[RNG_id[k]]))) {
                  write_id = k;
                  write_dist = data[target.idx].l2(&data[RNG_id[k]]);
//              k=-1;
                }
              }
            }
            else if(metric == 1) {
              for(int k=KVAL-1; RNG_dist[k] > target.dist && k>=0; k--) { // Remove farthest neighbor that will now violate RNG
                if(RNG_id[k] != -1 && (write_dist > data[target.idx].cosine(&data[RNG_id[k]]))) {
                  write_id = k;
                  write_dist = data[target.idx].l2(&data[RNG_id[k]]);
                }
              }
            }
          }

          for(int k=write_id; k>=0; k--) {
            if(k==0 || RNG_dist[k-1] < target.dist) {
              RNG_dist[k] = target.dist;
              RNG_id[k] = target.idx;
              k = -1;
            }
            else {
              RNG_dist[k] = RNG_dist[k-1];
              RNG_id[k] = RNG_id[k-1];
            }
          }
        }
      }
    }
  }
}

/************************************************************************
 * Refine graph by performing check_neighbors_RNG on every node in the graph
 ************************************************************************/
template<typename T, typename SUMTYPE, int Dim, int BLOCK_DIM>
__global__ void neighbors_RNG(Point<T,SUMTYPE,Dim>* data, int* results, int N, int KVAL, int metric) {

  extern __shared__ char sharememory[];
  int* RNG_id = &((int*)sharememory)[2*KVAL*threadIdx.x];
  int* RNG_dist = &((int*)sharememory)[2*KVAL*threadIdx.x + KVAL];

  Point<T,SUMTYPE,Dim> query;
  int neighbor;

  for(long long int src_id=blockIdx.x*blockDim.x + threadIdx.x; src_id<N; src_id+= blockDim.x*gridDim.x) {
    query = data[src_id];

    // Load current result set into registers
    for(int i=0; i<KVAL; i++) {
      RNG_id[i] = results[src_id*KVAL + i];
      if(RNG_id[i] == -1) {
        RNG_dist[i] = INFTY<SUMTYPE>();
      }
      else {
        if(metric == 0) {
          RNG_dist[i] = query.l2(&data[RNG_id[i]]);
        }
        else if(metric == 1) {
          RNG_dist[i] = query.cosine(&data[RNG_id[i]]);
        }
      }
    }

    for(int i=0; i<KVAL && neighbor != -1; ++i) {
      neighbor = RNG_id[i];
      if(neighbor != -1) {
        check_neighbors_RNG<T, SUMTYPE, Dim, BLOCK_DIM>(data, results, KVAL, RNG_id, RNG_dist, neighbor, src_id, &query, metric);
      }
    }

    // Write KNN to result list in sorted order
    for(int i=0; i<KVAL; i++) {
      results[src_id*KVAL + i] = RNG_id[i];
    }
  }
}


/****************************************************************************************
 * Non-batched version of graph construction on GPU, only supports creating KNN or "loosely"
 * enforced RNG.
 *
 * Create either graph on the GPU, graph is saved into @results and is stored on the CPU
 * graphType: KNN=0, RNG=1
 * Note, vectors of MAX_DIM number dimensions are used, so an upper-bound must be determined
 * at compile time
 ***************************************************************************************/
template<typename DTYPE, typename SUMTYPE, int MAX_DIM>
void buildGraphGPU(SPTAG::VectorIndex* index, int dataSize, int KVAL, int trees, int* results, int refines, int graphtype, int initSize, int refineDepth, int leafSize) {

  LOG(SPTAG::Helper::LogLevel::LL_Info, "Starting buildGraphGPU\n");

  int dim = index->GetFeatureDim();
  int metric = (int)index->GetDistCalcMethod();
  DTYPE* data = (DTYPE*)index->GetSample(0);

  // Number of levels set to have approximately 500 points per leaf
  int levels = (int)std::log2(dataSize/leafSize);

  int KNN_blocks; // number of threadblocks used

  Point<DTYPE,SUMTYPE,MAX_DIM>* points = convertMatrix<DTYPE,SUMTYPE,MAX_DIM>(index, dataSize, dim);

  for(int i=0;  i<dataSize; i++) {
    points[i].id = i;
  }


  Point<DTYPE, SUMTYPE, MAX_DIM>* d_points;
  LOG(SPTAG::Helper::LogLevel::LL_Debug, "Alloc'ing Points on device: %ld bytes.\n", dataSize*sizeof(Point<DTYPE,SUMTYPE,MAX_DIM>));
  CUDA_CHECK(cudaMalloc(&d_points, dataSize*sizeof(Point<DTYPE,SUMTYPE,MAX_DIM>)));

  LOG(SPTAG::Helper::LogLevel::LL_Debug, "Copying to device.\n");
  CUDA_CHECK(cudaMemcpy(d_points, points, dataSize*sizeof(Point<DTYPE,SUMTYPE,MAX_DIM>), cudaMemcpyHostToDevice));

  LOG(SPTAG::Helper::LogLevel::LL_Debug, "Alloc'ing TPtree memory\n");
  TPtree<DTYPE,KEYTYPE,SUMTYPE,MAX_DIM>* tptree;
  CUDA_CHECK(cudaMallocManaged(&tptree, sizeof(TPtree<DTYPE,KEYTYPE,SUMTYPE, MAX_DIM>)));
  tptree->initialize(dataSize, levels);
//  KNN_blocks= max(tptree->num_leaves, BLOCKS);
  KNN_blocks= tptree->num_leaves;

  LOG(SPTAG::Helper::LogLevel::LL_Debug, "Alloc'ing memory for results on device: %lld bytes.\n", (long long int)dataSize*KVAL*sizeof(int));
  int* d_results;
  CUDA_CHECK(cudaMalloc(&d_results, (long long int)dataSize*KVAL*sizeof(int)));
  // Initialize results to all -1 (special value that is set to distance INFTY)
  CUDA_CHECK(cudaMemset(d_results, -1, (long long int)dataSize*KVAL*sizeof(int)));

  CUDA_CHECK(cudaDeviceSynchronize());

//  srand(time(NULL)); // random number seed for TP tree random hyperplane partitions
  srand(1); // random number seed for TP tree random hyperplane partitions


  double tree_time=0.0;
  double KNN_time=0.0;
  double refine_time = 0.0;
//  struct timespec start, end;
  time_t start_t, end_t;


  for(int tree_id=0; tree_id < trees; ++tree_id) { // number of TPTs used to create approx. KNN graph
  CUDA_CHECK(cudaDeviceSynchronize());

    LOG(SPTAG::Helper::LogLevel::LL_Debug, "TPT iteartion %d - ", tree_id);
    start_t = clock();
   // Create TPT
    tptree->reset();
    create_tptree<DTYPE, KEYTYPE, SUMTYPE,MAX_DIM>(tptree, d_points, dataSize, levels, 0, dataSize);
  CUDA_CHECK(cudaDeviceSynchronize());

    end_t = clock();

    tree_time += (double)(end_t-start_t)/CLOCKS_PER_SEC;

    start_t = clock();
   // Compute the KNN for each leaf node
/*
    if(graphtype == 0) {
      findKNN_leaf_nodes<DTYPE, KEYTYPE, SUMTYPE, MAX_DIM, THREADS><<<KNN_blocks,THREADS, sizeof(DistPair<SUMTYPE>) * (KVAL-1) * THREADS >>>(d_points, tptree, KVAL, d_results, metric);
    }
    else if(graphtype==1) {
      findRNG_leaf_nodes<DTYPE, KEYTYPE, SUMTYPE, MAX_DIM, THREADS><<<KNN_blocks,THREADS, sizeof(DistPair<SUMTYPE>) * (KVAL-1) * THREADS >>>(d_points, tptree, KVAL, d_results, metric);
    }
    else if(graphtype==2) {
      findRNG_strict<DTYPE, KEYTYPE, SUMTYPE, MAX_DIM, THREADS><<<KNN_blocks,THREADS, sizeof(DistPair<SUMTYPE>) * (KVAL) * THREADS >>>(d_points, tptree, KVAL, d_results, metric, 0, dataSize);
    }

  CUDA_CHECK(cudaDeviceSynchronize());
 */   

    //clock_gettime(CLOCK_MONOTONIC, &end);
    end_t = clock();

    KNN_time += (double)(end_t-start_t)/CLOCKS_PER_SEC;

    LOG(SPTAG::Helper::LogLevel::LL_Debug, "tree:%lfms, graph build:%lfms\n", tree_time, KNN_time);

  } // end TPT loop

  start_t = clock();

  if(refines > 0) { // Only call refinement if need to do at least 1 step
    refineGraphGPU<DTYPE, SUMTYPE, MAX_DIM>(index, d_points, d_results, dataSize, KVAL, initSize, refineDepth, refines, metric);
  }

  end_t = clock();
  refine_time += (double)(end_t-start_t)/CLOCKS_PER_SEC;


  LOG(SPTAG::Helper::LogLevel::LL_Debug, "%0.3lf, %0.3lf, %0.3lf, %0.3lf, ", tree_time, KNN_time, refine_time, tree_time+KNN_time+refine_time);
  CUDA_CHECK(cudaMemcpy(results, d_results, (long long int)dataSize*KVAL*sizeof(int), cudaMemcpyDeviceToHost));


  tptree->destroy();
  CUDA_CHECK(cudaFree(d_points));
  CUDA_CHECK(cudaFree(tptree));
  CUDA_CHECK(cudaFree(d_results));

}

/****************************************************************************************
 * Create graph on the GPU in a series of 1 or more batches, graph is saved into @results and is stored on the CPU
 * graphType: KNN=0, loose RNG=1, strict RNG=2
 * Note, vectors of MAX_DIM number dimensions are used, so an upper-bound must be determined
 * at compile time
 ***************************************************************************************/

template<typename DTYPE, typename SUMTYPE, int MAX_DIM>
void buildGraphGPU_Batch(SPTAG::VectorIndex* index, size_t dataSize, size_t KVAL, int trees, int* results, int graphtype, int leafSize, int NUM_GPUS, int balanceFactor) {

  int numDevicesOnHost;
  CUDA_CHECK(cudaGetDeviceCount(&numDevicesOnHost));

  if(numDevicesOnHost < NUM_GPUS) {
    LOG(SPTAG::Helper::LogLevel::LL_Error, "HeadNumGPUs parameter %d, but only %d devices available on system.  Exiting.\n", NUM_GPUS, numDevicesOnHost);
    exit(1);
  }

  LOG(SPTAG::Helper::LogLevel::LL_Info, "Building Head graph with %d GPUs...\n", NUM_GPUS);
  LOG(SPTAG::Helper::LogLevel::LL_Debug, "Total of %d GPU devices on system, using %d of them.\n", numDevicesOnHost, NUM_GPUS);

  double tree_time=0.0;
  double KNN_time=0.0;
  double D2H_time = 0.0;
  double prep_time = 0.0;

  auto start_t = std::chrono::high_resolution_clock::now();

  int dim = index->GetFeatureDim();
  int metric = (int)index->GetDistCalcMethod();

  DTYPE* data = (DTYPE*)index->GetSample(0);

  // Number of levels is based on the chosen leaf size
  int levels = (int)std::log2(dataSize/leafSize);

  int KNN_blocks; // number of threadblocks used

  Point<DTYPE,SUMTYPE,MAX_DIM>* points = convertMatrix<DTYPE,SUMTYPE,MAX_DIM>(index, dataSize, dim);

  for(size_t i=0;  i<dataSize; i++) {
    points[i].id = i;
  }

  std::vector<size_t> resPerGPU(NUM_GPUS);
  for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
    resPerGPU[gpuNum] = dataSize / NUM_GPUS;
    if(dataSize % NUM_GPUS > gpuNum) resPerGPU[gpuNum]++; 
  }
  std::vector<size_t> GPUOffset(NUM_GPUS);
  GPUOffset[0] = 0;
  LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU 0: results:%lu, offset:%lu\n", resPerGPU[0], GPUOffset[0]);
  for(int gpuNum=1; gpuNum < NUM_GPUS; ++gpuNum) {
    GPUOffset[gpuNum] = GPUOffset[gpuNum-1] + resPerGPU[gpuNum-1];
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU %d: results:%lu, offset:%lu\n", gpuNum, resPerGPU[gpuNum], GPUOffset[gpuNum]);
  }

  std::vector<cudaStream_t> streams(NUM_GPUS);
  std::vector<size_t> batchSize(NUM_GPUS);

  Point<DTYPE,SUMTYPE,MAX_DIM>** d_points = new Point<DTYPE,SUMTYPE,MAX_DIM>*[NUM_GPUS];
  TPtree<DTYPE,KEYTYPE,SUMTYPE,MAX_DIM>** tptrees = new TPtree<DTYPE,KEYTYPE,SUMTYPE,MAX_DIM>*[NUM_GPUS];
  TPtree<DTYPE,KEYTYPE,SUMTYPE,MAX_DIM>** d_tptrees = new TPtree<DTYPE,KEYTYPE,SUMTYPE,MAX_DIM>*[NUM_GPUS];

  int** d_results = new int*[NUM_GPUS];
  cudaError_t resultErr;

  // Allocate and initialize GPU data on each device
  for(int gpuNum=0; gpuNum < NUM_GPUS; gpuNum++) {
    CUDA_CHECK(cudaSetDevice(gpuNum));
    CUDA_CHECK(cudaStreamCreate(&streams[gpuNum]));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpuNum));

    LOG(SPTAG::Helper::LogLevel::LL_Info, "GPU %d - %s\n", gpuNum, prop.name);

    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

    // Auto-compute batch size based on available memory on the GPU
    size_t dataPointSize = dataSize*sizeof(Point<DTYPE,SUMTYPE,MAX_DIM>);
    size_t treeSize = 20*dataSize;
    size_t resMemAvail = (freeMem*0.9) - (dataPointSize+treeSize); // Only use 90% of total memory to be safe
    int maxEltsPerBatch = resMemAvail / (sizeof(Point<DTYPE,SUMTYPE,MAX_DIM>) + KVAL*sizeof(int));
    batchSize[gpuNum] = min(maxEltsPerBatch, (int)(resPerGPU[gpuNum]));

    // If GPU memory is insufficient or so limited that we need so many batches it becomes inefficient, return error
    if(batchSize[gpuNum] == 0 || ((int)resPerGPU[gpuNum]) / batchSize[gpuNum] > 10000) {
      LOG(SPTAG::Helper::LogLevel::LL_Error, "Insufficient GPU memory to build Head index on GPU %d.  Available GPU memory:%lu MB, Points and tpt require:%lu MB, leaving a maximum batch size of %d results to be computed, which is too small to run efficiently.\n", gpuNum, (freeMem)/1000000, (dataPointSize+treeSize)/1000000, maxEltsPerBatch);
      exit(1);
    }

    LOG(SPTAG::Helper::LogLevel::LL_Debug, "Memory for Points vectors:%lu MiB, Memory for TP trees:%lu MiB, Memory left for results:%lu MiB, total vectors:%lu, batch size:%d, total batches:%d\n", dataPointSize/1000000, treeSize/1000000, resMemAvail/1000000, resPerGPU[gpuNum], batchSize[gpuNum], (((batchSize[gpuNum]-1)+resPerGPU[gpuNum]) / batchSize[gpuNum]));


    // Allocate memory on GPUs and copy points to each GPU
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU:%d - Alloc'ing Points on device: %zu bytes and initializing TPTree memory.\n", gpuNum, dataSize*sizeof(Point<DTYPE,SUMTYPE,MAX_DIM>));
    CUDA_CHECK(cudaMalloc(&d_points[gpuNum], dataSize*sizeof(Point<DTYPE,SUMTYPE,MAX_DIM>)));
    CUDA_CHECK(cudaMemcpy(d_points[gpuNum], points, dataSize*sizeof(Point<DTYPE,SUMTYPE,MAX_DIM>), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_tptrees[gpuNum], sizeof(TPtree<DTYPE,KEYTYPE,SUMTYPE, MAX_DIM>)));
    tptrees[gpuNum] = new TPtree<DTYPE,KEYTYPE,SUMTYPE,MAX_DIM>;
    tptrees[gpuNum]->initialize(dataSize, levels);
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "TPT structure initialized for %lu points, %d levels, leaf size:%d\n", dataSize, levels, leafSize);

    CUDA_CHECK(cudaMallocManaged(&d_results[gpuNum], (size_t)batchSize[gpuNum]*KVAL*sizeof(int)));
  }

  // Set num blocks for all GPU kernel calls
  KNN_blocks= max(tptrees[0]->num_leaves, BLOCKS);

//  srand(time(NULL)); // random number seed for TP tree random hyperplane partitions
    srand(1); // random number seed for TP tree random hyperplane partitions
  
  std::vector<size_t> curr_batch_size(NUM_GPUS);
  std::vector<size_t> batchOffset(NUM_GPUS);
  for(int gpuNum=0; gpuNum<NUM_GPUS; ++gpuNum) {
    batchOffset[gpuNum]=0;
  }

  auto batch_start_t = std::chrono::high_resolution_clock::now();
  prep_time = ((double)std::chrono::duration_cast<std::chrono::seconds>(batch_start_t - start_t).count()) + (((double)std::chrono::duration_cast<std::chrono::milliseconds>(batch_start_t - start_t).count())/1000);

  bool done = false;
  while(!done) { // Continue until all GPUs have completed all of their batches
  
    // Prep next batch for each GPU
    for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
      curr_batch_size[gpuNum] = batchSize[gpuNum];
     // Check if final batch is smaller than previous
      if(batchOffset[gpuNum]+batchSize[gpuNum] > resPerGPU[gpuNum]) {
        curr_batch_size[gpuNum] = resPerGPU[gpuNum]-batchOffset[gpuNum];
      }
      LOG(SPTAG::Helper::LogLevel::LL_Debug, "GPU %d - starting batch with offset:%lu, with GPU offset:%lu, size:%lu, TailRows:%lld\n", gpuNum, batchOffset[gpuNum], GPUOffset[gpuNum], curr_batch_size[gpuNum], resPerGPU[gpuNum]);

      CUDA_CHECK(cudaSetDevice(gpuNum));
      // Initialize results to all -1 (special value that is set to distance INFTY)
      CUDA_CHECK(cudaMemset(d_results[gpuNum], -1, (size_t)batchSize[gpuNum]*KVAL*sizeof(int)));

    }

    CUDA_CHECK(cudaDeviceSynchronize());

    for(int tree_id=0; tree_id < trees; ++tree_id) { // number of TPTs used to create approx. KNN graph

auto before_tpt = std::chrono::high_resolution_clock::now();
      // Reset and create each TPT
      for(int gpuNum=0; gpuNum < NUM_GPUS; gpuNum++) {
        CUDA_CHECK(cudaSetDevice(gpuNum));
        tptrees[gpuNum]->reset();
      }
      create_tptree_multigpu<DTYPE, KEYTYPE, SUMTYPE, MAX_DIM>(tptrees, d_points, dataSize, levels, NUM_GPUS, streams.data(), balanceFactor);
      CUDA_CHECK(cudaDeviceSynchronize());

            // Copy TPTs to each GPU
      for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
          CUDA_CHECK(cudaSetDevice(gpuNum));
          CUDA_CHECK(cudaMemcpy(d_tptrees[gpuNum], tptrees[gpuNum], sizeof(TPtree<DTYPE,KEYTYPE,SUMTYPE,MAX_DIM>), cudaMemcpyHostToDevice));
      }

auto after_tpt = std::chrono::high_resolution_clock::now();

      for(int gpuNum=0; gpuNum < NUM_GPUS; gpuNum++) {

        CUDA_CHECK(cudaSetDevice(gpuNum));
        // Compute the STRICT RNG for each leaf node
        findRNG_strict<DTYPE, KEYTYPE, SUMTYPE, MAX_DIM, THREADS><<<KNN_blocks,THREADS, sizeof(DistPair<SUMTYPE>) * (KVAL) * THREADS>>>(d_points[gpuNum], d_tptrees[gpuNum], KVAL, d_results[gpuNum], metric, GPUOffset[gpuNum]+batchOffset[gpuNum], GPUOffset[gpuNum]+batchOffset[gpuNum]+curr_batch_size[gpuNum]);
      }
      CUDA_CHECK(cudaDeviceSynchronize());

auto after_work = std::chrono::high_resolution_clock::now();
 
      double loop_tpt_time = ((double)std::chrono::duration_cast<std::chrono::seconds>(after_tpt - before_tpt).count()) + (((double)std::chrono::duration_cast<std::chrono::milliseconds>(after_tpt - before_tpt).count())/1000);
      double loop_work_time = ((double)std::chrono::duration_cast<std::chrono::seconds>(after_work - after_tpt).count()) + (((double)std::chrono::duration_cast<std::chrono::milliseconds>(after_work - after_tpt).count())/1000);
      LOG(SPTAG::Helper::LogLevel::LL_Debug, "All GPUs finished tree %d - tree build time:%.2lf, neighbor compute time:%.2lf\n", tree_id, loop_tpt_time, loop_work_time);
      tree_time += loop_tpt_time;
      KNN_time += loop_work_time;

    } // TPT Loop

auto before_copy = std::chrono::high_resolution_clock::now();
    for(int gpuNum=0; gpuNum < NUM_GPUS; gpuNum++) {
      CUDA_CHECK(cudaMemcpy(&results[(GPUOffset[gpuNum]+batchOffset[gpuNum])*KVAL], d_results[gpuNum], curr_batch_size[gpuNum]*KVAL*sizeof(int), cudaMemcpyDeviceToHost));
    }
auto after_copy = std::chrono::high_resolution_clock::now();
    
    double batch_copy_time = ((double)std::chrono::duration_cast<std::chrono::seconds>(after_copy - before_copy).count()) + (((double)std::chrono::duration_cast<std::chrono::milliseconds>(after_copy - before_copy).count())/1000);
    LOG(SPTAG::Helper::LogLevel::LL_Debug, "All GPUs finished batch - time to copy result:%.2lf\n", batch_copy_time);
    D2H_time += batch_copy_time;

    // Update all batchOffsets and check if done
    done=true;
    for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
      batchOffset[gpuNum] += curr_batch_size[gpuNum];
      if(batchOffset[gpuNum] < resPerGPU[gpuNum]) {
        done=false;
      }
    }
  } // Batches loop (while !done) 

auto end_t = std::chrono::high_resolution_clock::now();

  LOG(SPTAG::Helper::LogLevel::LL_Info, "Total times - prep time:%0.3lf, tree build:%0.3lf, neighbor compute:%0.3lf, Copy results:%0.3lf, Total runtime:%0.3lf\n", prep_time, tree_time, KNN_time, D2H_time, ((double)std::chrono::duration_cast<std::chrono::seconds>(end_t - start_t).count()) + (((double)std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t).count())/1000));

  for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
    tptrees[gpuNum]->destroy();
    delete tptrees[gpuNum];
    cudaFree(d_tptrees[gpuNum]);
    cudaFree(d_points[gpuNum]);
    cudaFree(d_results[gpuNum]);
  }
  delete[] tptrees;
  delete[] d_points;
  delete[] d_results;
  delete[] d_tptrees;
}

/***************************************************************************************
 * Function called by SPTAG to create an initial graph on the GPU.  
 ***************************************************************************************/
template<typename T>
void buildGraph(SPTAG::VectorIndex* index, int m_iGraphSize, int m_iNeighborhoodSize, int trees, int* results, int refines, int refineDepth, int graph, int leafSize, int initSize, int NUM_GPUS, int balanceFactor) {

  int m_iFeatureDim = index->GetFeatureDim();
  int m_disttype = (int)index->GetDistCalcMethod();

  // Make sure that neighborhood size is a power of 2
  if(m_iNeighborhoodSize == 0 || (m_iNeighborhoodSize & (m_iNeighborhoodSize-1)) != 0) {
    LOG(SPTAG::Helper::LogLevel::LL_Error, "NeighborhoodSize (with scaling factor applied) is %d but must be a power of 2 for GPU construction.\n", m_iNeighborhoodSize);
    exit(1);
  }

  // Have to give compiler-time known bounds on dimensions so that we can store points in registers
  // This significantly speeds up distance comparisons.
  // Create other options here for other commonly-used dimension values.
  // TODO: Create slower, non-register version that can be used for very high-dimensional data
  if(m_iFeatureDim > 100) {
    LOG(SPTAG::Helper::LogLevel::LL_Error, ">100 dimensions not currently supported for GPU construction.\n");
    exit(1);
  }
  if(typeid(T) == typeid(float)) {
    if(m_iFeatureDim <= 64) {
      buildGraphGPU_Batch<T, float, 64>(index, (size_t)m_iGraphSize, (size_t)m_iNeighborhoodSize, trees, results, graph, leafSize, NUM_GPUS, balanceFactor);
    }
    else {
      buildGraphGPU_Batch<T, float, 100>(index, (size_t)m_iGraphSize, (size_t)m_iNeighborhoodSize, trees, results, graph, leafSize, NUM_GPUS, balanceFactor);
    }
  }
  else if(typeid(T) == typeid(uint8_t) || typeid(T) == typeid(int8_t)) {
    if(m_iFeatureDim <= 64) {
      buildGraphGPU_Batch<T, int32_t, 64>(index, (size_t)m_iGraphSize, (size_t)m_iNeighborhoodSize, trees, results, graph, leafSize, NUM_GPUS, balanceFactor);
    }
    else {
      buildGraphGPU_Batch<T, int32_t, 100>(index, (size_t)m_iGraphSize, (size_t)m_iNeighborhoodSize, trees, results, graph, leafSize, NUM_GPUS, balanceFactor);
    }
  }
  else {
    LOG(SPTAG::Helper::LogLevel::LL_Error, "Selected datatype not currently supported.\n");
    exit(1);
  }
}

#endif
