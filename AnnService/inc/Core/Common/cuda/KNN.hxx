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

#include "../../VectorIndex.h"
#include "ThreadHeap.hxx"
#include "TPtree.hxx"
#include "Distance.hxx"

//#include<cub/cub.cuh>

#include "/home/bkarsin/cub-1.8.0/cub/cub.cuh"


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
 * Refine KNN graph using neighbors' neighbors lookup process
 * Significantly improves accuracy once the approximate KNN is created.
 * DEPTH macro controls the depth of refinement that is performed.
 *****************************************************************************************/
template<typename T, typename SUMTYPE, int Dim, int BLOCK_DIM>
__global__ void refine_KNN(Point<T,SUMTYPE,Dim>* data, int* results, int N, int KVAL, int metric) {

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
//    max_K.idx = results[((long long int)query.id)*KVAL]; 
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
            if(!dup && (target.dist < nearest_dist || compute_accessibility<T>(results, query.id, tptree->leaf_points[leaf_offset+j], KVAL) == 0)) { 
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
__global__ void refine_RNG(Point<T,SUMTYPE,Dim>* data, int* results, int N, int KVAL, int metric) {

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

using namespace SPTAG;

template <typename SUMTYPE>
class ListElt {
  public:
  int id;
  SUMTYPE dist;
  bool checkedFlag;

  __device__ ListElt& operator=(const ListElt& other) {
    id = other.id;
    dist = other.dist;
    checkedFlag = other.checkedFlag;
    return *this;
  }
};



template<typename T>
void getCandidates(SPTAG::VectorIndex* index, int numVectors, int candidatesPerVector, int* candidates) {

#pragma omp parallel for schedule(dynamic)
  for (SizeType i = 0; i < numVectors; i++)
  {
    SPTAG::COMMON::QueryResultSet<T> query((const T*)index->GetSample(i), candidatesPerVector);
     index->SearchTree(query);
     for (SPTAG::DimensionType j = 0; j < candidatesPerVector; j++) {
       candidates[i*candidatesPerVector+j] = query.GetResult(j)->VID;
     }
  }
}

template<typename SUMTYPE, int NUM_REGS, int NUM_THREADS>
__device__ void loadRegisters(ListElt<SUMTYPE>* regs, ListElt<SUMTYPE>* listMem, int* listSize) {
  for(int i=0; i<NUM_REGS; i++) {
    if(i*NUM_THREADS + threadIdx.x < *listSize) {
      regs[i] = listMem[i*NUM_THREADS + threadIdx.x];
    }
    else {
      regs[i].id = INFTY<int>();
      regs[i].dist = INFTY<SUMTYPE>();
    }
  }
}

template<typename SUMTYPE, int NUM_REGS, int NUM_THREADS>
__device__ void storeRegisters(ListElt<SUMTYPE>* regs, ListElt<SUMTYPE>* listMem, int* listSize) {
  for(int i=0; i<NUM_REGS; i++) {
    if(i*NUM_THREADS + threadIdx.x < *listSize) {
      listMem[i*NUM_THREADS + threadIdx.x] = regs[i];
    }
    else {
      listMem[i*NUM_THREADS+threadIdx.x].id = INFTY<int>();
      listMem[i*NUM_THREADS+threadIdx.x].dist = INFTY<int>();
    }
  }
}


#define LISTCAP 2048
#define LISTSIZE 1024

template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__device__ void sortListById(ListElt<SUMTYPE>* listMem, int* listSize, void* temp_storage) {

  ListElt<SUMTYPE> sortMem[LISTCAP/NUM_THREADS];
  int sortKeys[LISTCAP/NUM_THREADS];

// Sort list by ID to remove duplicates
  // Load list into registers to sort
  loadRegisters<SUMTYPE,LISTCAP/NUM_THREADS, NUM_THREADS>(sortMem, listMem, listSize);

  for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
    sortKeys[i] = sortMem[i].id;
  }

  // Sort by ID in registers
  typedef cub::BlockRadixSort<int, NUM_THREADS, LISTCAP/NUM_THREADS, ListElt<SUMTYPE>> BlockRadixSort;
  BlockRadixSort(*(static_cast<typename BlockRadixSort::TempStorage*>(temp_storage))).SortBlockedToStriped(sortKeys, sortMem);

  __syncthreads();


  // Write back to list
  storeRegisters<SUMTYPE,LISTCAP/NUM_THREADS, NUM_THREADS>(sortMem, listMem, listSize);
  __syncthreads();

}

template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__device__ void sortListByDist(ListElt<SUMTYPE>* listMem, int* listSize, void* temp_storage) {

  ListElt<SUMTYPE> sortMem[LISTCAP/NUM_THREADS];
  SUMTYPE sortKeys[LISTCAP/NUM_THREADS];

// Sort list by ID to remove duplicates
  // Load list into registers to sort
  loadRegisters<SUMTYPE,LISTCAP/NUM_THREADS, NUM_THREADS>(sortMem, listMem, listSize);

  for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
    sortKeys[i] = sortMem[i].dist;
  }

  // Sort by ID in registers
  typedef cub::BlockRadixSort<SUMTYPE, NUM_THREADS, LISTCAP/NUM_THREADS, ListElt<SUMTYPE>> BlockRadixSort;
  BlockRadixSort(*(static_cast<typename BlockRadixSort::TempStorage*>(temp_storage))).SortBlockedToStriped(sortKeys, sortMem);
  __syncthreads();

  // Write back to list
  storeRegisters<SUMTYPE,LISTCAP/NUM_THREADS, NUM_THREADS>(sortMem, listMem, listSize);
  __syncthreads();
}



// Remove duplicates and compact list with prefix sums
template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__device__ void removeDuplicatesAndCompact(ListElt<SUMTYPE>* listMem, int* listSize, void *temp_storage, int src, int* /*borderVals*/) {

  ListElt<SUMTYPE> sortMem[LISTCAP/NUM_THREADS];
  int sortKeys[LISTCAP/NUM_THREADS];

  // Copy weather is duplicate or not into registers
  for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
    sortMem[i] = listMem[threadIdx.x*(LISTCAP/NUM_THREADS) + i];
    sortKeys[i] = 0;
    if(listMem[threadIdx.x*(LISTCAP/NUM_THREADS) + i].id == -1) {
      sortKeys[i] = 0;
    }
    else if(i==0 && threadIdx.x==0) {
      sortKeys[i] = 1;
    }
    else if(threadIdx.x*(LISTCAP/NUM_THREADS) + i < *listSize && listMem[threadIdx.x*(LISTCAP/NUM_THREADS) + i].id != INFTY<int>()) {
      sortKeys[i] = (sortMem[i].id != listMem[threadIdx.x*(LISTCAP/NUM_THREADS) + i - 1].id);
    }
  }

__syncthreads();
for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
	listMem[threadIdx.x*(LISTCAP/NUM_THREADS) + i] = sortMem[i];
//	listMem[threadIdx.x*(LISTCAP/NUM_THREADS) + i].checkedFlag = sortKeys[i];
}

  __syncthreads();
  typedef cub::BlockScan<int, NUM_THREADS> BlockScan;
  BlockScan(*(static_cast<typename BlockScan::TempStorage*>(temp_storage))).InclusiveSum(sortKeys, sortKeys);

  __syncthreads();
  for(int i=0; i<LISTCAP/NUM_THREADS; i++) {
    listMem[threadIdx.x*(LISTCAP/NUM_THREADS)+i] = sortMem[i];
  }

  // Share boarders of prefix sums
  __shared__ int borderVals[NUM_THREADS];
  borderVals[threadIdx.x] = sortKeys[LISTCAP/NUM_THREADS - 1];

  __syncthreads();

  if(threadIdx.x==0 || borderVals[threadIdx.x-1] != sortMem[0].id) {
    listMem[sortKeys[0]-1] = sortMem[0];
  }
  for(int i=1; i<LISTCAP/NUM_THREADS; i++) {
    if(sortKeys[i] > sortKeys[i-1]) {
      listMem[sortKeys[i]-1] = sortMem[i];
    }
  }
  __syncthreads();
  *listSize = borderVals[NUM_THREADS-1];
  __syncthreads(); 

}


#define MAX_CHECK_COUNT 32
template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__device__ void checkClosestNeighbors(Point<T,SUMTYPE,MAX_DIM>* d_points, int src, int* d_graph, ListElt<SUMTYPE>* listMem, int* listSize, int KVAL, int metric) {

// Maximum number of vertices to check before list fills up
  int max_check = min(MAX_CHECK_COUNT, (LISTCAP-*listSize)/KVAL);

  int write_offset;
  __shared__ int check_count;
  check_count=0;
  __shared__ int new_listSize;
  new_listSize = *listSize;
  __syncthreads();
  
// Fill up list with neighbors of nearest unchecked vectices
  for(int i=threadIdx.x; i<LISTSIZE && check_count < max_check; i+=NUM_THREADS) {
    if(!listMem[i].checkedFlag) {
      if(atomicAdd(&check_count, 1) < max_check) {
        write_offset = atomicAdd(&new_listSize, KVAL);
        listMem[i].checkedFlag=true;
        for(int j=0; j<KVAL; j++) {
//          listMem[write_offset+j].id = d_graph[i*KVAL+j];
          listMem[write_offset+j].id = d_graph[i*KVAL+j];
        }
      }
    }
  }
  __syncthreads();
// Compute distance to all newly added vetrices
  if(metric == 0) {
    for(int i=*listSize+threadIdx.x; i<new_listSize; i+=NUM_THREADS) {
      if(listMem[i].id == src) {
        listMem[i].id = -1;
        listMem[i].dist = INFTY<SUMTYPE>();
      }
      else if (metric == 0){
        listMem[i].dist = d_points[src].l2(&d_points[listMem[i].id]);
      }
      else {
        listMem[i].dist = d_points[src].cosine(&d_points[listMem[i].id]);
      }
    }
  }
  __syncthreads();
  *listSize = new_listSize;
  __syncthreads();

}

template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__device__ void shrinkListRNG_OLD(Point<T,SUMTYPE,MAX_DIM>* d_points, int* d_graph, int src, ListElt<SUMTYPE>* listMem, int listSize, int KVAL, int metric) {
  bool good=true;
  int write_idx=0;
  int read_idx=0;

  // Test sequential version of this...
  if(threadIdx.x==0) {
    for(read_idx=0; write_idx < KVAL && read_idx < listSize; read_idx++) {
      if(listMem[read_idx].id == -1) break;
      if(listMem[read_idx].id == src) continue;

      good = true;
      if(metric == 0) {
      for(int j=0; j<write_idx; j++) {
        // If it violates RNG
        if(listMem[read_idx].dist >= d_points[listMem[read_idx].id].l2(&d_points[d_graph[src*KVAL+j]])) {
          good=false;
          continue;
        }
      }
      }
      else {
      for(int j=0; j<write_idx; j++) {
        // If it violates RNG
        if(listMem[read_idx].dist >= d_points[listMem[read_idx].id].cosine(&d_points[d_graph[src*KVAL+j]])) {
          good=false;
          continue;
        }
      }
      }
      if(good) {
        d_graph[src*KVAL+write_idx] = listMem[read_idx].id;
        write_idx++;
      }
    }
    if(write_idx < KVAL) {
      for(int i=write_idx; i<KVAL; i++) {
        d_graph[src*KVAL+i] = -1;
      }
    }
  }
  __syncthreads();
}

template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__device__ void shrinkListRNG(Point<T,SUMTYPE,MAX_DIM>* d_points, int* d_graph, int src, ListElt<SUMTYPE>* listMem, int listSize, int KVAL, int metric) {

  __shared__ bool good;
  int write_idx=0;
  int read_idx=0;

  for(read_idx=0; write_idx < KVAL && read_idx < listSize; read_idx++) {
/*
    if(listMem[read_idx].id == -1) break;
    if(listMem[read_idx].id == src) continue;
*/
	if(listMem[read_idx].id == -1 || listMem[read_idx].id == src) {
		good = false;
	}
	else {
	    good = true;
	}
    __syncthreads();

    if(metric == 0) {
    for(int j=threadIdx.x; j<write_idx && good; j+=NUM_THREADS) {
      // If it violates RNG
      if(listMem[read_idx].dist >= d_points[listMem[read_idx].id].l2(&d_points[d_graph[src*KVAL+j]])) {
        good=false;
      }
    }
    }
    else {
    for(int j=threadIdx.x; j<write_idx && good; j+=NUM_THREADS) {
      // If it violates RNG
      if(listMem[read_idx].dist >= d_points[listMem[read_idx].id].cosine(&d_points[d_graph[src*KVAL+j]])) {
        good=false;
      }
    }
    }
    __syncthreads();
    if(good) {
      write_idx++;
      d_graph[src*KVAL+write_idx] = listMem[read_idx].id;
    }
    __syncthreads();
  }
  if(write_idx < KVAL) {
    for(int i=write_idx+threadIdx.x; i<KVAL; i+=NUM_THREADS) {
      d_graph[src*KVAL+i] = -1;
    }
  }
  __syncthreads();
}

#define TEST_THREADS 64
#define TEST_BLOCKS 1024
template<typename T, typename SUMTYPE, int MAX_DIM, int NUM_THREADS>
__global__ void refineBatchGPU(Point<T,SUMTYPE,MAX_DIM>* d_points, int batchSize, int batchOffset, int* d_graph, int* candidates, ListElt<SUMTYPE>* listMemAll, int candidatesPerVector, int KVAL, int refineDepth, int metric) {

if(threadIdx.x==0 && blockIdx.x==0) {
printf("refineBatchGPU! batchSize:%d, batchOffset:%d\n", batchSize, batchOffset);
}

  // Offset to the memory allocated for this block's list
  ListElt<SUMTYPE>* listMem = &listMemAll[blockIdx.x*LISTCAP];

  __shared__ int listSize;
  __shared__ int borderVals[NUM_THREADS];
  typedef cub::BlockRadixSort<SUMTYPE, NUM_THREADS, LISTCAP/NUM_THREADS, ListElt<SUMTYPE>> BlockRadixSortT;
  __shared__ typename BlockRadixSortT::TempStorage temp_storage;

  for(int src=blockIdx.x+batchOffset; src<batchOffset+batchSize; src+=gridDim.x) {

    // Initialize listMem for the source vertex
    for(int i=threadIdx.x; i<KVAL; i+=NUM_THREADS) {
      listMem[i].id = d_graph[src*KVAL+i];
      if(metric=0) {
        listMem[i].dist = d_points[src].l2(&d_points[listMem[i].id]);
      }
      else {
        listMem[i].dist = d_points[src].cosine(&d_points[listMem[i].id]);
      }
      listMem[i].checkedFlag = false;
    }
    listSize = KVAL;

    for(int i=threadIdx.x; i<candidatesPerVector; i+=NUM_THREADS) {
      listMem[i+KVAL].id = candidates[src*candidatesPerVector+i];
      if(metric == 0) {
        listMem[i+KVAL].dist = d_points[src].l2(&d_points[listMem[i+KVAL].id]);
      }
      else {
        listMem[i+KVAL].dist = d_points[src].cosine(&d_points[listMem[i+KVAL].id]);
      }
      listMem[i+KVAL].checkedFlag = false;
      listSize = KVAL+candidatesPerVector;
    }
    
    __syncthreads();
    sortListByDist<T,SUMTYPE,MAX_DIM,NUM_THREADS>(listMem, &listSize, &temp_storage);

if(src==0 && threadIdx.x==0) {
	for(int i=0; i<KVAL; i++) {
		printf("%d (%0.3f), ", listMem[i].id, listMem[i].dist);
	}
printf("\n");
}

    for(int iter=0; iter < refineDepth; iter++) {
      listSize = min(LISTSIZE, listSize);
      __syncthreads();
      checkClosestNeighbors<T,SUMTYPE,MAX_DIM,NUM_THREADS>(d_points, src, d_graph, listMem, &listSize, KVAL, metric);
/*
      sortListByDist<T,SUMTYPE,MAX_DIM,NUM_THREADS>(listMem, &listSize, &temp_storage);
__syncthreads();
if(src==0 && threadIdx.x==0) {
printf("listSize:%d\n", listSize);
	for(int i=0; i<KVAL; i++) {
		printf("%d (%0.3f), ", listMem[i].id, listMem[i].dist);
	}
printf("\n");
}
__syncthreads();
*/

      sortListById<T,SUMTYPE,MAX_DIM,NUM_THREADS>(listMem, &listSize, &temp_storage);
      removeDuplicatesAndCompact<T,SUMTYPE,MAX_DIM,NUM_THREADS>(listMem, &listSize, &temp_storage, src, borderVals);
      sortListByDist<T,SUMTYPE,MAX_DIM,NUM_THREADS>(listMem, &listSize, &temp_storage);
      __syncthreads();
    }

if(src==0 && threadIdx.x==0) {
	for(int i=0; i<KVAL; i++) {
		printf("%d (%0.3f), ", listMem[i].id, listMem[i].dist);
	}
printf("\n");
}
__syncthreads();
    // Prune nearest RNG vectors and write them to d_graph
//    shrinkListRNG<T,SUMTYPE,MAX_DIM,NUM_THREADS>(d_points, d_graph, src, listMem, listSize, KVAL, metric);
    for(int i=threadIdx.x; i<KVAL; i+=NUM_THREADS) {
      d_graph[src*KVAL+i] = listMem[i].id;
    }
  }
}


template<typename T, typename SUMTYPE, int MAX_DIM>
void refineGraphGPU(SPTAG::VectorIndex* index, Point<T,SUMTYPE,MAX_DIM>* d_points, int* d_graph, int dataSize, int KVAL, int candidatesPerVector, int refineDepth, int refines, int metric) {

  auto t1 = std::chrono::high_resolution_clock::now();
  int* candidates = (int*)malloc(dataSize*candidatesPerVector*sizeof(int));
  getCandidates<T>(index, dataSize, candidatesPerVector, candidates);

  int* d_candidates;
  cudaMalloc(&d_candidates, dataSize*candidatesPerVector*sizeof(int));
  cudaMemcpy(d_candidates, candidates, dataSize*candidatesPerVector*sizeof(int), cudaMemcpyHostToDevice);

  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "find candidates time (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;

  int NUM_BATCHES = 1;
  int batch_size = dataSize/NUM_BATCHES;

  ListElt<SUMTYPE>* listMem;

  cudaMalloc(&listMem, TEST_BLOCKS*LISTCAP*sizeof(ListElt<SUMTYPE>));

  t1 = std::chrono::high_resolution_clock::now();
  for(int iter=0; iter < refines; iter++) {

    for(int i=0; i<NUM_BATCHES; i++) {
      refineBatchGPU<T,SUMTYPE,MAX_DIM, TEST_THREADS><<<TEST_BLOCKS,TEST_THREADS>>>(d_points, batch_size, i*batch_size, d_graph, d_candidates, listMem, candidatesPerVector, KVAL, refineDepth, metric);
      cudaDeviceSynchronize();
    }
  }
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "GPU refine time (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;

  cudaFree(listMem);
  cudaFree(d_candidates);
  free(candidates);
}


/****************************************************************************************
 * Create either graph on the GPU, graph is saved into @results and is stored on the CPU
 * graphType: KNN=0, RNG=1
 * Note, vectors of MAX_DIM number dimensions are used, so an upper-bound must be determined
 * at compile time
 ***************************************************************************************/
template<typename DTYPE, typename SUMTYPE, int MAX_DIM>
//void buildGraphGPU(DTYPE* data, int dataSize, int dim, int KVAL, int trees, int* results, int metric, int refines, int graphtype) {
void buildGraphGPU(SPTAG::VectorIndex* index, int dataSize, int KVAL, int trees, int* results, int refines, int graphtype, int initSize, int refineDepth) {

  int dim = index->GetFeatureDim();
  int metric = (int)index->GetDistCalcMethod();
  DTYPE* data = (DTYPE*)index->GetSample(0);

  // Number of levels set to have approximately 500 points per leaf
  int levels = (int)std::log2(dataSize/500);

  int KNN_blocks; // number of threadblocks used

  printf("Copying to Point array\n");
  Point<DTYPE,SUMTYPE,MAX_DIM>* points = convertMatrix<DTYPE,SUMTYPE,MAX_DIM>(data, dataSize, dim);

  for(int i=0;  i<dataSize; i++) {
    points[i].id = i;
  }

  Point<DTYPE, SUMTYPE, MAX_DIM>* d_points;
  LOG("Alloc'ing Points on device: %ld bytes.\n", dataSize*sizeof(Point<DTYPE,SUMTYPE,MAX_DIM>));
  cudaMalloc(&d_points, dataSize*sizeof(Point<DTYPE,SUMTYPE,MAX_DIM>));

  LOG("Copying to device.\n");
  cudaMemcpy(d_points, points, dataSize*sizeof(Point<DTYPE,SUMTYPE,MAX_DIM>), cudaMemcpyHostToDevice);

  LOG("Alloc'ing TPtree memory\n");
  TPtree<DTYPE,KEYTYPE,SUMTYPE,MAX_DIM>* tptree;
  cudaMallocManaged(&tptree, sizeof(TPtree<DTYPE,KEYTYPE,SUMTYPE, MAX_DIM>));
  tptree->initialize(dataSize, levels);
  KNN_blocks= max(tptree->num_leaves, BLOCKS);

  LOG("Alloc'ing memory for results on device: %lld bytes.\n", (long long int)dataSize*KVAL*sizeof(int));
  int* d_results;
  cudaMalloc(&d_results, (long long int)dataSize*KVAL*sizeof(int));
  // Initialize results to all -1 (special value that is set to distance INFTY)
  cudaMemset(d_results, -1, (long long int)dataSize*KVAL*sizeof(int));

  cudaDeviceSynchronize();

//  srand(time(NULL)); // random number seed for TP tree random hyperplane partitions
  srand(1); // random number seed for TP tree random hyperplane partitions


  double tree_time=0.0;
  double KNN_time=0.0;
  double refine_time = 0.0;
  struct timespec start, end;
  time_t start_t, end_t;


  for(int tree_id=0; tree_id < trees; ++tree_id) { // number of TPTs used to create approx. KNN graph
    cudaDeviceSynchronize();

    LOG("Starting TPT construction timer\n");
    //clock_gettime(CLOCK_MONOTONIC, &start);
    start_t = clock();
   // Create TPT
    tptree->reset();
    create_tptree<DTYPE, KEYTYPE, SUMTYPE,MAX_DIM>(tptree, d_points, dataSize, levels);
    cudaDeviceSynchronize();

    //clock_gettime(CLOCK_MONOTONIC, &end);
    end_t = clock();

    tree_time += (double)(end_t-start_t)/CLOCKS_PER_SEC;
    //LOG("TPT construction time (ms): %lf\n", (1000*end.tv_sec + 1e-6*end.tv_nsec) - (1000*start.tv_sec + 1e-6*start.tv_nsec));


    //clock_gettime(CLOCK_MONOTONIC, &start);
    start_t = clock();
   // Compute the KNN for each leaf node
    if(graphtype == 0) {
      findKNN_leaf_nodes<DTYPE, KEYTYPE, SUMTYPE, MAX_DIM, THREADS><<<KNN_blocks,THREADS, sizeof(DistPair<SUMTYPE>) * (KVAL-1) * THREADS >>>(d_points, tptree, KVAL, d_results, metric);
    }
    /*
    else {
      findRNG_leaf_nodes<DTYPE, KEYTYPE, SUMTYPE, MAX_DIM, THREADS><<<KNN_blocks,THREADS, sizeof(DistPair<SUMTYPE>) * (KVAL-1) * THREADS >>>(d_points, tptree, KVAL, d_results, metric);
    }
    */
    cudaDeviceSynchronize();

    //clock_gettime(CLOCK_MONOTONIC, &end);
    end_t = clock();

    KNN_time += (double)(end_t-start_t)/CLOCKS_PER_SEC;
    //LOG("KNN Leaf time (ms): %lf\n", (1000*end.tv_sec + 1e-6*end.tv_nsec) - (1000*start.tv_sec + 1e-6*start.tv_nsec));
  } // end TPT loop

  //clock_gettime(CLOCK_MONOTONIC, &start);
  start_t = clock();

  /*
  for(int r=0; r<refines; r++) {
  // Perform a final refinement step of KNN graph
    if(graphtype == 0) {
      refine_KNN<DTYPE, SUMTYPE, MAX_DIM, THREADS><<<KNN_blocks,THREADS, sizeof(DistPair<SUMTYPE>) * (KVAL-1) * THREADS >>>(d_points, d_results, dataSize, KVAL, (int)metric);
    }
    else {
      refine_RNG<DTYPE, SUMTYPE, MAX_DIM, THREADS><<<KNN_blocks,THREADS, KVAL*THREADS*sizeof(int)*2>>>(d_points, d_results, dataSize, KVAL, (int)metric);
    }
    cudaDeviceSynchronize();
  }
  */

  //clock_gettime(CLOCK_MONOTONIC, &end);
  end_t = clock();
  refine_time += (double)(end_t-start_t)/CLOCKS_PER_SEC;

  if(refines > 0) {
  refineGraphGPU<DTYPE, SUMTYPE, MAX_DIM>(index, d_points, d_results, dataSize, KVAL, initSize, refineDepth, refines, metric);
  }

printf("Done building, copying memory back to CPU...\n");

  LOG("%0.3lf, %0.3lf, %0.3lf, %0.3lf, ", tree_time, KNN_time, refine_time, tree_time+KNN_time+refine_time);
  cudaMemcpy(results, d_results, (long long int)dataSize*KVAL*sizeof(int), cudaMemcpyDeviceToHost);

  tptree->destroy();
  cudaFree(tptree);
  cudaFree(d_points);
  cudaFree(tptree);
  cudaFree(d_results);

}


/***************************************************************************************
 * Function called by SPTAG to create an initial graph on the GPU.  
 ***************************************************************************************/
template<typename T>
//void buildGraph(T* data, int m_iFeatureDim, int m_iGraphSize, int m_iNeighborhoodSize, int trees, int* results, int m_disttype, int refines, int graph) {
void buildGraph(SPTAG::VectorIndex* index, int m_iGraphSize, int m_iNeighborhoodSize, int trees, int* results, int refines, int refineDepth, int graph, int initSize) {

  int m_iFeatureDim = index->GetFeatureDim();
  int m_disttype = (int)index->GetDistCalcMethod();
//  T* data = (T*)index->GetSample(0);

  // Make sure that neighborhood size is a power of 2
  if(m_iNeighborhoodSize == 0 || (m_iNeighborhoodSize & (m_iNeighborhoodSize-1)) != 0) {
    std::cout << "NeighborhoodSize (with scaling factor applied) is " << m_iNeighborhoodSize << " but must be a power of 2 for GPU construction." << std::endl;
    exit(1);
  }

  // Have to give compiler-time known bounds on dimensions so that we can store points in registers
  // This significantly speeds up distance comparisons.
  // Create other options here for other commonly-used dimension values.
  // TODO: Create slower, non-register version that can be used for very high-dimensional data
  if(m_iFeatureDim > 100) {
    std::cout << ">100 dimensions not currently supported for GPU construction." << std::endl;
    exit(1);
  }
  else {
    
    if(m_disttype == 1 || typeid(T) == typeid(float)) {
      buildGraphGPU<T, float, 100>(index, m_iGraphSize, m_iNeighborhoodSize, trees, results, refines, graph, initSize, refineDepth);
    }
/*
    else {
      if(typeid(T) == typeid(uint8_t) || typeid(T) == typeid(int8_t) || typeid(T) == typeid(int16_t) || typeid(T) == typeid(uint16_t)) {
        buildGraphGPU<T, int32_t, 100>(index, m_iGraphSize, m_iNeighborhoodSize, trees, results, refines, graph, initSize, refineDepth);
      }
    }
*/
  }
}

#endif
