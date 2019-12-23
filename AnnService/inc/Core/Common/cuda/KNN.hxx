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

#ifndef _SPTAG_COMMON_CUDA_KNN_H_
#define _SPTAG_COMMON_CUDA_KNN_H_

#include "ThreadHeap.hxx"
#include "TPtree.hxx"
#include "Distance.hxx"

template<typename T, int Stride>
__device__ void loadTransposePoint(T* dst, T* src, int dim) {
    for (int i = 0; i < dim; i++)
        dst[i*Stride] = src[i];
}


/************************************************************************************
* Brute-force K nearest neighbor kernel using shared memory (transposed to avoid conflicts)
* Each thread keeps a heap of K elements to determine K smallest distances found
* VAR data - linear matrix fo data
* VAR queries - linear matrix of query vectors
* RET results - linear vector of K pairs for each query vector
************************************************************************************/
template<typename T, int BLOCK_DIM>
__global__ void findKNN_SMEM_transpose(T* data, int dim, int dataSize, int query_offset, int numQueries, int KVAL, DistPair* results, int distance_type) {

    float(*f)(T* a, T* b, int length);
    if (distance_type == 0) f = l2<T, BLOCK_DIM>;
    else f = cosine<T, BLOCK_DIM>;

    extern __shared__ char sharememory[];

    __shared__ ThreadHeap<T, BLOCK_DIM> heapMem[BLOCK_DIM];

    heapMem[threadIdx.x].initialize(KVAL - 1, (DistPair*)(sharememory + sizeof(DistPair) * (KVAL - 1) * threadIdx.x));

    T* transpose_mem = (T*)(sharememory + sizeof(DistPair) * (KVAL - 1) * BLOCK_DIM);

    DistPair extra; // extra variable to store the largest distance/id for all KNN of the point

    float dist;
    // Loop through all query points
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < numQueries; i += blockDim.x*gridDim.x) {

        heapMem[threadIdx.x].reset();
        extra.dist = FLT_MAX;
        loadTransposePoint<T, BLOCK_DIM>(&transpose_mem[threadIdx.x], data + (query_offset + i) * (long long int)dim, dim);

        // Compare with all points in the dataset
        for (int j = 0; j < dataSize; j++) {
            if (j != (i + query_offset)) {
                dist = f(&transpose_mem[threadIdx.x], data + j * (long long int)dim, dim);
                if (dist < extra.dist) {
                    if (dist < heapMem[threadIdx.x].top()) {
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
        results[(i + 1)*KVAL - 1].idx = extra.idx;
        results[(i + 1)*KVAL - 1].dist = extra.dist;
        for (int j = KVAL - 2; j >= 0; j--) {
            results[i*KVAL + j].idx = heapMem[threadIdx.x].vals[0].idx;
            results[i*KVAL + j].dist = heapMem[threadIdx.x].vals[0].dist;
            heapMem[threadIdx.x].vals[0].dist = -1;
            heapMem[threadIdx.x].heapify();
        }
    }
}



/*****************************************************************************************
* Perform brute-force KNN on each leaf node, where only list of point ids is stored as leafs.
* Returns for each point: the K nearest neighbors within the leaf node containing it.
*      ** Memory footprint reduced compared with brute-force approach (above) **
*****************************************************************************************/
template<typename T, int BLOCK_DIM, int RAND_ITERS>
__global__ void findKNN_leaf_nodes_transpose(T* data, int dim, TPtree<T, RAND_ITERS>* tptree, int dataSize, int KVAL, int* results, long long int N, int distance_type) {

    float(*f)(T* a, T* b, int length);
    if (distance_type == 0) f = l2<T, BLOCK_DIM>;
    else f = cosine<T, BLOCK_DIM>;

    extern __shared__ char sharememory[];

    __shared__ ThreadHeap<T, BLOCK_DIM> heapMem[BLOCK_DIM];

    heapMem[threadIdx.x].initialize(KVAL - 1, (DistPair*)(sharememory + sizeof(DistPair) * (KVAL - 1) * threadIdx.x));

    T* transpose_mem = (T*)(sharememory + sizeof(DistPair) * (KVAL - 1) * BLOCK_DIM);

    DistPair max_K; // Stores largest of the K nearest neighbor of the query point
    bool dup; // Is the target already in the KNN list?

    DistPair target;
    long long int src_id; // Id of query vector

    int blocks_per_leaf = gridDim.x / tptree->num_leaves;
    int threads_per_leaf = blocks_per_leaf*blockDim.x;
    int thread_id_in_leaf = blockIdx.x % blocks_per_leaf * blockDim.x + threadIdx.x;
    int leafIdx = blockIdx.x / blocks_per_leaf;
    long long int leaf_offset = tptree->leafs[leafIdx].offset;

    // Each point in the leaf is handled by a separate thread
    for (int i = thread_id_in_leaf; i < tptree->leafs[leafIdx].size; i += threads_per_leaf) {

        heapMem[threadIdx.x].reset();

        loadTransposePoint<T, BLOCK_DIM>(&transpose_mem[threadIdx.x], data + tptree->leaf_points[leaf_offset + i] * (long long int)dim, dim);

        src_id = tptree->leaf_points[leaf_offset + i];

        // Load results from previous iterations into shared memory heap
        // and re-compute distances since they are not stored in result set
        heapMem[threadIdx.x].load_mem(data, dim, &results[src_id*KVAL], &transpose_mem[threadIdx.x], f);

        max_K.idx = results[(src_id + 1)*KVAL - 1];
        if (max_K.idx == -1) {
            max_K.dist = FLT_MAX;
        }
        else {
            max_K.dist = f(&transpose_mem[threadIdx.x], data + max_K.idx * (long long int)dim, dim);
        }

        // Compare source query with all points in the leaf
        for (long long int j = 0; j < tptree->leafs[leafIdx].size; ++j) {
            if (j != i) {
                target.dist = f(&transpose_mem[threadIdx.x], data + tptree->leaf_points[leaf_offset + j] * (long long int)dim, dim);
                if (target.dist < max_K.dist) {
                    target.idx = tptree->leaf_points[leaf_offset + j];
                    dup = false;
                    for (int dup_id = 0; dup_id < KVAL - 1; ++dup_id) {
                        if (heapMem[threadIdx.x].vals[dup_id].idx == target.idx) {
                            dup = true;
                            dup_id = KVAL;
                        }
                    }
                    if (!dup) { // Only consider it if not already in the KNN list
                        if (target.dist < heapMem[threadIdx.x].top()) {
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
        results[(src_id + 1)*KVAL - 1] = max_K.idx;
        for (int j = KVAL - 2; j >= 0; j--) {
            results[src_id*KVAL + j] = heapMem[threadIdx.x].vals[0].idx;
            heapMem[threadIdx.x].vals[0].dist = -1;
            heapMem[threadIdx.x].heapify();
        }
    }
}


/*****************************************************************************************
* Determines if @target is a nearer neighbor than the current contents of heap @heapMem.
* RET: true if @target is nearer than the current KNN, false if @target is more distant,
*      or if @target.idx is already in the heap (i.e., it is a duplicate)
*****************************************************************************************/
template<typename T, int BLOCK_DIM>
__forceinline__ __device__ bool is_NN(ThreadHeap<T, BLOCK_DIM>* heapMem, DistPair* max_K, DistPair target, int KVAL) {
    bool ret_val = false;;
    if (target.dist < max_K->dist) {
        ret_val = true;
        for (int dup_id = 0; dup_id < KVAL - 1; ++dup_id) {
            if (heapMem[threadIdx.x].vals[dup_id].idx == target.idx) {
                ret_val = false;
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
template<typename T, int BLOCK_DIM>
__device__ void check_neighbors(T* data, int dim, int KVAL, int* results, ThreadHeap<T, BLOCK_DIM>* heapMem, DistPair* max_K, long long int neighbor, long long int src_id, T* query, int dfs_level, float(*f)(T* a, T* b, int d)) {
    DistPair target;

    for (long long int j = 0; j < KVAL; ++j) { // Check each neighbor of this neighbor
        target.idx = results[neighbor*KVAL + j];
        if (target.idx != src_id) { // Don't include the source itself
            target.dist = f(query, data + target.idx * (long long int)dim, dim);

            if (is_NN<T, BLOCK_DIM>(heapMem, max_K, target, KVAL)) {
                if (target.dist < heapMem[threadIdx.x].top()) {
                    max_K->dist = heapMem[threadIdx.x].vals[0].dist;
                    max_K->idx = heapMem[threadIdx.x].vals[0].idx;
                    heapMem[threadIdx.x].insert(target.dist, target.idx);
                }
                else {
                    max_K->dist = target.dist;
                    max_K->idx = target.idx;
                }
                if (dfs_level < REFINE_DEPTH) {
                    check_neighbors<T, BLOCK_DIM>(data, dim, KVAL, results, heapMem, max_K, target.idx, src_id, query, dfs_level + 1, f);
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
template<typename T, int BLOCK_DIM>
__global__ void refine_KNN(T* data, int dim, int KVAL, int* results, long long int N, int distance_type) {

    float(*f)(T* a, T* b, int length);
    if (distance_type == 0) f = l2<T, BLOCK_DIM>;
    else f = cosine<T, BLOCK_DIM>;

    extern __shared__ char sharememory[];

    __shared__ ThreadHeap<T, BLOCK_DIM> heapMem[BLOCK_DIM];

    heapMem[threadIdx.x].initialize(KVAL - 1, (DistPair*)(sharememory + sizeof(DistPair) * (KVAL - 1) * threadIdx.x));

    T* transpose_mem = (T*)(sharememory + sizeof(DistPair) * (KVAL - 1) * BLOCK_DIM);

    int* neighbors = (int*)(sharememory + sizeof(DistPair) * (KVAL - 1) * BLOCK_DIM + sizeof(T)* dim * BLOCK_DIM + sizeof(int) * KVAL * threadIdx.x);

    DistPair max_K;

    for (long long int src_id = blockIdx.x*blockDim.x + threadIdx.x; src_id < N; src_id += blockDim.x*gridDim.x) {

        loadTransposePoint<T, BLOCK_DIM>(&transpose_mem[threadIdx.x], data + src_id * (long long int)dim, dim);

        // Load current result set into heap
        heapMem[threadIdx.x].reset();
        heapMem[threadIdx.x].load_mem(data, dim, &results[src_id*KVAL], &transpose_mem[threadIdx.x], f);
        max_K.idx = results[(src_id + 1)*KVAL - 1];
        max_K.dist = f(&transpose_mem[threadIdx.x], data + max_K.idx * (long long int)dim, dim);

        neighbors[0] = max_K.idx;
        // Load all neighbor ids
        for (int i = 1; i < KVAL; ++i) {
            neighbors[i] = heapMem[threadIdx.x].vals[i - 1].idx;
        }
#pragma unroll
        for (int i = 1; i < KVAL; ++i) {
            check_neighbors<T, BLOCK_DIM>(data, dim, KVAL, results, heapMem, &max_K, neighbors[i], src_id, &transpose_mem[threadIdx.x], 1, f);
        }

        results[(src_id + 1)*KVAL - 1] = max_K.idx;
        // Write KNN to result list in sorted order
        for (int j = KVAL - 2; j >= 0; j--) {
            results[src_id*KVAL + j] = heapMem[threadIdx.x].vals[0].idx;
            heapMem[threadIdx.x].vals[0].dist = -1;
            heapMem[threadIdx.x].heapify();
        }
    }
}


/************************************************************************************
* Check the Top-X accuracy from brute-force solution
************************************************************************************/
template<typename T>
__host__ void calc_topX_accuracy(int* results, long long int size, int KVAL, DistPair* solution, int X) {
    long long int total_correct = 0;
    for (long long int i = 0; i<size; i++) {
        for (long long int j = 0; j<X; j++) {
            for (int k = 0; k<KVAL; k++) {
                if (results[i*KVAL + k] == solution[i*KVAL + j].idx) {
                    total_correct++;
                }
            }
        }
    }
    printf("Top-%d correct: %lld\n", X, total_correct);
    printf("Top-%d accuracy (%%): %.2f\n", X, 100 * (double)total_correct / (double)(size*X));
}


template <typename T>
void buildGraph(T* mtx, int m_iFeatureDim, int m_iGraphSize, int m_iNeighborhoodSize, int trees, int m_numTopDimensionTPTSplit, int* results, int m_disttype)
{
    LOG("Alloc'ing Points on device: %lld bytes.\n", sizeof(T) * m_iGraphSize * m_iFeatureDim);
    T* data;
    cudaMallocManaged(&data, sizeof(T) * m_iGraphSize * m_iFeatureDim);
    cudaMemcpy(data, mtx, sizeof(T) * m_iGraphSize * m_iFeatureDim, cudaMemcpyHostToDevice);

    // Number of levels set to have approximately 500 points per leaf
    int levels = (int)std::log2(m_iGraphSize / 500);

    LOG("Alloc'ing memory for results on device: %lld bytes.\n", sizeof(int) * m_iGraphSize * m_iNeighborhoodSize);
    int* graph;
    cudaMallocManaged(&graph, sizeof(int) * m_iGraphSize * m_iNeighborhoodSize);

    for (int i = 0; i < m_iGraphSize; i++)
        for (int j = 0; j < m_iNeighborhoodSize; j++)
            graph[i*m_iNeighborhoodSize + j] = -1;

    cudaMemPrefetchAsync(mtx, sizeof(T) * m_iGraphSize * m_iFeatureDim, 0);
    cudaDeviceSynchronize();

    srand(time(NULL)); // random number seed for TP tree random hyperplane partitions
    //srand(1); // random number seed for TP tree random hyperplane partitions

    LOG("Alloc'ing memory for TPTree");
    TPtree<T, TPT_ITERS>* tptree;
    cudaMallocManaged(&tptree, sizeof(TPtree<T, TPT_ITERS>));
    tptree->initialize(m_iGraphSize, levels, m_numTopDimensionTPTSplit);

    double tree_time = 0.0, KNN_time = 0.0, refine_time = 0.0, use_time;
    time_t start_t, end_t;

    // Need at least 1 block per leaf
    int KNN_blocks = max(tptree->num_leaves, BLOCKS);

    for (int tree_id = 0; tree_id < trees; ++tree_id) { // number of TPTs used to create approx. KNN graph
        tptree->reset();
        cudaDeviceSynchronize();

        start_t = clock();

        // Create TPT
        create_tptree_device<T, TPT_ITERS>(tptree, data, m_iFeatureDim, m_numTopDimensionTPTSplit, m_iGraphSize, levels);
        cudaDeviceSynchronize();

        end_t = clock();

        use_time = (double)(end_t - start_t) / CLOCKS_PER_SEC;
        tree_time += use_time;
        LOG("TPT construction time (s): %.2f\n", use_time);

        LOG("levels:%d\n", tptree->levels);
        int maxLeaf = 0;
        int minLeaf = 99999999.9;
        for (int i = 0; i<tptree->num_leaves; i++) {
            if (maxLeaf < tptree->leafs[i].size)
                maxLeaf = tptree->leafs[i].size;
            if (minLeaf > tptree->leafs[i].size)
                minLeaf = tptree->leafs[i].size;
        }
        LOG("minLeaf:%d, maxLeaf:%d\n", minLeaf, maxLeaf);

        start_t = clock();

        // Compute the KNN for each leaf node
        findKNN_leaf_nodes_transpose<T, THREADS, TPT_ITERS> << <KNN_blocks, THREADS, sizeof(T) * m_iFeatureDim * THREADS + sizeof(DistPair) * (m_iNeighborhoodSize - 1) * THREADS >> >(data, m_iFeatureDim, tptree, m_iGraphSize, m_iNeighborhoodSize, graph, m_iGraphSize, m_disttype);
        cudaDeviceSynchronize();

        end_t = clock();

        use_time = (double)(end_t - start_t) / CLOCKS_PER_SEC;
        KNN_time += use_time;
        LOG("KNN Leaf time (s): %.2f\n", use_time);
    } // end TPT loop

#if REFINE_DEPTH > 0
    start_t = clock();

    // Perform a final refinement step of KNN graph
    refine_KNN<T, THREADS> << <KNN_blocks, THREADS, sizeof(T) * m_iFeatureDim * THREADS + sizeof(DistPair) * (m_iNeighborhoodSize - 1) * THREADS + sizeof(int) * m_iNeighborhoodSize * THREADS >> >(data, m_iFeatureDim, m_iNeighborhoodSize, graph, m_iGraphSize, m_disttype);
    cudaDeviceSynchronize();

    end_t = clock();
    refine_time += (double)(end_t - start_t) / CLOCKS_PER_SEC;
#endif

    tptree->destroy();
    cudaFree(tptree);
    cudaFree(data);

    LOG("Total TPT construction time: %.2f\n", tree_time);
    LOG("Total KNN time: %.2f\n", KNN_time);
    LOG("Refine time: %.2f\n", refine_time);
    LOG("Total GPU time (sec.): %.2f\n", tree_time + KNN_time + refine_time);

    cudaMemcpy(results, graph, sizeof(int) * m_iGraphSize * m_iNeighborhoodSize, cudaMemcpyDeviceToHost);
    cudaFree(graph);

#if defined(DEBUG)
    DistPair* bfresults;
    cudaMalloc(&bfresults, sizeof(DistPair) * m_iGraphSize * m_iNeighborhoodSize);
    findKNN_SMEM_transpose<T, THREADS> << <KNN_blocks, THREADS, sizeof(T) * m_iFeatureDim * THREADS + sizeof(DistPair) * (m_iNeighborhoodSize - 1) * THREADS >> > (data, m_iFeatureDim, m_iGraphSize, 0, m_iGraphSize, m_iNeighborhoodSize, bfresults, m_disttype);
    DistPair* tmp = (DistPair*)malloc(sizeof(DistPair) * m_iGraphSize * m_iNeighborhoodSize);
    cudaMemcpy(tmp, bfresults, sizeof(DistPair) * m_iGraphSize * m_iNeighborhoodSize, cudaMemcpyDeviceToHost);
    cudaFree(bfresults);
    calc_topX_accuracy<T>(results, m_iGraphSize, m_iNeighborhoodSize, tmp, 5);
    calc_topX_accuracy<T>(results, m_iGraphSize, m_iNeighborhoodSize, tmp, 32);
    free(tmp);
#endif
}

#endif
