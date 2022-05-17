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
#include <curand_kernel.h>
#include <type_traits>

#include "inc/Core/Common/TruthSet.h"
#include "inc/Core/Common/cuda/params.h"
#include "inc/Core/Common/cuda/TPtree.hxx"
#include "inc/Core/Common/cuda/KNN.hxx"

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


//#define BAL 2 // Balance factor - will only rebalance nodes that are at least 2x larger than their sibling

// Computes the fraction of points that need to be moved from each unbalanced node on the level
__global__ void check_for_imbalance(int* node_ids, int* node_sizes, int nodes_on_level, int node_start, float* frac_to_move, int bal_factor) {
  int neighborId;
  for(int i=node_start + blockIdx.x*blockDim.x + threadIdx.x; i<node_start+nodes_on_level; i+=blockDim.x*gridDim.x) {
    frac_to_move[i] = 0.0;
    neighborId = (i-1) + 2*(i&1); // neighbor is either left or right of current
    if(node_sizes[i] > bal_factor*node_sizes[neighborId]) {
      frac_to_move[i] = ((float)node_sizes[i] - (((float)(node_sizes[i]+node_sizes[neighborId]))/2.0)) / (float)node_sizes[i];
    }
  }
}

// Initialize random number generator for each thread
__global__ void initialize_rands(curandState* states, int iter) {
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  curand_init(1234, id, iter, &states[id]);
}

// Randomly move points to sibling nodes based on the fraction that need to be moved out of unbalanced nodes
__global__ void rebalance_nodes(int* node_ids, int N, float* frac_to_move, curandState* states) {
  int neighborId;
  int threadId = blockIdx.x*blockDim.x+threadIdx.x;

  for(int i=threadId; i<N; i+=blockDim.x*gridDim.x) {
    if((frac_to_move[node_ids[i]] > 0.0) && (curand_uniform(&states[threadId]) < frac_to_move[node_ids[i]])) {
      neighborId = (node_ids[i]-1) + 2*(node_ids[i]&1); // Compute idx of left or right neighbor
      node_ids[i] = neighborId;
    }
  }
}

template <typename T>
__global__ void GenerateTruthGPU(std::shared_ptr<VectorSet> querySet, std::shared_ptr<VectorSet> vectorSet, const std::string truthFile,
    const SPTAG::DistCalcMethod distMethod, const int K, const SPTAG::TruthFileType p_truthFileType, const std::shared_ptr<SPTAG::COMMON::IQuantizer>& quantizer) {
    int NUM_GPUS = 1;
    if (querySet->Dimension() != vectorSet->Dimension() && !quantizer)
    {
        LOG(Helper::LogLevel::LL_Error, "query and vector have different dimensions.");
        exit(1);
    }

    using SUMTYPE = std::conditional_t<std::is_same<T, float>::value, float, int32_t>;

    //Convert VectorSet to 2d matrix as GPU KNN source code
    //LOG_INFO("Converting vector sets...");
    //std::vector< std::vector<T> > vectorSet = (const T*)(vectorSet_obj->GetData());
    //std::vector< std::vector<T> > querySet = (const T*)(vectorSet_obj->GetData());
    //truthset & distset save the ground truth for idx and dist
    LOG(Helper::LogLevel::LL_Info, "Begin to generate truth for query(%d,%d) and doc(%d,%d)...\n", querySet->Count(), querySet->Dimension(), vectorSet->Count(), vectorSet->Dimension());
    std::vector< std::vector<SPTAG::SizeType> > truthset(querySet->Count(), std::vector<SPTAG::SizeType>(K, 0));
    std::vector< std::vector<float> > distset(vectorSet->Count(), std::vector<float>(K, 0));
    //Starts GPU KNN
    int result_size = querySet->Count();
    int vector_size = vectorSet->Count();
    int sub_vector_size = vector_size / NUM_GPUS;
    int dim = querySet->Dimension();

    LOG_INFO("QueryDatatype: %s, Rows:%ld, Columns:%d\n", (STR(T)), result_size, dim);
    LOG_INFO("Datatype: %s, Rows:%ld, Columns:%d\n", (STR(T)), vector_size, dim);

    LOG_INFO("Copying to Point array\n");
    Point<T, SUMTYPE, dim>* points = convertMatrix < T, SUMTYPE, dim > (querySet, result_size, dim);
    //split vectorSet into NUM_GPUS chunks
    LOG_INFO("Copying to Sub Point Array\n");
    Point<T, SUMTYPE, dim>* sub_vectors_points[NUM_GPUS];
    for (int i = 0; i < NUM_GPUS; i++) {
        std::vector<std::vector<T>> sub_vector;
        for (int idx = i * sub_vector_size; idx < (i + 1) * sub_vector_size; idx++) {
            sub_vector.emplace_back((const T*)(vectorSet->GetVector(i)));
        }
        LOG_INFO("GPU%d has %ld vectors\n", i, sub_vector.size());
        sub_vectors_points[i] = convertMatrix < T, SUMTYPE, dim > (sub_vector, result_size / NUM_GPUS, dim);
    }

    int per_gpu_size = (result_size)*K;//Every gpu do brute force to subset assigned to it

    DistPair<SUMTYPE>* d_results[NUM_GPUS]; // malloc space for each gpu to save bf result
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaMalloc(&d_results[i], per_gpu_size * sizeof(DistPair<SUMTYPE>)); // Device memory on each GPU
    }

    cudaStream_t streams[NUM_GPUS];
    /*
    LOG_INFO("Starting KNN Kernel timer\n");
    struct timespec start, end;
    time_t start_t, end_t;
    clock_gettime(CLOCK_MONOTONIC, &start);
    start_t = clock();
    */
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);

        cudaStreamCreate(&streams[i]); // Copy data over on a separate stream for each GPU
        //Copy Queryvectors
        LOG_INFO("Alloc'ing Points on device: %ld bytes.\n", querySet->Count() * sizeof(Point<T, SUMTYPE, dim>));
        Point<T, SUMTYPE, dim>* d_points;
        cudaMalloc(&d_points, querySet->Count() * sizeof(Point<T, SUMTYPE, dim>));
        LOG_INFO("Copying to device.\n");
        cudaMemcpyAsync(d_points, points, querySet->Count() * sizeof(Point<T, SUMTYPE, dim>), cudaMemcpyHostToDevice, streams[i]);

        //copy one chunk of vectorSet
        LOG_INFO("Alloc'ing Check_Points on device: %ld bytes.\n", vectorSet->Count() * sizeof(Point<T, SUMTYPE, dim>) / NUM_GPUS);
        Point<T, SUMTYPE, dim>* d_check_points;
        cudaMalloc(&d_check_points, vectorSet->Count() * sizeof(Point<T, SUMTYPE, dim>) / NUM_GPUS);
        LOG_INFO("Copying to device.\n");
        cudaMemcpyAsync(d_check_points, sub_vectors_points[i], vectorSet->Count() * sizeof(Point<T, SUMTYPE, dim>) / NUM_GPUS, cudaMemcpyHostToDevice, streams[i]);

        LOG_INFO("Alloc'ing memory for results on device: %ld bytes.\n", querySet->Count() * K * sizeof(int));

        int KNN_blocks = querySet->Count() / THREADS;

        // Perfrom brute-force KNN from the subsets assigned to the GPU for the querySets 
        query_KNN<T, dim, K, THREADS, SUMTYPE> << <KNN_blocks, THREADS >> > (d_points, d_check_points, vector_size / NUM_GPUS, i * (vector_size / NUM_GPUS), result_size, d_results[i]);
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    /*
    clock_gettime(CLOCK_MONOTONIC, &end);
    end_t = clock();

    LOG_ALL("Total GPU time (sec): %lf\n", (((double)(end_t - start_t)) / CLOCKS_PER_SEC));
    LOG_INFO("GPU runtime (ms): %lf\n", (1000 * end.tv_sec + 1e-6 * end.tv_nsec) - (1000 * start.tv_sec + 1e-6 * start.tv_nsec));
    */

    DistPair<SUMTYPE>* results = (DistPair<SUMTYPE>*)malloc(per_gpu_size * sizeof(DistPair<SUMTYPE>) * NUM_GPUS);

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaMemcpy(&results[(i * per_gpu_size)], d_results[i], per_gpu_size * sizeof(DistPair<SUMTYPE>), cudaMemcpyDeviceToHost);
    }


    // To combine the results, we need to compare value from different GPU 
    //Results [KNN from GPU0][KNN from GPU1][KNN from GPU2][KNN from GPU3]
    // Every block will be [result_size*K] [K|K|K|...|K]
    // The real KNN is selected from 4*KNN
    int* reader_ptrs = (int*)malloc(NUM_GPUS * sizeof(int));
    for (int i = 0; i < result_size; i++) {
        //For each result, there should be writer pointer, four reader pointers
        // reader ptr is assigned to i+gpu_idx*(result_size*K)/4
        for (int gpu_idx = 0; gpu_idx < NUM_GPUS; gpu_idx++) {
            reader_ptrs[gpu_idx] = i * K + gpu_idx * (result_size * K) / 4;
        }
        for (int writer_ptr = i * K; writer_ptr < (i + 1) * K; writer_ptr++) {
            int selected = 0;
            for (int gpu_idx = 1; gpu_idx < NUM_GPUS; gpu_idx++) {
                //if the other gpu has shorter dist
                if (results[reader_ptrs[gpu_idx]].dist < results[reader_ptrs[selected]].dist) {
                    selected = gpu_idx;
                }
            }
            truthset[i][writer_ptr - (i * K)] = results[reader_ptrs[selected]++].idx;
            distset[i][writer_ptr - (i * K)] = results[reader_ptrs[selected]++].dist;
        }
    }

    LOG(Helper::LogLevel::LL_Info, "Start to write truth file...\n");
    SPTAG::COMMON::TruthSet::writeTruthFile(truthFile, querySet->Count(), K, truthset, distset, p_truthFileType);

    auto ptr = SPTAG::f_createIO();
    if (ptr == nullptr || !ptr->Initialize((truthFile + ".dist.bin").c_str(), std::ios::out | std::ios::binary)) {
        LOG(Helper::LogLevel::LL_Error, "Fail to create the file:%s\n", (truthFile + ".dist.bin").c_str());
        exit(1);
    }

    int int32_queryNumber = (int)querySet->Count();
    ptr->WriteBinary(4, (char*)&int32_queryNumber);
    ptr->WriteBinary(4, (char*)&K);

    for (size_t i = 0; i < int32_queryNumber; i++)
    {
        for (int k = 0; k < K; k++) {
            if (ptr->WriteBinary(4, (char*)(&(truthset[i][k]))) != 4) {
                LOG(Helper::LogLevel::LL_Error, "Fail to write the truth dist file!\n");
                exit(1);
            }
            if (ptr->WriteBinary(4, (char*)(&(distset[i][k]))) != 4) {
                LOG(Helper::LogLevel::LL_Error, "Fail to write the truth dist file!\n");
                exit(1);
            }
        }
    }
}