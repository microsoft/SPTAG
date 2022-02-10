/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#ifndef _SPTAG_COMMON_CUDA_KMEANS_H_
#define _SPTAG_COMMON_CUDA_KMEANS_H_

#include "params.h"
#include "Distance.hxx"
#include "../../Common.h"

#include <chrono>

__device__ __inline__ void distCASMax(float* oldDist, SizeType* oldIdx, float newDist, SizeType newIdx, int* lock) {
    int old;

    // Spinlock
    do {
        old = atomicCAS(lock, 0, 1);
    } while (old != 0);

    if(newDist > *oldDist) {
        *oldDist = newDist;
        *oldIdx = newIdx;
    }
    
    atomicExch(lock, 1); // Unlock
}

__device__ __inline__ void distCASMin(float* oldDist, SizeType* oldIdx, float newDist, SizeType newIdx, int* lock) {
    int old;

    // Spinlock
    do {
        old = atomicCAS(lock, 0, 1);
    } while (old != 0);

    if(newDist <= *oldDist) {
        *oldDist = newDist;
        *oldIdx = newIdx;
    }
    
    atomicExch(lock, 1); // Unlock
}

#define COPY_BUFF_SIZE 100000

// Convert Dataset vector to an array of Point structures on the GPU and reorders them based on @indices.
// Works in small batches to reduce CPU memory overhead
template<typename T, typename SUMTYPE, int MAX_DIM>
void ConvertDatasetToPoints(const Dataset<T>& data, std::vector<SizeType>& indices, Point<T,SUMTYPE,MAX_DIM>* d_points, size_t workSize, int dim) {

  Point<T,SUMTYPE,MAX_DIM>* pointBuffer = new Point<T,SUMTYPE,MAX_DIM>[COPY_BUFF_SIZE];

  size_t rows = workSize;
  size_t copy_size = COPY_BUFF_SIZE;
  for(size_t i=0; i<rows; i+=COPY_BUFF_SIZE) {
    if(rows-i < COPY_BUFF_SIZE) copy_size = rows-i; // Last copy may be smaller

    for(int j=0; j<copy_size; j++) {
      pointBuffer[j].loadChunk((T*)(data[indices[i+j]]), dim);
      pointBuffer[j].id = indices[i+j];
    }

    CUDA_CHECK(cudaMemcpy(d_points+i, pointBuffer, copy_size*sizeof(Point<T,SUMTYPE,MAX_DIM>), cudaMemcpyHostToDevice));
  }
}

template <typename T, typename SUMTYPE, int MAX_DIM>
__global__ void KmeansKernel(Point<T,SUMTYPE,MAX_DIM>* points, T* centers, size_t workSize, int* label, SizeType* counts, float lambda, float* clusterDist, SizeType* clusterIdx, SizeType* newCounts, float* weightedCounts, float* newCenters, int* clusterLocks, float* currDist, int _D, int _DK, int metric, const bool updateCenters)
{

    Point<T,float,MAX_DIM> centroid;
    Point<T,float,MAX_DIM> target;

    for(size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < workSize; i += blockDim.x*gridDim.x) {
        // Load target int Point structure
        target = points[i];
        int clusterid = 0;
        float smallestDist = MaxDist;

        for(int k = 0; k < _DK; k++) {
            // Load centroid coordinates into Point structure
            centroid.loadChunk(centers, _D);

            float dist = target.dist(&centroid, metric) + lambda*counts[k];

            if(dist > -MaxDist && dist < smallestDist) {
                clusterid = k;
                smallestDist = dist;
            }
        }
        label[i] = clusterid;
        atomicAdd(&(newCounts[clusterid]), 1);
        atomicAdd(&(weightedCounts[clusterid]), smallestDist);

        atomicAdd(currDist, smallestDist);

        if(updateCenters) {
            const T* v = (const T*)(&(points[i].coords[0]));
            for(DimensionType j=0; j<_D; ++j) {
                atomicAdd(&(newCenters[clusterid+j]), v[j]);
            }
           
            if(smallestDist > clusterDist[clusterid]) {
                distCASMax(&clusterDist[clusterid], &clusterIdx[clusterid], smallestDist, points[i].id, &clusterLocks[clusterid]);
            }
        }
        else {
            if(smallestDist <= clusterDist[clusterid]) {
                distCASMin(&clusterDist[clusterid], &clusterIdx[clusterid], smallestDist, points[i].id, &clusterLocks[clusterid]);
            }
        }
    }
}

template <typename T, typename SUMTYPE, int MAX_DIM>
float computeKmeansGPU(const Dataset<T>& data,
                  std::vector<SizeType>& indices,
                  const SizeType first, const SizeType last,
                  int _K, DimensionType _D, int _DK, float lambda, T* centers, int* label, 
                  SizeType* counts, SizeType* newCounts, float* newCenters, SizeType* clusterIdx, 
                  float* clusterDist, float* weightedCounts, float* newWeightedCounts,
                  int distMetric, const bool updateCenters) {

    size_t workSize = last - first;

    Point<T,float,MAX_DIM>* d_points;
    CUDA_CHECK(cudaMalloc(&d_points, workSize*sizeof(Point<T,float,MAX_DIM>)));

    ConvertDatasetToPoints<T,float,MAX_DIM>(data, indices, d_points, workSize, _D);
  
    T* d_centers;
    CUDA_CHECK(cudaMalloc(&d_centers, _K*_D*sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_centers, centers, _K*_D*sizeof(T), cudaMemcpyHostToDevice));

    int* d_clusterLocks;
    CUDA_CHECK(cudaMalloc(&d_clusterLocks, _K*sizeof(int)));
    CUDA_CHECK(cudaMemset(&d_clusterLocks, 0, _K*sizeof(int)));

    // Does this need to be the total dataset size?
    int* d_label;
    CUDA_CHECK(cudaMalloc(&d_label, indices.size()*sizeof(int)));

    SizeType* d_counts;
    CUDA_CHECK(cudaMalloc(&d_counts, _K*sizeof(SizeType)));
    CUDA_CHECK(cudaMemcpy(d_counts, counts, _K*sizeof(SizeType), cudaMemcpyHostToDevice));

    SizeType* d_newCounts;
    CUDA_CHECK(cudaMalloc(&d_newCounts, _K*sizeof(SizeType)));
    CUDA_CHECK(cudaMemset(&d_newCounts, 0, _K*sizeof(SizeType)));

    float* d_weightedCounts;
    CUDA_CHECK(cudaMalloc(&d_weightedCounts, _K*sizeof(float)));
    CUDA_CHECK(cudaMemset(&d_weightedCounts, 0, _K*sizeof(float)));

    float* d_newCenters;
    CUDA_CHECK(cudaMalloc(&d_newCenters, _D*_K*sizeof(float)));
    

    float* d_clusterDist;
    CUDA_CHECK(cudaMalloc(&d_clusterDist, _K*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_clusterDist, clusterDist, _K*sizeof(float), cudaMemcpyHostToDevice));

    SizeType* d_clusterIdx;
    CUDA_CHECK(cudaMalloc(&d_clusterIdx, _K*sizeof(SizeType)));
    CUDA_CHECK(cudaMemcpy(d_clusterIdx, clusterIdx, _K*sizeof(SizeType), cudaMemcpyHostToDevice));

    float* d_currDist;
    CUDA_CHECK(cudaMalloc(&d_currDist, sizeof(float))); // just 1 aggregate float value 

    KmeansKernel<T,float,MAX_DIM><<<1024, 128>>>(d_points, d_centers, workSize, d_label, d_counts, lambda, d_clusterDist, d_clusterIdx, d_newCounts, d_weightedCounts, d_newCenters, d_clusterLocks, d_currDist, _D, _DK, distMetric, updateCenters);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back results...
    CUDA_CHECK(cudaMemcpy(newCounts, d_newCounts, _K*sizeof(SizeType), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(clusterIdx, d_clusterIdx, _K*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(newCounts, d_newCounts, _K*sizeof(SizeType), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weightedCounts, d_weightedCounts, _K*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(newCenters, d_newCenters, _D*_K*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(clusterDist, d_clusterDist, _K*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(clusterIdx, d_clusterIdx, _K*sizeof(SizeType), cudaMemcpyDeviceToHost));

    float currDist;
    CUDA_CHECK(cudaMemcpy(d_currDist, &currDist, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_clusterLocks));
    CUDA_CHECK(cudaFree(d_label));
    CUDA_CHECK(cudaFree(d_newCounts));
    CUDA_CHECK(cudaFree(d_weightedCounts));
    CUDA_CHECK(cudaFree(d_newCenters));
    CUDA_CHECK(cudaFree(d_currDist));
  
    return currDist;
}

#endif
