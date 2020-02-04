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

#ifndef _SPTAG_COMMON_CUDA_DISTANCE_H_
#define _SPTAG_COMMON_CUDA_DISTANCE_H_

 /******************************************************************************************
 * Version of L2 distance metric that uses ILP to try and improve performance
 ******************************************************************************************/
template<typename T, int Stride>
__forceinline__ __device__ __host__ float l2_ILP(T* trans_p1, T* p2, int dim) {
    float totals[ILP];
#pragma unroll
    for (int i = 0; i < ILP; ++i)
        totals[i] = 0.0;

    for (int i = 0; i < dim - ILP + 1; i += ILP) {

#pragma unroll
        for (int l = 0; l < ILP; ++l) {
            totals[l] += (trans_p1[(i + l)*Stride] - p2[i + l])*(trans_p1[(i + l)*Stride] - p2[i + l]);
        }
    }

#pragma unroll
    for (int i = 1; i < ILP; ++i)
        totals[0] += totals[i];

    return totals[0];
}

template<typename T, int Stride>
__forceinline__ __device__ __host__ float l2(T* trans_p1, T* p2, int dim) {
    float total = 0;
#pragma unroll
    for (int i = 0; i < dim; ++i) {
        total += ((float)trans_p1[i*Stride] - (float)p2[i]) * ((float)trans_p1[i*Stride] - (float)p2[i]);
    }
    return total;
}

/******************************************************************************************
* Cosine distance metric comparison operation.  Requires that the float is floating point,
* regardless of the datatype T, because it requires squareroot.
******************************************************************************************/
template<typename T, int Stride>
__forceinline__ __device__ __host__ float cosine(T* trans_p1, T* p2, int dim) {
    float prod = 0;
    float a = 0;
    float b = 0;
#pragma unroll
    for (int i = 0; i < dim; ++i) {
        a += (float)trans_p1[i*Stride] * (float)trans_p1[i*Stride];
        b += (float)p2[i] * (float)p2[i];
        prod += (float)trans_p1[i*Stride] * (float)p2[i];
    }

    return 1 - (prod / (sqrt(a*b)));
}

/******************************************************************************************
* Simple euclidean distance calculation used for correctness check
******************************************************************************************/
template<typename T>
__host__ float l2_dist(T* a, T* b, int dim) {
    float total = 0.0;
    float dist;
    for (int i = 0; i < dim; i++) {
        dist = __half2float(a[i]) - __half2float(b[i]);
        total += dist*dist;
    }
    return sqrt((total));
}

#endif