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

#ifndef _SPTAG_COMMON_CUDA_PARAMS_H_
#define _SPTAG_COMMON_CUDA_PARAMS_H_

#include <stdio.h>
#include <stdarg.h>


/******************************************************
* Parameters that have been optimized experimentally 
******************************************************/
#define THREADS 32 // Number of threads per block
#define BLOCKS 10240 // Total blocks used
#define SAMPLES 5000 // number of samples used to determine median for TPT construction
#define KEYTYPE float // Keys used to divide TPTs at each node
#define ILP 1 // Increase of ILP using registers in distance calculations
#define TPT_ITERS 1 // Number of random sets of weights tried for each level
#define TPT_PART_DIMS D // Number of dimensions used to create hyperplane
#define REFINE_DEPTH 1 // Depth of refinement step.  No refinement if 0

#define REORDER 1 // Option to re-order queries for perf improvement (1 = reorder, 0 = no reorder)

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      printf(                                                                  \
          "GPU API call failure \"" #func "\" with CUDA error %d\n", (int)rt); \
      throw;                                                                   \
    }                                                                          \
  } while (0)




/******************************************************
* Parameters for output control
******************************************************/
#define LOG_LEVEL 5 // Log level: 0 (fewest messages) to 5 (all debug messages)

/************************************************************
* Datatype definitions based on selected DATATYPE value above
*                DO NOT EDIT BELOW
*************************************************************/
// for int8
//#if DATATYPE == 0
//#define DTYPE uint8_t 
//#if METRIC == 0 // L2 metric
//#define INFTY INT_MAX
//#define ACCTYPE uint16_t
//#define SUMTYPE int // Sumtype can be int for l2
//#else // Cosine
//#define INFTY FLT_MAX
//#define ACCTYPE float
//#define SUMTYPE float
//#endif
//
//#elif DATATYPE == 1
//#define INFTY FLT_MAX
//#define DTYPE float // Datatype of each coordinate
//#define ACCTYPE float
//#define SUMTYPE float
//
//#else
//#define INFTY FLT_MAX
//#define DTYPE float // Datatype of each coordinate
//#define ACCTYPE float
//#define SUMTYPE float
//#endif

#endif