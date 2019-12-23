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

#ifndef _SPTAG_COMMON_CUDA_PARAMS_H_
#define _SPTAG_COMMON_CUDA_PARAMS_H_

#include <stdio.h>
#include <stdarg.h>

/******************************************************
* Parameters that have been optimized experimentally 
******************************************************/
#define ILP 1 // Increase of ILP using registers in distance calculations
#define TPT_ITERS 1 // Number of random sets of weights tried for each level
#define THREADS 64 // Number of threads per block
#define BLOCKS 10240 // Total blocks used
#define REFINE_DEPTH 1 // Depth of refinement step.  No refinement if 0

#if defined(DEBUG)
#define LOG(f_, ...) printf((f_), ##__VA_ARGS__)
#else
#define LOG(f_, ...) {}
#endif

#endif


