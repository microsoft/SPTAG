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

#include <sstream>
//#include <boost/format.hpp>
#include <iostream>
#include <stdio.h>
#include <stdarg.h>
using namespace std;

#include "params.h"

enum log_level_t {
    LOG_NOTHING,
    LOG_CRITICAL,
    LOG_ERROR,
    LOG_WARNING,
    LOG_INFO,
    LOG_DEBUG
};


#define LOG_ALL(f_, ...) printf((f_), ##__VA_ARGS__)
#define DLOG_ALL(f_, ...) {if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0) printf((f_), ##__VA_ARGS__); }

#if LOG_LEVEL >= 1
#define LOG_CRIT(f_, ...) printf((f_), ##__VA_ARGS__)
#define DLOG_CRIT(f_, ...) {if(threadIdx.x==0 && blockIdx.x==0) printf((f_), ##__VA_ARGS__); }
#else
#define LOG_CRIT(f_, ...) {}
#define DLOG_CRIT(f_, ...) {}
#endif

#if LOG_LEVEL >= 2
#define LOG_ERR(f_, ...) printf((f_), ##__VA_ARGS__)
#define DLOG_ERR(f_, ...) {if(threadIdx.x==0 && blockIdx.x==0) printf((f_), ##__VA_ARGS__); }
#else
#define LOG_ERR(f_, ...) {}
#define DLOG_ERR(f_, ...) {}
#endif

#if LOG_LEVEL >= 3
#define LOG_WARN(f_, ...) printf((f_), ##__VA_ARGS__)
#define DLOG_WARN(f_, ...) {if(threadIdx.x==0 && blockIdx.x==0) printf((f_), ##__VA_ARGS__); }
#else
#define LOG_WARN(f_, ...) {}
#define DLOG_WARN(f_, ...) {}
#endif

#if LOG_LEVEL >= 4
#define LOG_INFO(f_, ...) printf((f_), ##__VA_ARGS__)
#define DLOG_INFO(f_, ...) {if(threadIdx.x==0 && blockIdx.x==0) printf((f_), ##__VA_ARGS__); }
#else
#define LOG_INFO(f_, ...) {}
#define DLOG_INFO(f_, ...) {}
#endif

#if LOG_LEVEL >= 5
#define LOG_DEBUG(f_, ...) printf((f_), ##__VA_ARGS__)
#define DLOG_DEBUG(f_, ...) {if(threadIdx.x==0 && blockIdx.x==0) printf((f_), ##__VA_ARGS__); }
#else
#define LOG_DEBUG(f_, ...) {}
#define DLOG_DEBUG(f_, ...) {}
#endif

#define STR_EXPAND(arg) #arg
#define STR(arg) STR_EXPAND(arg)
