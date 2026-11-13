#pragma once

#if defined(_MSC_VER)
#include <intrin.h>

#ifndef __restrict__
#define __restrict__ __restrict
#endif

#ifndef __builtin_popcount
#define __builtin_popcount(value) __popcnt(static_cast<unsigned int>(value))
#endif

#ifndef __builtin_popcountll
#define __builtin_popcountll(value) __popcnt64(static_cast<unsigned __int64>(value))
#endif

inline void* aligned_alloc(size_t size, size_t alignment)
{
    return _aligned_malloc(size, alignment);
}

inline void aligned_free(void *ptr)
{
    _aligned_free(ptr);
}
#else
using namespace std;
#endif
