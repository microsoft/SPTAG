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
#endif
