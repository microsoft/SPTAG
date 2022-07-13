// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_COMMONUTILS_H_
#define _SPTAG_COMMON_COMMONUTILS_H_

#include "inc/Core/Common.h"

#include <unordered_map>

#include <exception>
#include <algorithm>

#include <time.h>
#include <omp.h>
#include <string.h>
#include <vector>
#include <set>

#define PREFETCH

namespace SPTAG
{
    namespace COMMON
    {
        class Utils {
        public:
            static SizeType rand(SizeType high = MaxSize, SizeType low = 0)   // Generates a random int value.
            {
                return low + (SizeType)(float(high - low)*(std::rand() / (RAND_MAX + 1.0)));
            }

            static inline float atomic_float_add(volatile float* ptr, const float operand)
            {
                union {
                    volatile long iOld;
                    float fOld;
                };
                union {
                    long iNew;
                    float fNew;
                };

                while (true) {
                    iOld = *(volatile long *)ptr;
                    fNew = fOld + operand;
                    if (InterlockedCompareExchange((long *)ptr, iNew, iOld) == iOld) {
                        return fNew;
                    }
                }
            }

            template<typename T>
            static inline int GetBase() {
                if (GetEnumValueType<T>() != VectorValueType::Float) {
                    return (int)(std::numeric_limits<T>::max)();
                }
                return 1;
            }

            template <typename T>
            static void Normalize(T* arr, DimensionType col, int base) {
                double vecLen = 0;
                for (DimensionType j = 0; j < col; j++) {
                    double val = arr[j];
                    vecLen += val * val;
                }
                vecLen = std::sqrt(vecLen);
                if (vecLen < 1e-6) {
                    T val = (T)(1.0 / std::sqrt((double)col) * base);
                    for (DimensionType j = 0; j < col; j++) arr[j] = val;
                }
                else {
                    for (DimensionType j = 0; j < col; j++) arr[j] = (T)(arr[j] / vecLen * base);
                }
            }

            template <typename T>
            static void BatchNormalize(T* data, SizeType row, DimensionType col, int base, int threads);

            static inline void AddNeighbor(SizeType idx, float dist, SizeType* neighbors, float* dists, DimensionType size)
            {
                size--;
                if (dist < dists[size] || (dist == dists[size] && idx < neighbors[size]))
                {
                    DimensionType nb;
                    for (nb = 0; nb <= size && neighbors[nb] != idx; nb++);

                    if (nb > size)
                    {
                        nb = size;
                        while (nb > 0 && (dist < dists[nb - 1] || (dist == dists[nb - 1] && idx < neighbors[nb - 1])))
                        {
                            dists[nb] = dists[nb - 1];
                            neighbors[nb] = neighbors[nb - 1];
                            nb--;
                        }
                        dists[nb] = dist;
                        neighbors[nb] = idx;
                    }
                }
            }
        };
    }
}

#endif // _SPTAG_COMMON_COMMONUTILS_H_
