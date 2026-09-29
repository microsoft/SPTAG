// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_COMMONUTILS_H_
#define _SPTAG_COMMON_COMMONUTILS_H_

#include "inc/Core/Common.h"

#include <unordered_map>

#include <exception>
#include <algorithm>

#include <thread>
#include <time.h>
#include <string.h>
#include <vector>
#include <set>

#define PREFETCH

#ifndef _MSC_VER
#define EXCHANGE_TYPE_32 int
#else
#define EXCHANGE_TYPE_32 long
#endif

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
                    volatile EXCHANGE_TYPE_32 iOld;
                    float fOld;
                };
                union {
                    EXCHANGE_TYPE_32 iNew;
                    float fNew;
                };

                while (true) {
                    iOld = *(volatile EXCHANGE_TYPE_32 *)ptr;
                    fNew = fOld + operand;
                    if (InterlockedCompareExchange((EXCHANGE_TYPE_32 *)ptr, iNew, iOld) == iOld)
                    {
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

            static void PrintPostingDiff(std::string &p1, std::string &p2, const char *pos)
            {
                if (p1.size() != p2.size())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Merge %s: p1 and p2 have different sizes: before=%u after=%u\n", pos, p1.size(),
                                 p2.size());
                    return;
                }
                std::string diff = "";
                for (size_t i = 0; i < p1.size(); i++)
                {
                    if (p1[i] != p2[i])
                    {
                        diff += "[" + std::to_string(i) + "]:" + std::to_string(int(p1[i])) + "^" +
                                std::to_string(int(p2[i])) + " ";
                    }
                }
                if (diff.size() != 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "%s: %s\n", pos, diff.substr(0, 1000).c_str());
                }
            }
        };
    }
}

#endif // _SPTAG_COMMON_COMMONUTILS_H_
