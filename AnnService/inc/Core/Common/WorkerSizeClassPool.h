// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_WORKERSIZECLASSPOOL_H_
#define _SPTAG_COMMON_WORKERSIZECLASSPOOL_H_

#include <array>
#include <cstddef>
#include <memory>

namespace SPTAG
{
namespace COMMON
{
template <typename T> class WorkerSizeClassPool
{
    static constexpr std::size_t ClassCount = 12;

    static std::size_t ClassFor(std::size_t bytes)
    {
        std::size_t capacity = 4096;
        for (std::size_t i = 0; i < ClassCount; ++i, capacity *= 2)
            if (bytes <= capacity)
                return i;
        return ClassCount;
    }

    static auto &Slots()
    {
        thread_local std::array<std::unique_ptr<T>, ClassCount> slots;
        return slots;
    }

  public:
    static std::unique_ptr<T> Rent(std::size_t bytes)
    {
        const auto slot = ClassFor(bytes);
        return slot < ClassCount ? std::move(Slots()[slot]) : nullptr;
    }

    static void Return(std::unique_ptr<T> value, std::size_t bytes)
    {
        const auto slot = ClassFor(bytes);
        if (value && slot < ClassCount && !Slots()[slot])
            Slots()[slot] = std::move(value);
    }
};
} // namespace COMMON
} // namespace SPTAG

#endif // _SPTAG_COMMON_WORKERSIZECLASSPOOL_H_
