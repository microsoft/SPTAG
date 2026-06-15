#pragma once

#include <immintrin.h>
#include <sys/mman.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>

#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::memory {
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

template <typename T, size_t Alignment = 64, bool HugePage = false>
class AlignedAllocator {
   private:
    static_assert(Alignment >= alignof(T));

   public:
    using value_type = T;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    constexpr AlignedAllocator() noexcept = default;

    constexpr AlignedAllocator(const AlignedAllocator&) noexcept = default;

    template <typename U>
    constexpr explicit AlignedAllocator(AlignedAllocator<U, Alignment> const&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }

        auto nbytes = round_up_to_multiple_of<size_t>(n * sizeof(T), Alignment);
        auto* ptr = std::aligned_alloc(Alignment, nbytes);
        if (HugePage) {
            madvise(ptr, nbytes, MADV_HUGEPAGE);
        }
        return reinterpret_cast<T*>(ptr);
    }

    void deallocate(T* ptr, [[maybe_unused]] std::size_t n) { std::free(ptr); }
};

template <typename T>
struct Allocator {
   public:
    using value_type = T;

    constexpr Allocator() noexcept = default;

    template <typename U>
    explicit constexpr Allocator(const Allocator<U>&) noexcept {}

    [[nodiscard]] constexpr T* allocate(std::size_t n) { return ::new T[n]; }

    constexpr void deallocate(T* ptr, [[maybe_unused]] size_t n) noexcept {
        ::delete[] ptr;
    }

    // Intercept zero-argument construction to do default initialization.
    template <typename U>
    void construct(U* ptr) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(ptr)) U;
    }
};

template <size_t Alignment, typename T, bool HugePage = false>
inline T* align_allocate(size_t nbytes) {
    auto size = round_up_to_multiple_of<size_t>(nbytes, Alignment);
    void* ptr = std::aligned_alloc(Alignment, size);
    if (HugePage) {
        madvise(ptr, size, MADV_HUGEPAGE);
    }
    return static_cast<T*>(ptr);
}

static inline void prefetch_l1(const void* addr) {
#if defined(__SSE2__)
    _mm_prefetch(addr, _MM_HINT_T0);
#else
    __builtin_prefetch(addr, 0, 3);
#endif
}

static inline void prefetch_l2(const void* addr) {
#if defined(__SSE2__)
    _mm_prefetch((const char*)addr, _MM_HINT_T1);
#else
    __builtin_prefetch(addr, 0, 2);
#endif
}

inline void mem_prefetch_l1(const char* ptr, size_t num_lines) {
    switch (num_lines) {
        default:
            [[fallthrough]];
        case 20:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 19:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 18:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 17:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 16:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 15:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 14:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 13:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 12:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 11:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 10:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 9:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 8:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 7:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 6:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 5:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 4:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 3:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 2:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 1:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 0:
            break;
    }
}

inline void mem_prefetch_l2(const char* ptr, size_t num_lines) {
    switch (num_lines) {
        default:
            [[fallthrough]];
        case 20:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 19:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 18:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 17:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 16:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 15:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 14:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 13:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 12:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 11:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 10:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 9:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 8:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 7:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 6:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 5:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 4:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 3:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 2:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 1:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 0:
            break;
    }
}
}  // namespace rabitqlib::memory
