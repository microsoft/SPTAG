// The implementation is largely based on the implementation of SVS.
// https://github.com/intel/ScalableVectorSearch

/*
 * Copyright 2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <fstream>
#include <utility>

#include "rabitqlib/utils/memory.hpp"

namespace rabitqlib {
namespace array_impl {
/**
 * @brief get size of array
 */
template <typename Dims>
[[nodiscard]] constexpr auto size(const Dims& dims) -> size_t {
    static_assert(std::is_same_v<typename Dims::value_type, size_t>);

    size_t res = 1;
    std::for_each(dims.begin(), dims.end(), [&](auto cur_d) { res *= cur_d; });
    return res;
}
}  // namespace array_impl

template <
    typename T,
    typename Dims = std::vector<size_t>,
    typename Alloc = memory::Allocator<T>>
class Array {
   private:
    static_assert(std::is_trivial_v<T>);  // only handle trivial types

    /// @brief num of data objects
    [[nodiscard]] constexpr auto size() const -> size_t { return array_impl::size(dims_); }

    /// @brief num of bytes for all data objects
    [[nodiscard]] constexpr auto bytes() const -> size_t { return sizeof(T) * size(); }

    void destroy() {
        size_t num_elements = size();
        atraits::deallocate(allocator_, pointer_, num_elements);
        pointer_ = nullptr;
    }

   public:
    using allocator_type = Alloc;
    using atraits = std::allocator_traits<allocator_type>;
    using pointer = typename atraits::pointer;
    using const_pointer = typename atraits::const_pointer;

    using value_type = T;
    using reference = T&;
    using const_reference = const T&;

    Array() = default;

    explicit Array(Dims dims, const Alloc& allocator)
        : dims_(std::move(dims)), allocator_(allocator) {
        size_t num_elements = size();
        pointer_ = atraits::allocate(allocator_, num_elements);
    }

    explicit Array(Dims dims) : Array(std::move(dims), Alloc()) {}

    ~Array() noexcept {
        if (pointer_ != nullptr) {
            destroy();
        }
    }

    /// @brief move constructor
    Array(Array&& other) noexcept
        : pointer_{std::exchange(other.pointer_, nullptr)}
        , dims_{std::move(other.dims_)}
        , allocator_{std::move(other.allocator_)} {}

    Array& operator=(Array&& other) noexcept {
        if (pointer_ != nullptr) {
            destroy();
        }

        if constexpr (atraits::propagate_on_container_move_assignment::value) {
            allocator_ = std::move(other.allocator_);
        }
        dims_ = std::exchange(other.dims_, Dims());
        pointer_ = std::exchange(other.pointer_, nullptr);
        return *this;
    }

    [[nodiscard]] pointer data() { return pointer_; }
    [[nodiscard]] const_pointer data() const { return pointer_; }

    [[nodiscard]] reference at(size_t idx) { return pointer_[idx]; }
    [[nodiscard]] const_reference at(size_t idx) const { return pointer_[idx]; }

    void save(std::ofstream& output) const {
        if (output.good()) {
            output.write(reinterpret_cast<char*>(pointer_), bytes());
        }
    }
    void load(std::ifstream& input) {
        input.read(reinterpret_cast<char*>(pointer_), bytes());
    }

    reference operator[](size_t idx) { return pointer_[idx]; }

   private:
    pointer pointer_ = nullptr;
    [[no_unique_address]] Dims dims_;
    [[no_unique_address]] Alloc allocator_;
};
}  // namespace rabitqlib