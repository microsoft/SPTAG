// This code is modified based on NGT from Yahoo Japan
// https://github.com/yahoojapan/NGT
//
// Copyright (C) 2015 Yahoo Japan Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

#include <climits>
#include <cstring>
#include <iostream>
#include <unordered_set>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/memory.hpp"

namespace rabitqlib {
/**
 * @brief hash set to record visited vertices
 *
 */
class HashBasedBooleanSet {
   private:
    size_t table_size_ = 0;
    PID mask_ = 0;
    std::vector<PID, memory::AlignedAllocator<PID>> table_;
    std::unordered_set<PID> stl_hash_;

    [[nodiscard]] auto hash1(const PID value) const { return value & mask_; }

   public:
    HashBasedBooleanSet() = default;
    ~HashBasedBooleanSet() = default;

    HashBasedBooleanSet(const HashBasedBooleanSet&) = default;
    HashBasedBooleanSet(HashBasedBooleanSet&&) noexcept = default;
    HashBasedBooleanSet& operator=(HashBasedBooleanSet&&) noexcept = default;

    explicit HashBasedBooleanSet(size_t size) {
        size_t bit_size = 0;
        size_t bit = size;
        while (bit != 0) {
            bit_size++;
            bit >>= 1;
        }
        size_t bucket_size = 0x1 << ((bit_size + 4) / 2 + 3);
        initialize(bucket_size);
    }

    void initialize(const size_t table_size) {
        table_size_ = table_size;
        mask_ = static_cast<PID>(table_size_ - 1);
        const PID check_val = hash1(static_cast<PID>(table_size));
        if (check_val != 0) {
            std::cerr << "[WARN] table size is not 2^N :  " << table_size << '\n';
        }

        table_ = std::vector<PID, memory::AlignedAllocator<PID>>(table_size);
        std::fill(table_.begin(), table_.end(), kPidMax);
        stl_hash_.clear();
    }

    void clear() {
        std::fill(table_.begin(), table_.end(), kPidMax);
        stl_hash_.clear();
    }

    // get if data_id is in the hashset
    [[nodiscard]] bool get(PID data_id) const {
        PID val = this->table_[hash1(data_id)];
        if (val == data_id) {
            return true;
        }
        return (val != kPidMax && stl_hash_.find(data_id) != stl_hash_.end());
    }

    void set(PID data_id) {
        PID& val = table_[hash1(data_id)];
        if (val == data_id) {
            return;
        }
        if (val == kPidMax) {
            val = data_id;
        } else {
            stl_hash_.emplace(data_id);
        }
    }
};
}  // namespace rabitqlib