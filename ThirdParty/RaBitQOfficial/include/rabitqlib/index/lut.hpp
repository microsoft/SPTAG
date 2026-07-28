#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib {

template <typename T>
class Lut {
    static constexpr size_t kNumBits = 8;
    static constexpr size_t kNumBitsHacc = 16;
    static_assert(std::is_floating_point_v<T>, "T must be an floating type in Lut");

   private:
    size_t table_length_ = 0;
    std::vector<uint8_t> lut_;
    T delta_;
    T sum_vl_lut_;

   public:
    explicit Lut() = default;
    explicit Lut(const T* rotated_query, size_t padded_dim, bool use_hacc = false)
        : table_length_(padded_dim << 2)
        , lut_(table_length_ * (static_cast<int>(use_hacc) + 1)) {
        // quantize float lut
        std::vector<float> lut_float(table_length_);
        fastscan::pack_lut(padded_dim, rotated_query, lut_float.data());
        T vl_lut;
        T vr_lut;
        data_range(lut_float.data(), table_length_, vl_lut, vr_lut);

        if (use_hacc) {
            delta_ = (vr_lut - vl_lut) / ((1 << kNumBitsHacc) - 1);

            // quantize float lut into uint16 then change to split table
            std::vector<uint16_t> lut_u16(table_length_);
            scalar_quantize(
                lut_u16.data(), lut_float.data(), table_length_, vl_lut, delta_
            );
            fastscan::transfer_lut_hacc(lut_u16.data(), padded_dim, lut_.data());
        } else {
            delta_ = (vr_lut - vl_lut) / ((1 << kNumBits) - 1);
            scalar_quantize(lut_.data(), lut_float.data(), table_length_, vl_lut, delta_);
        }

        size_t num_table = table_length_ / 16;
        sum_vl_lut_ = vl_lut * static_cast<float>(num_table);
    }
    Lut& operator=(Lut&& other) noexcept {
        lut_ = std::move(other.lut_);
        delta_ = other.delta_;
        sum_vl_lut_ = other.sum_vl_lut_;
        return *this;
    }

    [[nodiscard]] const uint8_t* lut() const { return lut_.data(); };
    [[nodiscard]] T delta() const { return delta_; };
    [[nodiscard]] T sum_vl() const { return sum_vl_lut_; };
};
}  // namespace rabitqlib