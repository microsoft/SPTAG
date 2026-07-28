#include "pack_excode_kernels.hpp"

#include "rabitqlib/simd/pack_excode_dispatch.hpp"

namespace rabitqlib::simd {

void packing_2bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    detail::packing_2bit_excode_intrinsics(o_raw, o_compact, dim);
}

void packing_3bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    detail::packing_3bit_excode_intrinsics(o_raw, o_compact, dim);
}

void packing_4bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    detail::packing_4bit_excode_intrinsics(o_raw, o_compact, dim);
}

void packing_5bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    detail::packing_5bit_excode_intrinsics(o_raw, o_compact, dim);
}

void packing_6bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    detail::packing_6bit_excode_intrinsics(o_raw, o_compact, dim);
}

void packing_7bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    detail::packing_7bit_excode_intrinsics(o_raw, o_compact, dim);
}

}  // namespace rabitqlib::simd
