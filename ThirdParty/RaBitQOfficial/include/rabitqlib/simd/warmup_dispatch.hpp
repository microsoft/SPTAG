#pragma once

#include <cstddef>
#include <cstdint>

namespace rabitqlib::simd {

float warmup_ip_x0_q_512_avx2(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
);

float warmup_ip_x0_q_512_avx512(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
);

}  // namespace rabitqlib::simd
