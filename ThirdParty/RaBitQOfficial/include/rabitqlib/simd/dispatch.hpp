#pragma once

#include <array>
#include <cstddef>

#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::simd {

using ExcodeIpTable = std::array<ex_ipfunc, 9>;

ExcodeIpTable resolve_excode_ip_table();

}  // namespace rabitqlib::simd
