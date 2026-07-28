#include "rabitqlib/utils/cpu_features.hpp"

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if (defined(__x86_64__) || defined(__i386__)) && !defined(_MSC_VER)
#include <cpuid.h>
#endif

namespace rabitqlib::cpu {
namespace {

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
void cpuid(uint32_t leaf, uint32_t subleaf, uint32_t* a, uint32_t* b, uint32_t* c, uint32_t* d) {
    int info[4];
    __cpuidex(info, static_cast<int>(leaf), static_cast<int>(subleaf));
    *a = static_cast<uint32_t>(info[0]);
    *b = static_cast<uint32_t>(info[1]);
    *c = static_cast<uint32_t>(info[2]);
    *d = static_cast<uint32_t>(info[3]);
}
#elif defined(__x86_64__) || defined(__i386__)
void cpuid(uint32_t leaf, uint32_t subleaf, uint32_t* a, uint32_t* b, uint32_t* c, uint32_t* d) {
    __cpuid_count(leaf, subleaf, *a, *b, *c, *d);
}
#else
void cpuid(uint32_t, uint32_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*) {}
#endif

Features detect_features() {
    Features detected{};

#if defined(__x86_64__) || defined(__i386__) || \
    (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
    // leaf 1: ECX[28]=AVX, ECX[12]=FMA, ECX[23]=POPCNT
    uint32_t eax, ebx, ecx, edx;
    cpuid(1, 0, &eax, &ebx, &ecx, &edx);

    // leaf 7 (subleaf 0): EBX[5]=AVX2, EBX[16]=AVX512F,
    //                     EBX[17]=AVX512DQ, EBX[30]=AVX512BW,
    //                     ECX[14]=AVX512_VPOPCNTDQ
    uint32_t l7_eax, l7_ebx, l7_ecx, l7_edx;
    cpuid(7, 0, &l7_eax, &l7_ebx, &l7_ecx, &l7_edx);

    detected.fma = (ecx >> 12) & 1;
    detected.avx2 = (l7_ebx >> 5) & 1;
    detected.avx512f = (l7_ebx >> 16) & 1;
    detected.avx512dq = (l7_ebx >> 17) & 1;
    detected.avx512bw = (l7_ebx >> 30) & 1;
    detected.avx512vpopcntdq = (l7_ecx >> 14) & 1;
#endif

    return detected;
}

}  // namespace

const Features& features() {
    static const Features detected = detect_features();
    return detected;
}

bool has_avx2() {
    const Features& detected = features();
    return detected.avx2 && detected.fma;
}

bool has_avx512_core() {
    const Features& detected = features();
    return detected.avx512f && detected.avx512bw && detected.avx512dq;
}

bool has_avx512_popcnt() {
    return has_avx512_core() && features().avx512vpopcntdq;
}

}  // namespace rabitqlib::cpu
