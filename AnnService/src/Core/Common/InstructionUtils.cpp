#include "inc/Core/Common/InstructionUtils.h"
#include "inc/Core/Common.h"

#ifndef _MSC_VER
void cpuid(int info[4], int InfoType) {
    __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}
#endif

namespace SPTAG {
    namespace COMMON {
        const InstructionSet::InstructionSet_Internal InstructionSet::CPU_Rep;

        bool InstructionSet::SSE(void) { return CPU_Rep.HW_SSE; }
        bool InstructionSet::SSE2(void) { return CPU_Rep.HW_SSE2; }
        bool InstructionSet::AVX(void) { return CPU_Rep.HW_AVX; }
        bool InstructionSet::AVX2(void) { return CPU_Rep.HW_AVX2; }
        bool InstructionSet::AVX512(void) { return CPU_Rep.HW_AVX512; }
        
        void InstructionSet::PrintInstructionSet(void) 
        {
            if (CPU_Rep.HW_AVX512)
                LOG(Helper::LogLevel::LL_Info, "Using AVX512 InstructionSet!\n");
            else if (CPU_Rep.HW_AVX2)
                LOG(Helper::LogLevel::LL_Info, "Using AVX2 InstructionSet!\n");
            else if (CPU_Rep.HW_AVX)
                LOG(Helper::LogLevel::LL_Info, "Using AVX InstructionSet!\n");
            else if (CPU_Rep.HW_SSE2)
                LOG(Helper::LogLevel::LL_Info, "Using SSE2 InstructionSet!\n");
            else if (CPU_Rep.HW_SSE)
                LOG(Helper::LogLevel::LL_Info, "Using SSE InstructionSet!\n");
            else
                LOG(Helper::LogLevel::LL_Info, "Using NONE InstructionSet!\n");
        }

        // from https://stackoverflow.com/a/7495023/5053214
        InstructionSet::InstructionSet_Internal::InstructionSet_Internal() :
            HW_SSE{ false },
            HW_SSE2{ false },
            HW_AVX{ false },
            HW_AVX512{ false },
            HW_AVX2{ false }
        {
            int info[4];
            cpuid(info, 0);
            int nIds = info[0];

            //  Detect Features
            if (nIds >= 0x00000001) {
                cpuid(info, 0x00000001);
                HW_SSE = (info[3] & ((int)1 << 25)) != 0;
                HW_SSE2 = (info[3] & ((int)1 << 26)) != 0;
                HW_AVX = (info[2] & ((int)1 << 28)) != 0;
            }
            if (nIds >= 0x00000007) {
                cpuid(info, 0x00000007);
                HW_AVX2 = (info[1] & ((int)1 << 5)) != 0;
                HW_AVX512 = (info[1] & (((int)1 << 16) | ((int) 1 << 30)));

// If we are not compiling support for AVX-512 due to old compiler version, we should not call it
#ifdef _MSC_VER
#if _MSC_VER < 1920
                HW_AVX512 = false;
#endif
#endif
            }
            if (HW_AVX512)
                LOG(Helper::LogLevel::LL_Info, "Using AVX512 InstructionSet!\n");
            else if (HW_AVX2)
                LOG(Helper::LogLevel::LL_Info, "Using AVX2 InstructionSet!\n");
            else if (HW_AVX)
                LOG(Helper::LogLevel::LL_Info, "Using AVX InstructionSet!\n");
            else if (HW_SSE2)
                LOG(Helper::LogLevel::LL_Info, "Using SSE2 InstructionSet!\n");
            else if (HW_SSE)
                LOG(Helper::LogLevel::LL_Info, "Using SSE InstructionSet!\n");
            else
                LOG(Helper::LogLevel::LL_Info, "Using NONE InstructionSet!\n");
        }
    }
}
