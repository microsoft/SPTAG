#pragma once
#include <cstdint>

namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch
        {
            namespace DiskUtils
            {
                uint64_t GetSectorSize(const char* p_filePath);
            }
        }
    }
}