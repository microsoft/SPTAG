#pragma once
#include "inc/SSDServing/Common/stdafx.h"

namespace SPTAG {
    namespace SSDServing {
        namespace SelectHead_BKT {
            struct BKTNodeInfo
            {
                BKTNodeInfo() : leafSize(0), minDepth(INT32_MAX), maxDepth(0), parent(-1)
                {
                }

                int leafSize;

                int minDepth;

                int maxDepth;

                int parent;
            };
        }
    }
}