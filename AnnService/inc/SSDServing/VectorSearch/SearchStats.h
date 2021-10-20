// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <chrono>

namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch {
            struct SearchStats
            {
                SearchStats()
                    : m_check(0),
                    m_exCheck(0),
                    m_totalListElementsCount(0),
                    m_filteredCount(0),
                    m_distCheckFilterCount(0),
                    m_diskAccessCount(0),
                    m_postingElementCount(0),
                    m_totalSearchLatency(0),
                    m_totalLatency(0),
                    m_exLatency(0),
                    m_asyncLatency0(0),
                    m_asyncLatency1(0),
                    m_asyncLatency2(0),
                    m_queueLatency(0),
                    m_sleepLatency(0)
                {
                }

                int m_check;

                int m_exCheck;

                int m_totalListElementsCount;

                int m_filteredCount;

                int m_distCheckFilterCount;

                int m_diskAccessCount;

                int m_postingElementCount;

                double m_totalSearchLatency;

                double m_totalLatency;

                double m_exLatency;

                double m_asyncLatency0;

                double m_asyncLatency1;

                double m_asyncLatency2;

                double m_queueLatency;

                double m_sleepLatency;

                std::chrono::steady_clock::time_point m_searchRequestTime;

                int m_threadID;
            };
        }
    }
}