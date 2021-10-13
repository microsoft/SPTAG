// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <chrono>

namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch {
            namespace TimeUtils {
                typedef std::chrono::steady_clock SteadClock;

                /// Clock class
                class StopW {
                private:
                    std::chrono::steady_clock::time_point time_begin;
                public:
                    StopW();

                    double getElapsedMs();

                    double getElapsedSec();
                    
                    double getElapsedMin();

                    void reset();
                };

                double getMsInterval(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end);

                double getSecInterval(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end);

                double getMinInterval(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end);
            }
        }
    }
}