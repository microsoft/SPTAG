// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <chrono>

namespace SPTAG {
    namespace SSDServing {
        namespace Utils {
            typedef std::chrono::steady_clock SteadClock;

            double getMsInterval(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
                return (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1.0) / 1000.0;
            }

            double getSecInterval(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
                return (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 1.0) / 1000.0;
            }

            double getMinInterval(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
                return (std::chrono::duration_cast<std::chrono::seconds>(end - start).count() * 1.0) / 60.0;
            }

            /// Clock class
            class StopW {
            private:
                std::chrono::steady_clock::time_point time_begin;
            public:
                StopW() {
                    time_begin = std::chrono::steady_clock::now();
                }

                double getElapsedMs() {
                    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
                    return getMsInterval(time_begin, time_end);
                }

                double getElapsedSec() {
                    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
                    return getSecInterval(time_begin, time_end);
                }
                    
                double getElapsedMin() {
                    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
                    return getMinInterval(time_begin, time_end);
                }

                void reset() {
                    time_begin = std::chrono::steady_clock::now();
                }
            };
        }
    }
}