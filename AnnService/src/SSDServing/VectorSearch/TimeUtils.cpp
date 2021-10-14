#include "inc/SSDServing/VectorSearch/TimeUtils.h"

namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch {
            namespace TimeUtils {
                /// Clock class
                
                StopW::StopW() {
                    time_begin = std::chrono::steady_clock::now();
                }

                double StopW::getElapsedMs() {
                    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
                    return getMsInterval(time_begin, time_end);
                }

                double StopW::getElapsedSec() {
                    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
                    return getSecInterval(time_begin, time_end);
                }

                double StopW::getElapsedMin() {
                    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
                    return getMinInterval(time_begin, time_end);
                }

                void StopW::reset() {
                    time_begin = std::chrono::steady_clock::now();
                }

                double getMsInterval(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
                    return (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1.0) / 1000.0;
                }

                double getSecInterval(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
                    return (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 1.0) / 1000.0;
                }

                double getMinInterval(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
                    return (std::chrono::duration_cast<std::chrono::seconds>(end - start).count() * 1.0) / 60.0;
                }

            }
        }
    }
}