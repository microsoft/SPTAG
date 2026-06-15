#pragma once

#include <chrono>

namespace rabitqlib {
class StopW {
    using clock = std::chrono::steady_clock;
    clock::time_point time_begin_;

    [[nodiscard]] clock::duration elapsed() const { return clock::now() - time_begin_; }

   public:
    StopW() : time_begin_(clock::now()) {}

    void reset() { time_begin_ = clock::now(); }

    [[nodiscard]] float get_elapsed_sec() const {
        return std::chrono::duration<float>(elapsed()).count();
    }

    [[nodiscard]] float get_elapsed_mili() const {
        return std::chrono::duration<float, std::milli>(elapsed()).count();
    }

    [[nodiscard]] float get_elapsed_micro() const {
        return std::chrono::duration<float, std::micro>(elapsed()).count();
    }

    [[nodiscard]] float get_elapsed_nano() const {
        return std::chrono::duration<float, std::nano>(elapsed()).count();
    }
};
}  // namespace rabitqlib