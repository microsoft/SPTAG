#pragma once

#include <cstddef>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/memory.hpp"

namespace rabitqlib::buffer {
/**
 * @brief sorted linear buffer, used as beam set for graph-based ANN search. In symphonyqg,
 * the search buffer may contain duplicate id with different distances
 *
 */
template <typename T = float>
class SearchBuffer {
   private:
    std::vector<AnnCandidate<T>, memory::AlignedAllocator<AnnCandidate<T>>> data_;
    size_t size_ = 0, cur_ = 0, capacity_;

    [[nodiscard]] auto binary_search(T dist) const {
        size_t lo = 0;
        size_t len = size_;
        size_t half;
        while (len > 1) {
            half = len >> 1;
            len -= half;
            lo += static_cast<size_t>(data_[lo + half - 1].distance < dist) * half;
        }
        return (lo < size_ && data_[lo].distance < dist) ? lo + 1 : lo;
    }

    // set top bit to 1 as checked
    static void set_checked(PID& data_id) { data_id |= (1 << 31); }

    [[nodiscard]] static auto is_checked(PID data_id) -> bool {
        return static_cast<bool>(data_id >> 31);
    }

   public:
    SearchBuffer() = default;

    explicit SearchBuffer(size_t capacity) : data_(capacity + 1), capacity_(capacity) {}

    // insert a data point into buffer
    void insert(PID data_id, T dist) {
        if (is_full(dist)) {
            return;
        }

        size_t lo = binary_search(dist);
        std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(AnnCandidate<T>));
        data_[lo] = AnnCandidate<T>(data_id, dist);
        size_ += static_cast<size_t>(size_ < capacity_);
        cur_ = lo < cur_ ? lo : cur_;
    }

    // get unchecked candidate with minimum distance
    PID pop() {
        PID cur_id = data_[cur_].id;
        set_checked(data_[cur_].id);
        ++cur_;
        while (cur_ < size_ && is_checked(data_[cur_].id)) {
            ++cur_;
        }
        return cur_id;
    }

    void clear() {
        size_ = 0;
        cur_ = 0;
    }

    // return candidate id for next pop()
    [[nodiscard]] auto next_id() const { return data_[cur_].id; }

    [[nodiscard]] auto has_next() const -> bool { return cur_ < size_; }

    void resize(size_t new_size) {
        this->capacity_ = new_size;
        data_ = std::vector<AnnCandidate<T>, memory::AlignedAllocator<AnnCandidate<T>>>(
            capacity_ + 1
        );
    }

    void copy_results(PID* knn) const {
        for (size_t i = 0; i < size_; ++i) {
            knn[i] = data_[i].id;
        }
    }

    void copy_results(PID* knn, T* dist) const {
        for (size_t i = 0; i < size_; ++i) {
            knn[i] = data_[i].id;
            dist[i] = data_[i].distance;
        }
    }

    T top_dist() const {
        return is_full() ? data_[size_ - 1].distance : std::numeric_limits<T>::max();
    }

    [[nodiscard]] auto is_full() const -> bool { return size_ == capacity_; }

    // judge if dist can be inserted into buffer
    [[nodiscard]] auto is_full(T dist) const -> bool { return dist > top_dist(); }

    const std::vector<AnnCandidate<T>, memory::AlignedAllocator<AnnCandidate<T>>>& data() {
        return data_;
    }
};
}  // namespace rabitqlib::buffer