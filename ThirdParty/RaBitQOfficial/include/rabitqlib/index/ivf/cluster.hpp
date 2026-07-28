#pragma once

#include <cassert>

#include "rabitqlib/defines.hpp"

namespace rabitqlib::ivf {

/**
 * @brief Cluster is used for ivf index with rabitq+. Components are only used for record
 * the addresses for different part of data.
 *
 */
class Cluster {
   private:
    size_t num_;                  // Num of vectors in this cluster
    char* batch_data_ = nullptr;  // RaBitQ code and factors
    char* ex_data_ = nullptr;     // Ex code and factors
    PID* ids_ = nullptr;          // PID of vectors

   public:
    explicit Cluster(size_t, char*, char*, PID*);
    Cluster(const Cluster& other);
    Cluster(Cluster&& other) noexcept;
    ~Cluster() {}

    [[nodiscard]] char* batch_data() const { return this->batch_data_; }

    [[nodiscard]] char* ex_data() const { return ex_data_; }

    [[nodiscard]] PID* ids() const { return this->ids_; }

    [[nodiscard]] size_t num() const { return num_; }
};

inline Cluster::Cluster(size_t num, char* batch_data, char* ex_data, PID* ids)
    : num_(num), batch_data_(batch_data), ex_data_(ex_data), ids_(ids) {}

inline Cluster::Cluster(const Cluster& other)
    : num_(other.num_)
    , batch_data_(other.batch_data_)
    , ex_data_(other.ex_data_)
    , ids_(other.ids_) {}

inline Cluster::Cluster(Cluster&& other) noexcept
    : num_(other.num_)
    , batch_data_(other.batch_data_)
    , ex_data_(other.ex_data_)
    , ids_(other.ids_) {}
}  // namespace rabitqlib::ivf