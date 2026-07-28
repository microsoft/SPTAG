#pragma once

#include <cstdint>

#include "rabitqlib/fastscan/fastscan.hpp"

namespace rabitqlib {
template <typename T>
struct BatchDataMap {
   public:
    explicit BatchDataMap(char* data, size_t padded_dim)
        : batch_bin_code_(reinterpret_cast<uint8_t*>(data))
        , f_add_(
              reinterpret_cast<T*>(data + (padded_dim * fastscan::kBatchSize / 8))
          )  // 1 bit code
        , f_rescale_(f_add_ + fastscan::kBatchSize)
        , f_error_(f_rescale_ + fastscan::kBatchSize) {}

    [[nodiscard]] uint8_t* bin_code() { return batch_bin_code_; }
    [[nodiscard]] T* f_add() { return f_add_; }
    [[nodiscard]] T* f_rescale() { return f_rescale_; }
    [[nodiscard]] T* f_error() { return f_error_; }

    static size_t data_bytes(size_t padded_dim) {
        return (padded_dim * fastscan::kBatchSize / 8) +
               (sizeof(T) * fastscan::kBatchSize * 3);
    }

   private:
    uint8_t* batch_bin_code_;
    T* f_add_;
    T* f_rescale_;
    T* f_error_;
};

template <typename T>
struct ConstBatchDataMap {
   public:
    explicit ConstBatchDataMap(const char* data, size_t padded_dim)
        : batch_bin_code_(reinterpret_cast<const uint8_t*>(data))
        , f_add_(
              reinterpret_cast<const T*>(data + (padded_dim * fastscan::kBatchSize / 8))
          )  // 1 bit code
        , f_rescale_(f_add_ + fastscan::kBatchSize)
        , f_error_(f_rescale_ + fastscan::kBatchSize) {}

    [[nodiscard]] const uint8_t* bin_code() const { return batch_bin_code_; }
    [[nodiscard]] const T* f_add() const { return f_add_; }
    [[nodiscard]] const T* f_rescale() const { return f_rescale_; }
    [[nodiscard]] const T* f_error() const { return f_error_; }

   private:
    const uint8_t* batch_bin_code_;
    const T* f_add_;
    const T* f_rescale_;
    const T* f_error_;
};

template <typename T>
struct QGBatchDataMap {
   public:
    explicit QGBatchDataMap(char* data, size_t padded_dim)
        : batch_bin_code_(reinterpret_cast<uint8_t*>(data))
        , f_add_(
              reinterpret_cast<T*>(data + (padded_dim * fastscan::kBatchSize / 8))
          )  // 1 bit code
        , f_rescale_(f_add_ + fastscan::kBatchSize) {}

    [[nodiscard]] uint8_t* bin_code() { return batch_bin_code_; }
    [[nodiscard]] T* f_add() { return f_add_; }
    [[nodiscard]] T* f_rescale() { return f_rescale_; }

    static size_t data_bytes(size_t padded_dim) {
        return (padded_dim * fastscan::kBatchSize / 8) +
               (sizeof(T) * fastscan::kBatchSize * 2);
    }

   private:
    uint8_t* batch_bin_code_;
    T* f_add_;
    T* f_rescale_;
};

template <typename T>
struct ConstQGBatchDataMap {
   public:
    explicit ConstQGBatchDataMap(const char* data, size_t padded_dim)
        : batch_bin_code_(reinterpret_cast<const uint8_t*>(data))
        , f_add_(
              reinterpret_cast<const T*>(data + (padded_dim * fastscan::kBatchSize / 8))
          )  // 1 bit code
        , f_rescale_(f_add_ + fastscan::kBatchSize) {}

    [[nodiscard]] const uint8_t* bin_code() { return batch_bin_code_; }
    [[nodiscard]] const T* f_add() { return f_add_; }
    [[nodiscard]] const T* f_rescale() { return f_rescale_; }

    static size_t data_bytes(size_t padded_dim) {
        return (padded_dim * fastscan::kBatchSize / 8) +
               (sizeof(T) * fastscan::kBatchSize * 2);
    }

   private:
    const uint8_t* batch_bin_code_;
    const T* f_add_;
    const T* f_rescale_;
};

template <typename T>
struct ExDataMap {
   public:
    explicit ExDataMap(char* data, size_t padded_dim, size_t ex_bits)
        : ex_code_(reinterpret_cast<uint8_t*>(data))
        , f_add_ex_(*reinterpret_cast<T*>(data + (padded_dim * ex_bits / 8)))
        , f_recale_ex_(*(reinterpret_cast<T*>(data + (padded_dim * ex_bits / 8)) + 1)) {}

    static size_t data_bytes(size_t padded_dim, size_t ex_bits) {
        return ex_bits > 0 ? (padded_dim * ex_bits / 8) + (sizeof(T) * 2) : 0;
    }

    [[nodiscard]] uint8_t* ex_code() { return ex_code_; }
    [[nodiscard]] T& f_add_ex() { return f_add_ex_; }
    [[nodiscard]] T& f_rescale_ex() { return f_recale_ex_; }

   private:
    uint8_t* ex_code_;
    T& f_add_ex_;
    T& f_recale_ex_;
};

template <typename T>
struct ConstExDataMap {
   public:
    explicit ConstExDataMap(const char* data, size_t padded_dim, size_t ex_bits)
        : ex_code_(reinterpret_cast<const uint8_t*>(data))
        , f_add_ex_(*reinterpret_cast<const T*>(data + (padded_dim * ex_bits / 8)))
        , f_recale_ex_(
              *(reinterpret_cast<const T*>(data + (padded_dim * ex_bits / 8)) + 1)
          ) {}

    [[nodiscard]] const uint8_t* ex_code() const { return ex_code_; }
    [[nodiscard]] const T& f_add_ex() const { return f_add_ex_; }
    [[nodiscard]] const T& f_rescale_ex() const { return f_recale_ex_; }

   private:
    const uint8_t* ex_code_;
    const T& f_add_ex_;
    const T& f_recale_ex_;
};

template <typename T>
struct BinDataMap {
   public:
    explicit BinDataMap(char* data, size_t padded_dim)
        : bin_code_(reinterpret_cast<uint64_t*>(data))
        , f_add_(*reinterpret_cast<T*>(data + (padded_dim / 8)))
        , f_rescale_(*(reinterpret_cast<T*>(data + (padded_dim / 8)) + 1))
        , f_error_(*(reinterpret_cast<T*>(data + (padded_dim / 8)) + 2)) {}

    [[nodiscard]] uint64_t* bin_code() { return bin_code_; }
    [[nodiscard]] T& f_add() { return f_add_; }
    [[nodiscard]] T& f_rescale() { return f_rescale_; }
    [[nodiscard]] T& f_error() { return f_error_; }

    static size_t data_bytes(size_t padded_dim) {
        return (padded_dim / 8) + (sizeof(T) * 3);
    }

   private:
    uint64_t* bin_code_;
    T& f_add_;
    T& f_rescale_;
    T& f_error_;
};

template <typename T>
struct ConstBinDataMap {
   public:
    explicit ConstBinDataMap(const char* data, size_t padded_dim)
        : bin_code_(reinterpret_cast<const uint64_t*>(data))
        , f_add_(*reinterpret_cast<const T*>(data + (padded_dim / 8)))
        , f_rescale_(*(reinterpret_cast<const T*>(data + (padded_dim / 8)) + 1))
        , f_error_(*(reinterpret_cast<const T*>(data + (padded_dim / 8)) + 2)) {}

    [[nodiscard]] const uint64_t* bin_code() { return bin_code_; }
    [[nodiscard]] const T& f_add() { return f_add_; }
    [[nodiscard]] const T& f_rescale() { return f_rescale_; }
    [[nodiscard]] const T& f_error() { return f_error_; }

    static size_t data_bytes(size_t padded_dim) {
        return (padded_dim / 8) + (sizeof(T) * 3);
    }

   private:
    const uint64_t* bin_code_;
    const T& f_add_;
    const T& f_rescale_;
    const T& f_error_;
};
}  // namespace rabitqlib