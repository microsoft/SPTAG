#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/fht_avx.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib {

enum class RotatorType : uint8_t { MatrixRotator, FhtKacRotator };

// abstract rotator
template <typename T>
class Rotator {
   protected:
    size_t dim_;
    size_t padded_dim_;

   public:
    explicit Rotator() = default;
    explicit Rotator(size_t dim, size_t padded_dim) : dim_(dim), padded_dim_(padded_dim) {};
    virtual ~Rotator() = default;
    virtual void rotate(const T* src, T* dst) const = 0;
    virtual void load(std::ifstream&) = 0;
    virtual void save(std::ofstream&) const = 0;
    // Buffer I/O
    virtual void load(const char *data) = 0;
    virtual void save(char *data) const = 0; // dump to buffer
    virtual size_t dump_bytes() const = 0;
    [[nodiscard]] size_t size() const { return this->padded_dim_; }
};

namespace rotator_impl {

// get padding requirement for different rotator
inline size_t padding_requirement(size_t dim, RotatorType type) {
    if (type == RotatorType::MatrixRotator) {
        return dim;
    }
    if (type == RotatorType::FhtKacRotator) {
        return round_up_to_multiple(dim, 64);
    }
    std::cerr << "Invalid rotator type in padding_requirement()\n" << std::flush;
    exit(1);
}

template <typename T = float>
class MatrixRotator : public Rotator<T> {
   private:
    RowMajorMatrix<T> rand_mat_;  // Rotation Maxtrix
   public:
    explicit MatrixRotator(size_t dim, size_t padded_dim)
        : Rotator<T>(dim, padded_dim), rand_mat_(dim, padded_dim) {
        RowMajorMatrix<T> rand = random_gaussian_matrix<T>(padded_dim, padded_dim);
        Eigen::HouseholderQR<RowMajorMatrix<T>> qr(rand);
        RowMajorMatrix<T> q_inv =
            qr.householderQ().transpose();  // inverse of orthogonal mat is its inverse

        // the random matrix only need the first dim rows, since we just pad zeros for
        // the vector to be rotated to padded dimension
        std::memcpy(&rand_mat_(0, 0), &q_inv(0, 0), sizeof(T) * dim * padded_dim);
    }
    MatrixRotator() = default;
    ~MatrixRotator() = default;

    MatrixRotator& operator=(const MatrixRotator& other) {
        this->dim_ = other.dim_;
        this->padded_dim_ = other.padded_dim_;
        this->rand_mat_ = other.rand_mat_;
        return *this;
    }

    void load(std::ifstream& input) override {
        input.read(
            reinterpret_cast<char*>(rand_mat_.data()),
            static_cast<long>(sizeof(float) * this->dim_ * this->padded_dim_)
        );
    }

    void save(std::ofstream& output) const override {
        output.write(
            reinterpret_cast<const char*>(rand_mat_.data()),
            (sizeof(float) * this->dim_ * this->padded_dim_)
        );
    }

    void load(const char *data) override {
        std::memcpy(rand_mat_.data(), data, sizeof(float) * this->dim_ * this->padded_dim_);
    }

    void save(char *data) const override {
        std::memcpy(data, rand_mat_.data(), sizeof(float) * this->dim_ * this->padded_dim_);
    }

    size_t dump_bytes() const override {
        return sizeof(float) * this->dim_ * this->padded_dim_;
  }

    void rotate(const T* vec, T* rotated_vec) const override {
        ConstRowMajorMatrixMap<T> v(vec, 1, this->dim_);
        RowMajorMatrixMap<T> rv(rotated_vec, 1, this->padded_dim_);
        rv = v * this->rand_mat_;
    }
};

static inline void flip_sign(const uint8_t* flip, float* data, size_t dim) {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    constexpr size_t kFloatsPerChunk = 64;  // Process 64 floats per iteration
    // constexpr size_t bits_per_chunk = floats_per_chunk;  // 64 bits = 8 bytes

    static_assert(
        kFloatsPerChunk % 16 == 0,
        "floats_per_chunk must be divisible by AVX512 register width"
    );

    for (size_t i = 0; i < dim; i += kFloatsPerChunk) {
        // Load 64 bits (8 bytes) from the bit sequence
        uint64_t mask_bits;
        std::memcpy(&mask_bits, &flip[i / 8], sizeof(mask_bits));

        // Split into four 16-bit mask segments
        const __mmask16 mask0 = _cvtu32_mask16(static_cast<uint32_t>(mask_bits & 0xFFFF));
        const __mmask16 mask1 =
            _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 16) & 0xFFFF));
        const __mmask16 mask2 =
            _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 32) & 0xFFFF));
        const __mmask16 mask3 =
            _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 48) & 0xFFFF));

        // Prepare sign-flip constant
        const __m512 sign_flip = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000));

        // Process 16 floats at a time with each mask segment
        __m512 vec0 = _mm512_loadu_ps(&data[i]);
        vec0 = _mm512_mask_xor_ps(vec0, mask0, vec0, sign_flip);
        _mm512_storeu_ps(&data[i], vec0);

        __m512 vec1 = _mm512_loadu_ps(&data[i + 16]);
        vec1 = _mm512_mask_xor_ps(vec1, mask1, vec1, sign_flip);
        _mm512_storeu_ps(&data[i + 16], vec1);

        __m512 vec2 = _mm512_loadu_ps(&data[i + 32]);
        vec2 = _mm512_mask_xor_ps(vec2, mask2, vec2, sign_flip);
        _mm512_storeu_ps(&data[i + 32], vec2);

        __m512 vec3 = _mm512_loadu_ps(&data[i + 48]);
        vec3 = _mm512_mask_xor_ps(vec3, mask3, vec3, sign_flip);
        _mm512_storeu_ps(&data[i + 48], vec3);
    }
#elif defined(__AVX2__)
   // Process 32 floats (4 AVX2 registers) per iteration
    constexpr size_t kFloatsPerChunk = 32; 
    
    const __m256i bit_select = _mm256_setr_epi32(
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
    );
    const __m256 sign_flip = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    // utility lambda to create a mask for flipping signs
    auto create_mask = [&](uint8_t byte_mask) -> __m256 {
        __m256i mask_bits = _mm256_set1_epi32(byte_mask);
        __m256i test = _mm256_and_si256(mask_bits, bit_select);
        __m256i cmp = _mm256_cmpeq_epi32(test, bit_select);
        return _mm256_and_ps(_mm256_castsi256_ps(cmp), sign_flip);
    } ;

    for (size_t i = 0; i < dim; i += kFloatsPerChunk) {
        uint32_t mask_bits;
        std::memcpy(&mask_bits, &flip[i / 8], sizeof(mask_bits));

        for (int b = 0; b < 4; ++b) {
            __m256 xor_mask = create_mask((mask_bits >> (b * 8)) & 0xFF);
            __m256 vec = _mm256_loadu_ps(&data[i + b * 8]);
            vec = _mm256_xor_ps(vec, xor_mask);
            _mm256_storeu_ps(&data[i + b * 8], vec);
        }
    }
#else
    std:: cerr << "Sign flip requires AVX512 or AVX2 support!\n";
    exit(1);
#endif
}

class FhtKacRotator : public Rotator<float> {
   private:
    std::vector<uint8_t> flip_;
    std::function<void(float*)> fht_float_ = helper_float_6;
    size_t trunc_dim_ = 0;
    float fac_ = 0;

    static constexpr size_t kByteLen = 8;

   public:
    explicit FhtKacRotator(size_t dim, size_t padded_dim)
        : Rotator<float>(dim, padded_dim), flip_(4 * padded_dim / kByteLen) {
        std::random_device rd;   // Seed
        std::mt19937 gen(rd());  // Mersenne Twister RNG

        // Uniform distribution in the range [0, 255]
        std::uniform_int_distribution<int> dist(0, 255);

        // Generate a single random uint8_t value
        for (auto& i : flip_) {
            i = static_cast<uint8_t>(dist(gen));
        }

        // TODO(lib): is it portable?
        size_t bottom_log_dim = floor_log2(dim);
        trunc_dim_ = 1 << bottom_log_dim;
        fac_ = 1.0F / std::sqrt(static_cast<float>(trunc_dim_));

        switch (bottom_log_dim) {
            case 6:
                this->fht_float_ = helper_float_6;
                break;
            case 7:
                this->fht_float_ = helper_float_7;
                break;
            case 8:
                this->fht_float_ = helper_float_8;
                break;
            case 9:
                this->fht_float_ = helper_float_9;
                break;
            case 10:
                this->fht_float_ = helper_float_10;
                break;
            case 11:
                this->fht_float_ = helper_float_11;
                break;
            default:
                // TODO(lib): should we do more?
                std::cerr << "dimension of vector is too big\n";
                exit(1);
        }
    }
    FhtKacRotator() = default;
    ~FhtKacRotator() override = default;

    void load(std::ifstream& input) override {
        input.read(
            reinterpret_cast<char*>(flip_.data()),
            static_cast<long>(sizeof(uint8_t) * flip_.size())
        );
    }

    void save(std::ofstream& output) const override {
        output.write(
            reinterpret_cast<const char*>(flip_.data()),
            static_cast<long>(sizeof(uint8_t) * flip_.size())
        );
    }

    void load(const char *data) override {
        std::memcpy(flip_.data(), data, sizeof(uint8_t) * flip_.size());
    }

    void save(char *data) const override {
        std::memcpy(data, flip_.data(), sizeof(uint8_t) * flip_.size());
    }

    size_t dump_bytes() const override {
        return sizeof(uint8_t) * flip_.size();
    }

    FhtKacRotator& operator=(const FhtKacRotator& other) {
        this->dim_ = other.dim_;
        this->padded_dim_ = other.padded_dim_;
        this->flip_ = other.flip_;
        this->fht_float_ = other.fht_float_;
        this->trunc_dim_ = other.trunc_dim_;
        this->fac_ = other.fac_;
        return *this;
    }

    static void kacs_walk(float* data, size_t len) {
    #if defined(__AVX512F__)
        // ! len % 32 == 0;
        for (size_t i = 0; i < len / 2; i += 16) {
            __m512 x = _mm512_loadu_ps(&data[i]);
            __m512 y = _mm512_loadu_ps(&data[i + (len / 2)]);

            __m512 new_x = _mm512_add_ps(x, y);
            __m512 new_y = _mm512_sub_ps(x, y);

            _mm512_storeu_ps(&data[i], new_x);
            _mm512_storeu_ps(&data[i + (len / 2)], new_y);
        }
    #elif defined(__AVX2__)
        // ! len % 16 == 0;
        for (size_t i = 0; i < len / 2; i += 8) {
            __m256 x = _mm256_loadu_ps(&data[i]);
            __m256 y = _mm256_loadu_ps(&data[i + (len / 2)]);

            __m256 new_x = _mm256_add_ps(x, y);
            __m256 new_y = _mm256_sub_ps(x, y);

            _mm256_storeu_ps(&data[i], new_x);
            _mm256_storeu_ps(&data[i + (len / 2)], new_y);
        }
    #else
        std:: cerr << "FhtKacRotator requires AVX512 or AVX2 support!\n";
        exit(1);
    #endif
    }

    void rotate(const float* data, float* rotated_vec) const override {
        std::memcpy(rotated_vec, data, sizeof(float) * dim_);
        std::fill(rotated_vec + dim_, rotated_vec + padded_dim_, 0);

        if (trunc_dim_ == padded_dim_) {
            flip_sign(flip_.data(), rotated_vec, padded_dim_);
            fht_float_(rotated_vec);
            vec_rescale(rotated_vec, trunc_dim_, fac_);

            flip_sign(flip_.data() + (padded_dim_ / kByteLen), rotated_vec, padded_dim_);
            fht_float_(rotated_vec);
            vec_rescale(rotated_vec, trunc_dim_, fac_);

            flip_sign(
                flip_.data() + (2 * padded_dim_ / kByteLen), rotated_vec, padded_dim_
            );
            fht_float_(rotated_vec);
            vec_rescale(rotated_vec, trunc_dim_, fac_);

            flip_sign(
                flip_.data() + (3 * padded_dim_ / kByteLen), rotated_vec, padded_dim_
            );
            fht_float_(rotated_vec);
            vec_rescale(rotated_vec, trunc_dim_, fac_);

            return;
        }

        size_t start = padded_dim_ - trunc_dim_;

        flip_sign(flip_.data(), rotated_vec, padded_dim_);
        fht_float_(rotated_vec);
        vec_rescale(rotated_vec, trunc_dim_, fac_);
        kacs_walk(rotated_vec, padded_dim_);

        flip_sign(flip_.data() + (padded_dim_ / kByteLen), rotated_vec, padded_dim_);
        fht_float_(rotated_vec + start);
        vec_rescale(rotated_vec + start, trunc_dim_, fac_);
        kacs_walk(rotated_vec, padded_dim_);

        flip_sign(flip_.data() + (2 * padded_dim_ / kByteLen), rotated_vec, padded_dim_);
        fht_float_(rotated_vec);
        vec_rescale(rotated_vec, trunc_dim_, fac_);
        kacs_walk(rotated_vec, padded_dim_);

        flip_sign(flip_.data() + (3 * padded_dim_ / kByteLen), rotated_vec, padded_dim_);
        fht_float_(rotated_vec + start);
        vec_rescale(rotated_vec + start, trunc_dim_, fac_);
        kacs_walk(rotated_vec, padded_dim_);

        // This can be removed if we don't care about the absolute value of
        // similarities.
        vec_rescale(rotated_vec, padded_dim_, 0.25F);
    }
};
}  // namespace rotator_impl

// for given dim & type, set rotator, return padded dimension
template <typename T>
Rotator<T>* choose_rotator(
    size_t dim, RotatorType type = RotatorType::FhtKacRotator, size_t padded_dim = 0
) {
    if (padded_dim == 0) {
        padded_dim = rotator_impl::padding_requirement(dim, type);
        if (padded_dim != dim) {
            std::cerr << "vectors are padded to " << padded_dim
                      << " dimensions for aligned computation\n";
            std::cerr << "check rabitqlib/utils/rotator.hpp in case that users want to "
                         "remove padding\n";
        }
    }

    if (padded_dim != rotator_impl::padding_requirement(padded_dim, type)) {
        std::cerr << "Invalid padded dim for the given rotator type\n" << std::flush;
        exit(1);
    }

    if (type == RotatorType::FhtKacRotator) {
        if (!std::is_same_v<T, float>) {
            std::cerr << "FhtKacRotator is only for float type currently\n";
            exit(1);
        }
        std::cerr << "FhtKacRotator is selected\n";
        return ::new rotator_impl::FhtKacRotator(dim, padded_dim);
    }

    if (type == RotatorType::MatrixRotator) {
        std::cerr << "MatrixRotator is selected\n";
        return ::new rotator_impl::MatrixRotator<T>(dim, padded_dim);
    }

    std::cerr << "Invaid rotator type in choose_rotator()\n";
    exit(1);
}
}  // namespace rabitqlib
