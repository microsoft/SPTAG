// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// SQ8Compressor: Scalar Quantization to 8-bit for HeadIndex vectors.
// Compresses float32 vectors to uint8 with per-dimension min/scale.
// Used to keep evicted tenant head vectors in memory at 25% size.
//
#ifndef _SPTAG_SQ8_COMPRESSOR_H_
#define _SPTAG_SQ8_COMPRESSOR_H_

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>

namespace SPTAG {
namespace Cache {

struct SQ8CompressedVectors {
    int num_vectors = 0;
    int dimension = 0;
    std::vector<float> mins;      // per-dimension min (dimension floats)
    std::vector<float> scales;    // per-dimension scale (dimension floats)
    std::vector<uint8_t> data;    // quantized data (num_vectors × dimension bytes)

    size_t MemoryBytes() const {
        return sizeof(*this)
            + mins.capacity() * sizeof(float)
            + scales.capacity() * sizeof(float)
            + data.capacity();
    }

    // Compress float32 vectors to SQ8
    static std::shared_ptr<SQ8CompressedVectors> Compress(
        const float* vectors, int num, int dim)
    {
        auto result = std::make_shared<SQ8CompressedVectors>();
        result->num_vectors = num;
        result->dimension = dim;
        result->mins.resize(dim);
        result->scales.resize(dim);
        result->data.resize((size_t)num * dim);

        // Find per-dimension min/max
        for (int d = 0; d < dim; d++) {
            float vmin = vectors[d];
            float vmax = vectors[d];
            for (int i = 1; i < num; i++) {
                float v = vectors[i * dim + d];
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
            result->mins[d] = vmin;
            float range = vmax - vmin;
            result->scales[d] = (range > 1e-10f) ? (255.0f / range) : 0.0f;
        }

        // Quantize
        for (int i = 0; i < num; i++) {
            for (int d = 0; d < dim; d++) {
                float v = vectors[i * dim + d];
                float q = (v - result->mins[d]) * result->scales[d];
                result->data[i * dim + d] = static_cast<uint8_t>(
                    std::min(255.0f, std::max(0.0f, std::round(q))));
            }
        }
        return result;
    }

    // Decompress SQ8 back to float32
    void Decompress(float* output) const {
        for (int i = 0; i < num_vectors; i++) {
            for (int d = 0; d < dimension; d++) {
                float scale = scales[d];
                float inv_scale = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;
                output[i * dimension + d] =
                    mins[d] + data[i * dimension + d] * inv_scale;
            }
        }
    }

    // Decompress to a newly allocated buffer
    std::vector<float> Decompress() const {
        std::vector<float> out(num_vectors * dimension);
        Decompress(out.data());
        return out;
    }
};

}  // namespace Cache
}  // namespace SPTAG

#endif // _SPTAG_SQ8_COMPRESSOR_H_
