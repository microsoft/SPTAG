// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_PIPEPQ_H_
#define _SPTAG_SPANN_PIPEPQ_H_

#include "inc/Core/Common.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace SPTAG::SPANN {

class PipePQTable
{
public:
    bool Load(const std::string& pivotsPath, int expectedChunks)
    {
        std::ifstream in(pivotsPath, std::ios::binary);
        if (!in) return false;

        std::uint32_t rows = 0, cols = 0;
        if (!ReadHeader(in, 0, rows, cols) || rows != 5 || cols != 1) return false;

        std::uint64_t offsets[5] = {0, 0, 0, 0, 0};
        in.seekg(8, std::ios::beg);
        in.read(reinterpret_cast<char*>(offsets), sizeof(offsets));
        if (!in) return false;

        std::uint32_t tableCols = 0;
        if (!ReadFloatMatrix(in, offsets[0], 256, tableCols, m_tables)) return false;
        m_dim = static_cast<int>(tableCols);
        std::uint32_t centroidCols = 0;
        if (!ReadFloatMatrix(in, offsets[1], 0, centroidCols, m_centroid)) return false;
        if (centroidCols != 1 || static_cast<int>(m_centroid.size()) != m_dim) return false;

        std::uint32_t offRows = 0, offCols = 0;
        if (!ReadHeader(in, offsets[3], offRows, offCols) || offCols != 1) return false;
        if (expectedChunks > 0 && offRows != static_cast<std::uint32_t>(expectedChunks + 1)) return false;
        m_chunkOffsets.resize(offRows);
        in.seekg(static_cast<std::streamoff>(offsets[3] + 8), std::ios::beg);
        in.read(reinterpret_cast<char*>(m_chunkOffsets.data()),
                static_cast<std::streamsize>(m_chunkOffsets.size() * sizeof(std::uint32_t)));
        if (!in || m_chunkOffsets.empty() || m_chunkOffsets.front() != 0 ||
            m_chunkOffsets.back() != static_cast<std::uint32_t>(m_dim)) {
            return false;
        }

        m_chunks = static_cast<int>(m_chunkOffsets.size()) - 1;
        return m_chunks > 0 && (expectedChunks <= 0 || m_chunks == expectedChunks);
    }

    int Dim() const { return m_dim; }
    int Chunks() const { return m_chunks; }

    void PopulateDistances(const float* query, float* distLut, DistCalcMethod metric) const
    {
        std::fill(distLut, distLut + static_cast<size_t>(m_chunks) * 256, 0.0f);
        for (int chunk = 0; chunk < m_chunks; ++chunk) {
            float* chunkDists = distLut + static_cast<size_t>(chunk) * 256;
            const std::uint32_t begin = m_chunkOffsets[chunk];
            const std::uint32_t end = m_chunkOffsets[chunk + 1];
            for (std::uint32_t d = begin; d < end; ++d) {
                if (metric == DistCalcMethod::InnerProduct) {
                    const float q = query[d];
                    for (int center = 0; center < 256; ++center)
                        chunkDists[center] -= q * m_tables[static_cast<size_t>(center) * m_dim + d];
                } else {
                    // Match PipeANN's AVX2 scalar fallback exactly: subtraction is
                    // performed in float, then the square is evaluated in double
                    // before accumulating as float.
                    const float q = query[d] - m_centroid[d];
                    for (int center = 0; center < 256; ++center) {
                        const float diff = m_tables[static_cast<size_t>(center) * m_dim + d] - q;
                        chunkDists[center] += static_cast<float>(
                            static_cast<double>(diff) * static_cast<double>(diff));
                    }
                }
            }
        }
    }

    void Encode(const float* vector, std::uint8_t* code) const
    {
        for (int chunk = 0; chunk < m_chunks; ++chunk) {
            const std::uint32_t begin = m_chunkOffsets[chunk];
            const std::uint32_t end = m_chunkOffsets[chunk + 1];
            float best = std::numeric_limits<float>::max();
            int bestCenter = 0;
            for (int center = 0; center < 256; ++center) {
                float dist = 0.0f;
                for (std::uint32_t d = begin; d < end; ++d) {
                    const float diff = m_tables[static_cast<size_t>(center) * m_dim + d] -
                                       (vector[d] - m_centroid[d]);
                    dist += diff * diff;
                }
                if (dist < best) {
                    best = dist;
                    bestCenter = center;
                }
            }
            code[chunk] = static_cast<std::uint8_t>(bestCenter);
        }
    }

private:
    static bool ReadHeader(std::ifstream& in, std::uint64_t offset,
                           std::uint32_t& rows, std::uint32_t& cols)
    {
        in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        return static_cast<bool>(in);
    }

    static bool ReadFloatMatrix(std::ifstream& in, std::uint64_t offset,
                                std::uint32_t expectedRows, std::uint32_t& cols,
                                std::vector<float>& out)
    {
        std::uint32_t rows = 0;
        if (!ReadHeader(in, offset, rows, cols)) return false;
        if (expectedRows != 0 && rows != expectedRows) return false;
        out.resize(static_cast<size_t>(rows) * cols);
        in.seekg(static_cast<std::streamoff>(offset + 8), std::ios::beg);
        in.read(reinterpret_cast<char*>(out.data()),
                static_cast<std::streamsize>(out.size() * sizeof(float)));
        return static_cast<bool>(in);
    }

    int m_dim = 0;
    int m_chunks = 0;
    std::vector<float> m_tables;              // [256][dim], centered chunk centroids
    std::vector<float> m_centroid;            // [dim]
    std::vector<std::uint32_t> m_chunkOffsets; // [chunks + 1]
};

} // namespace SPTAG::SPANN

#endif // _SPTAG_SPANN_PIPEPQ_H_
