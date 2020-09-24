#pragma once

#include <memory>

namespace SPTAG {
namespace Helper {
    class DynamicNeighbors
    {
    public:
        DynamicNeighbors(const int* p_data, const int p_length);

        ~DynamicNeighbors();

        int operator[](const int p_id) const;

        int Size() const;

    private:
        const int* const c_data;

        const int c_length;
    };


    class DynamicNeighborsSet
    {
    public:
        DynamicNeighborsSet(const char* p_filePath);

        ~DynamicNeighborsSet();

        DynamicNeighbors operator[](const int p_id) const;

        int VectorCount() const
        {
            return m_vectorCount;
        }

    private:
        std::unique_ptr<int[]> m_data;

        std::unique_ptr<int[]> m_neighborOffset;

        int m_vectorCount;
    };
}
}

