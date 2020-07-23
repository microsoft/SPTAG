#include "inc/Helper/DynamicNeighbors.h"

#include <fstream>

using namespace SPTAG::Helper;

DynamicNeighbors::DynamicNeighbors(const int* p_data, const int p_length)
    : c_data(p_data),
    c_length(p_length)
{
}


DynamicNeighbors:: ~DynamicNeighbors()
{
}


int
DynamicNeighbors::operator[](const int p_id) const
{
    if (p_id < c_length && p_id >= 0)
    {
        return c_data[p_id];
    }

    return -1;
}


int
DynamicNeighbors::Size() const
{
    return c_length;
}


DynamicNeighborsSet::DynamicNeighborsSet(const char* p_filePath)
{
    std::ifstream graph(p_filePath, std::ios::binary);

    if (!graph.is_open())
    {
        fprintf(stderr, "Failed open graph file: %s\n", p_filePath);
        exit(1);
    }

    graph.read(reinterpret_cast<char*>(&m_vectorCount), sizeof(m_vectorCount));

    m_neighborOffset.reset(new int[m_vectorCount + 1]);
    m_neighborOffset[0] = 0;
    graph.read(reinterpret_cast<char*>(m_neighborOffset.get() + 1), m_vectorCount * sizeof(int));

    size_t graphSize = static_cast<size_t>(m_neighborOffset[m_vectorCount]);

    fprintf(stderr, "Vector count: %d, Graph size: %llu\n", m_vectorCount, graphSize);

    m_data.reset(new int[graphSize]);
    graph.read(reinterpret_cast<char*>(m_data.get()), graphSize * sizeof(int));

    if (graph.gcount() != graphSize * sizeof(int))
    {
        fprintf(stderr,
            "Failed read graph: size not match, expected %llu, actually %llu\n",
            static_cast<uint64_t>(graphSize * sizeof(int)),
            static_cast<uint64_t>(graph.gcount()));

        exit(1);
    }

    graph.close();
}


DynamicNeighborsSet::~DynamicNeighborsSet()
{
}


DynamicNeighbors DynamicNeighborsSet::operator[](const int p_id) const
{
    if (p_id >= m_vectorCount)
    {
        return DynamicNeighbors(nullptr, 0);
    }

    return DynamicNeighbors(m_data.get() + static_cast<uint64_t>(m_neighborOffset[p_id]),
        m_neighborOffset[p_id + 1] - m_neighborOffset[p_id]);
}
