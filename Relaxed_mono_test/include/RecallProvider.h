#pragma once

#include <vector>
#include <cstdint>

class IRecallProvider {
public:
    virtual ~IRecallProvider() = default;
    virtual bool load(std::vector<int8_t>& queryVectors, std::vector<int32_t>& groundTruth, int& queryCount, int& dim, int& topk) = 0;
};