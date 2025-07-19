#pragma once

#include <vector>
#include <cstdint>

class IVectorProvider {
public:
    using Vector = std::vector<int8_t>;
    virtual ~IVectorProvider() = default;
    virtual bool getNext(Vector& output) = 0;
    virtual bool reset() = 0;
};