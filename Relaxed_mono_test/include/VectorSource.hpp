#pragma once

#include <memory>

#include "VectorProvider.h"
#include "RandomVectorProvider.hpp"
#include "PreloadedVectorProvider.hpp"
#include "Config.h"

class VectorSource {
public:
    using Vector = std::vector<int8_t>;

    VectorSource(std::unique_ptr<IVectorProvider> insertProvider,
                 std::unique_ptr<IVectorProvider> queryProvider)
        : m_insertProvider(std::move(insertProvider)),
          m_queryProvider(std::move(queryProvider)) {}

    bool getVectorForInsert(Vector& vec) {
        return m_insertProvider->getNext(vec);
    }

    bool getVectorForQuery(Vector& vec) {
        return m_queryProvider->getNext(vec);
    }

    void resetQueryProvider() {
        m_queryProvider->reset();
    }

private:
    std::unique_ptr<IVectorProvider> m_insertProvider;
    std::unique_ptr<IVectorProvider> m_queryProvider;
};
