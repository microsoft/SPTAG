#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/DistanceUtils.h"

using namespace SPTAG;
using namespace SPTAG::COMMON;

#define DefineVectorValueType(Name, Type) template int Utils::GetBase<Type>();
#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

template <typename T> void Utils::BatchNormalize(T *data, SizeType row, DimensionType col, int base, int threads)
{
    std::vector<std::thread> mythreads;
    mythreads.reserve(threads);
    std::atomic_size_t sent(0);
    for (int tid = 0; tid < threads; tid++)
    {
        mythreads.emplace_back([&, tid]() {
            size_t i = 0;
            while (true)
            {
                i = sent.fetch_add(1);
                if (i < row)
                {
                    SPTAG::COMMON::Utils::Normalize(data + i * (size_t)col, col, base);
                }
                else
                {
                    return;
                }
            }
        });
    }
    for (auto &t : mythreads)
    {
        t.join();
    }
    mythreads.clear();
}

#define DefineVectorValueType(Name, Type)                                                                              \
    template void Utils::BatchNormalize<Type>(Type * data, SizeType row, DimensionType col, int base, int threads);
#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType