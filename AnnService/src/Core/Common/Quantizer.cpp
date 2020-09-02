#include <inc/Core/Common/Quantizer.h>
#include <inc/Core/Common/PQQuantizer.h>

namespace SPTAG
{
    namespace COMMON
    {
        ErrorCode Quantizer::LoadQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_in, QuantizerType quantizerType) {
            std::shared_ptr<Quantizer> ptr;
            switch (quantizerType) {
            case QuantizerType::None:
                break;
            case QuantizerType::Undefined:
                break;
            case QuantizerType::PQQuantizer:
                DistanceUtils::Quantizer.reset(new PQQuantizer());
                return DistanceUtils::Quantizer->LoadQuantizer(p_in);
            }
            return ErrorCode::Success;
        }
    }
}