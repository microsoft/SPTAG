#include <inc/Core/Common/IQuantizer.h>
#include <inc/Core/Common/PQQuantizer.h>

namespace SPTAG
{
    namespace COMMON
    {
        ErrorCode IQuantizer::LoadQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_in, QuantizerType quantizerType, VectorValueType reconstructType) {
            std::shared_ptr<IQuantizer> ptr;
            switch (quantizerType) {
            case QuantizerType::None:
                break;
            case QuantizerType::Undefined:
                break;
            case QuantizerType::PQQuantizer:
                switch (reconstructType) {
                    #define DefineVectorValueType(Name, Type) \
                    case VectorValueType::Name: \
                        DistanceUtils::Quantizer.reset(new PQQuantizer<Type>()); \
                        break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
                }
                
                return DistanceUtils::Quantizer->LoadQuantizer(p_in);
            }
            return ErrorCode::Success;
        }
    }
}