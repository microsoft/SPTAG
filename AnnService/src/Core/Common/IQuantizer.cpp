#include <inc/Core/Common/IQuantizer.h>
#include <inc/Core/Common/PQQuantizer.h>

namespace SPTAG
{
    namespace COMMON
    {
        ErrorCode IQuantizer::LoadIQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_in) {
            QuantizerType quantizerType = QuantizerType::Undefined;
            VectorValueType reconstructType = VectorValueType::Undefined;
            IOBINARY(p_in, ReadBinary, sizeof(QuantizerType), (char*)&quantizerType);
            IOBINARY(p_in, ReadBinary, sizeof(VectorValueType), (char*)&reconstructType);
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