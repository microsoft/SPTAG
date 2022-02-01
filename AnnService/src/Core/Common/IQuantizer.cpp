#include <inc/Core/Common/IQuantizer.h>
#include <inc/Core/Common/PQQuantizer.h>
#include <inc/Core/Common/OPQQuantizer.h>
#include <inc/Helper/StringConvert.h>

namespace SPTAG
{
    namespace COMMON
    {
        ErrorCode IQuantizer::LoadIQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_in) {
            QuantizerType quantizerType = QuantizerType::Undefined;
            VectorValueType reconstructType = VectorValueType::Undefined;
            IOBINARY(p_in, ReadBinary, sizeof(QuantizerType), (char*)&quantizerType);
            IOBINARY(p_in, ReadBinary, sizeof(VectorValueType), (char*)&reconstructType);
            LOG(Helper::LogLevel::LL_Info, "Loading quantizer of type %s with reconstructtype %s.\n", Helper::Convert::ConvertToString<QuantizerType>(quantizerType).c_str(), Helper::Convert::ConvertToString<VectorValueType>(reconstructType).c_str());
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

                default: break;
                }
                
                return DistanceUtils::Quantizer->LoadQuantizer(p_in);
            case QuantizerType::OPQQuantizer:
                switch (reconstructType) {
#define DefineVectorValueType(Name, Type) \
                    case VectorValueType::Name: \
                        DistanceUtils::Quantizer.reset(new OPQQuantizer<Type>()); \
                        break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
                default: break;
                }
                return DistanceUtils::Quantizer->LoadQuantizer(p_in);
            }
            return ErrorCode::Fail;
        }

        ErrorCode IQuantizer::LoadIQuantizer(SPTAG::ByteArray bytes)
        {
            auto raw_bytes = bytes.Data();
            QuantizerType quantizerType = *(QuantizerType*) raw_bytes;
            raw_bytes += sizeof(QuantizerType);

            VectorValueType reconstructType = *(VectorValueType*)raw_bytes;
            raw_bytes += sizeof(VectorValueType);
            LOG(Helper::LogLevel::LL_Info, "Loading quantizer of type %s with reconstructtype %s.\n", Helper::Convert::ConvertToString<QuantizerType>(quantizerType).c_str(), Helper::Convert::ConvertToString<VectorValueType>(reconstructType).c_str());

            switch (quantizerType) {
            case QuantizerType::None:
                return ErrorCode::FailedParseValue;
            case QuantizerType::Undefined:
                return ErrorCode::FailedParseValue;
            case QuantizerType::PQQuantizer:
                switch (reconstructType) {
#define DefineVectorValueType(Name, Type) \
                    case VectorValueType::Name: \
                        DistanceUtils::Quantizer.reset(new PQQuantizer<Type>()); \
                        break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
                default: break;
                }

                return DistanceUtils::Quantizer->LoadQuantizer(raw_bytes);
            case QuantizerType::OPQQuantizer:
                switch (reconstructType) {
#define DefineVectorValueType(Name, Type) \
                    case VectorValueType::Name: \
                        DistanceUtils::Quantizer.reset(new OPQQuantizer<Type>()); \
                        break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
                default: break;
                }

                return DistanceUtils::Quantizer->LoadQuantizer(raw_bytes);
            }
            return ErrorCode::Fail;
        }
    }
}
