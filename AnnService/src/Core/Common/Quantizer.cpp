#include <inc/Core/Common/Quantizer.h>
#include <inc/Core/Common/PQQuantizer.h>

namespace SPTAG
{
    namespace COMMON
    {
        void Quantizer::LoadQuantizer(std::string path, QuantizerType quantizerType) {
            switch (quantizerType) {
            case QuantizerType::None:
                break;
            case QuantizerType::Undefined:
                break;
            case QuantizerType::PQQuantizer:
                PQQuantizer::LoadQuantizer(path);
            }
        }
    }
}