// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/VectorSetReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common.h"
#include "inc/Helper/SimpleIniReader.h"
#include <inc/Core/Common/DistanceUtils.h>
#include "inc/Quantizer/Training.h"

#include <memory>

using namespace SPTAG;

int main(int argc, char* argv[])
{
    std::shared_ptr<QuantizerOptions> options(new QuantizerOptions(10000, true, 1.0, SPTAG::QuantizerType::None, std::string(), -1));
    if (!options->Parse(argc - 1, argv + 1))
    {
        exit(1);
    }
    auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
    if (ErrorCode::Success != vectorReader->LoadFile(options->m_inputFiles))
    {
        LOG(Helper::LogLevel::LL_Error, "Failed to read input file.\n");
        exit(1);
    }
    switch (options->m_quantizerType)
    {
    case QuantizerType::None:
    {
        vectorReader->GetVectorSet()->Save(options->m_outputFile);
        auto metadataSet = vectorReader->GetMetadataSet();
        if (metadataSet)
        {
            metadataSet->SaveMetadata(options->m_outputMetadataFile, options->m_outputMetadataIndexFile);
        }
        break;
    }
    case QuantizerType::PQQuantizer:
    {
        std::shared_ptr<VectorSet> quantized_vectors;
        auto fullvectors = vectorReader->GetVectorSet();
        ByteArray PQ_vector_array = ByteArray::Alloc(sizeof(std::uint8_t) * options->m_quantizedDim * fullvectors->Count());
        quantized_vectors.reset(new BasicVectorSet(PQ_vector_array, VectorValueType::UInt8, options->m_quantizedDim, fullvectors->Count()));
        switch (options->m_inputValueType)
        {
#define DefineVectorValueType(Name, Type) \
                    case VectorValueType::Name: \
                        COMMON::DistanceUtils::Quantizer.reset(new COMMON::PQQuantizer<Type>(options->m_quantizedDim, 256, (DimensionType)(options->m_dimension/options->m_quantizedDim), false, TrainPQQuantizer<Type>(options, fullvectors, quantized_vectors))); \
                        break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
        }
        if (ErrorCode::Success != quantized_vectors->Save(options->m_outputFile))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to save quantized vectors.\n");
            exit(1);
        }

        auto ptr = SPTAG::f_createIO();
        if (ptr != nullptr && ptr->Initialize(options->m_outputQuantizerFile.c_str(), std::ios::binary | std::ios::out))
        {
            if (ErrorCode::Success != COMMON::DistanceUtils::Quantizer->SaveQuantizer(ptr))
            {
                LOG(Helper::LogLevel::LL_Error, "Failed to write quantizer file.\n");
                exit(1);
            }
        }
        else
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to open quantizer file.\n");
            exit(1);
        }

        auto metadataSet = vectorReader->GetMetadataSet();
        if (metadataSet)
        {
            metadataSet->SaveMetadata(options->m_outputMetadataFile, options->m_outputMetadataIndexFile);
        }
        
        break;
    }
    default:
    {
        LOG(Helper::LogLevel::LL_Error, "Failed to read quantizer type.\n");
        exit(1);
    }
    }
}