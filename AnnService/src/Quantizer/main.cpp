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

void QuantizeAndSave(std::shared_ptr<SPTAG::Helper::VectorSetReader>& vectorReader, std::shared_ptr<QuantizerOptions>& options, std::shared_ptr<SPTAG::COMMON::IQuantizer>& quantizer)
{
    std::shared_ptr<SPTAG::VectorSet> set;
    for (int i = 0; (set = vectorReader->GetVectorSet(i, i + options->m_trainingSamples))->Count() > 0; i += options->m_trainingSamples)
    {
        if (i % (options->m_trainingSamples *10) == 0 || i % options->m_trainingSamples != 0)
        {
            LOG(Helper::LogLevel::LL_Info, "Saving vector batch starting at %d\n", i);
        }
        std::shared_ptr<VectorSet> quantized_vectors;
        if (options->m_normalized)
        {
            LOG(Helper::LogLevel::LL_Info, "Normalizing vectors.\n");
            set->Normalize(options->m_threadNum);
        }
        ByteArray PQ_vector_array = ByteArray::Alloc(sizeof(std::uint8_t) * options->m_quantizedDim * set->Count());
        quantized_vectors = std::make_shared<BasicVectorSet>(PQ_vector_array, VectorValueType::UInt8, options->m_quantizedDim, set->Count());

#pragma omp parallel for
        for (int i = 0; i < set->Count(); i++)
        {
            quantizer->QuantizeVector(set->GetVector(i), (uint8_t*)quantized_vectors->GetVector(i));
        }

        ErrorCode code;
        if ((code = quantized_vectors->AppendSave(options->m_outputFile)) != ErrorCode::Success)
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to save quantized vectors, ErrorCode: %s.\n", SPTAG::Helper::Convert::ConvertToString(code).c_str());
            exit(1);
        }
        if (!options->m_outputFullVecFile.empty())
        {
            if (ErrorCode::Success != set->AppendSave(options->m_outputFullVecFile))
            {
                LOG(Helper::LogLevel::LL_Error, "Failed to save uncompressed vectors.\n");
                exit(1);
            }
        }
        if (!options->m_outputReconstructVecFile.empty())
        {
#pragma omp parallel for
            for (int i = 0; i < set->Count(); i++)
            {
                quantizer->ReconstructVector((uint8_t*)quantized_vectors->GetVector(i), set->GetVector(i));
            }
            if (ErrorCode::Success != set->AppendSave(options->m_outputReconstructVecFile))
            {
                LOG(Helper::LogLevel::LL_Error, "Failed to save uncompressed vectors.\n");
                exit(1);
            }
        }
    }
}

int main(int argc, char* argv[])
{
    std::shared_ptr<QuantizerOptions> options = std::make_shared<QuantizerOptions>(10000, true, 0.0, SPTAG::QuantizerType::None, std::string(), -1, std::string(), std::string());

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
        std::shared_ptr<SPTAG::VectorSet> set;
        for (int i = 0; (set = vectorReader->GetVectorSet(i, i + options->m_trainingSamples))->Count() > 0; i += options-> m_trainingSamples)
        {
            set->AppendSave(options->m_outputFile);
        }
        
        if (!options->m_outputMetadataFile.empty() && !options->m_outputMetadataIndexFile.empty())
        {
            auto metadataSet = vectorReader->GetMetadataSet();
            if (metadataSet)
            {
                metadataSet->SaveMetadata(options->m_outputMetadataFile, options->m_outputMetadataIndexFile);
            }
        }

        break;
    }
    case QuantizerType::PQQuantizer:
    {
        std::shared_ptr<COMMON::IQuantizer> quantizer;
        auto fp_load = SPTAG::f_createIO();
        if (fp_load == nullptr || !fp_load->Initialize(options->m_outputQuantizerFile.c_str(), std::ios::binary | std::ios::in))
        {
            auto set = vectorReader->GetVectorSet(0, options->m_trainingSamples);
            ByteArray PQ_vector_array = ByteArray::Alloc(sizeof(std::uint8_t) * options->m_quantizedDim * set->Count());
            std::shared_ptr<VectorSet> quantized_vectors = std::make_shared<BasicVectorSet>(PQ_vector_array, VectorValueType::UInt8, options->m_quantizedDim, set->Count());
            LOG(Helper::LogLevel::LL_Info, "Quantizer Does not exist. Training a new one.\n");

            switch (options->m_inputValueType)
            {
#define DefineVectorValueType(Name, Type) \
                    case VectorValueType::Name: \
                        quantizer.reset(new COMMON::PQQuantizer<Type>(options->m_quantizedDim, 256, (DimensionType)(options->m_dimension/options->m_quantizedDim), false, TrainPQQuantizer<Type>(options, set, quantized_vectors))); \
                        break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
            }

            auto ptr = SPTAG::f_createIO();
            if (ptr != nullptr && ptr->Initialize(options->m_outputQuantizerFile.c_str(), std::ios::binary | std::ios::out))
            {
                if (ErrorCode::Success != quantizer->SaveQuantizer(ptr))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed to write quantizer file.\n");
                    exit(1);
                }
            }
        }
        else
        {
            quantizer = SPTAG::COMMON::IQuantizer::LoadIQuantizer(fp_load);
            if (!quantizer)
            {
                LOG(Helper::LogLevel::LL_Error, "Failed to open existing quantizer file.\n");
                exit(1);
            }
            quantizer->SetEnableADC(false);
        }

        QuantizeAndSave(vectorReader, options, quantizer);

        auto metadataSet = vectorReader->GetMetadataSet();
        if (metadataSet)
        {
            metadataSet->SaveMetadata(options->m_outputMetadataFile, options->m_outputMetadataIndexFile);
        }
        
        break;
    }
    case QuantizerType::OPQQuantizer:
    {
        std::shared_ptr<COMMON::IQuantizer> quantizer;
        auto fp_load = SPTAG::f_createIO();
        if (fp_load == nullptr || !fp_load->Initialize(options->m_outputQuantizerFile.c_str(), std::ios::binary | std::ios::in))
        {
            LOG(Helper::LogLevel::LL_Info, "Quantizer Does not exist. Not supported for OPQ.\n");
            exit(1);
        }
        else
        {
            quantizer = SPTAG::COMMON::IQuantizer::LoadIQuantizer(fp_load);
            if (!quantizer)
            {
                LOG(Helper::LogLevel::LL_Error, "Failed to open existing quantizer file.\n");
                exit(1);
            }
            quantizer->SetEnableADC(false);
        }

        QuantizeAndSave(vectorReader, options, quantizer);


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