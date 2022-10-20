// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/VectorSetReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common.h"
#include "inc/Helper/SimpleIniReader.h"
#include <inc/Core/Common/DistanceUtils.h>
#include "inc/Quantizer/Training.h"
#include <chrono>
#include <iostream>
#include "inc/Core/Common/IQuantizer.h"

#include <memory>

using namespace SPTAG;

class DistBenchOptions : public Helper::ReaderOptions
{
public:
    DistBenchOptions(bool enableADC, std::string quantizerFile, SizeType numLoops) : Helper::ReaderOptions(VectorValueType::Float, 0, VectorFileType::TXT, "|", 32), m_enableADC(enableADC), m_quantizerFile(quantizerFile), m_numLoops(numLoops)
    {
        AddRequiredOption(m_inputQueryFile, "-q", "--queries", "Input query data.");
        AddRequiredOption(m_inputDBFile, "-db", "--dbvecs", "Input database data.");
        
        AddRequiredOption(m_inputMapping, "-qm", "--querymapping", "Map which DB vectors to compare for each query.");
        AddRequiredOption(m_DBVecPerQuery, "-nc", "--numcompare", "Number of DB vectors to compare for each query.");
        
        AddOptionalOption(m_quantizerFile, "-qf", "--quantizer", "Input quantizer file.");
        AddOptionalOption(m_enableADC, "-adc", "--enableadc", "Enable ADC.");
        AddOptionalOption(m_numLoops, "-nl", "--numloops", "Number of loops to run for");
        
    }

    ~DistBenchOptions() {}

    std::string m_inputQueryFile;
    
    std::string m_inputDBFile;

    std::string m_inputMapping;

    std::string m_quantizerFile;

    bool m_enableADC;
    
    SizeType m_DBVecPerQuery;

    SizeType m_numLoops;
};

template <typename T>
void RunOneSearchLoop(std::shared_ptr<VectorSet> querySet, std::shared_ptr<VectorSet> DBSet, std::shared_ptr<VectorSet> DBMapping, std::function<float (const T*, const T*, SPTAG::DimensionType)> dist)
{
    auto dim = DBSet->Dimension();
    for (int i = 0; i < querySet->Count(); i++)
    {
        std::uint32_t* DBMap = (std::uint32_t *) DBMapping->GetVector(i);
        for (int j = 0; j < DBMapping->Dimension(); j++)
        {
            dist((const T*) querySet->GetVector(i), (const T*) DBSet->GetVector(DBMap[j]), dim);
        }
    }
}


int main(int argc, char* argv[])
{
    std::shared_ptr<DistBenchOptions> options = std::make_shared<DistBenchOptions>(false, std::string(), 1);

    if (!options->Parse(argc - 1, argv + 1))
    {
        LOG(Helper::LogLevel::LL_Error, "Failed to parse input file.\n");
        exit(1);
    }
    auto queryReader = Helper::VectorSetReader::CreateInstance(options);
    auto DBReader = Helper::VectorSetReader::CreateInstance(options);
    
    auto mappingOptions = std::make_shared<Helper::ReaderOptions>(VectorValueType::Float, options->m_DBVecPerQuery, VectorFileType::DEFAULT);
    auto mappingReader = Helper::VectorSetReader::CreateInstance(mappingOptions);
    
    if (ErrorCode::Success != queryReader->LoadFile(options->m_inputQueryFile))
    {
        LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
        exit(1);
    }

    if (ErrorCode::Success != DBReader->LoadFile(options->m_inputDBFile))
    {
        LOG(Helper::LogLevel::LL_Error, "Failed to read DB file.\n");
        exit(1);
    }

    if (ErrorCode::Success != mappingReader->LoadFile(options->m_inputMapping))
    {
        LOG(Helper::LogLevel::LL_Error, "Failed to read DB file.\n");
        exit(1);
    }

    std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer = nullptr;
    
    if (!options->m_quantizerFile.empty())
    {
        auto quantizerHandle = f_createIO();
        if (!quantizerHandle || !quantizerHandle->Initialize(options->m_quantizerFile.c_str(), std::ios::in | std::ios::binary))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to open quantizer file.\n");
            exit(1);
        }
        
        quantizer = SPTAG::COMMON::IQuantizer::LoadIQuantizer(quantizerHandle);
    }

    auto DBset_raw = DBReader->GetVectorSet();
    std::shared_ptr<VectorSet> DBSet = DBset_raw;
    if (quantizer)
    {
        ByteArray DB_vector_array = ByteArray::Alloc(quantizer->QuantizeSize() * DBSet->Count());
        DBSet = std::make_shared<BasicVectorSet>(DB_vector_array, VectorValueType::UInt8, quantizer->QuantizeSize(), DBSet->Count());
        
        for (int i = 0; i < DBSet->Count(); i++)
        {
            quantizer->QuantizeVector(DBset_raw->GetVector((std::uint32_t)i), (std::uint8_t*)DBSet->GetVector((std::uint32_t)i));
        }
    }

    auto queryset_raw = queryReader->GetVectorSet();
    std::shared_ptr<VectorSet> querySet = queryset_raw;
    if (quantizer)
    {
        quantizer->SetEnableADC(options->m_enableADC);
        ByteArray query_vector_array = ByteArray::Alloc(quantizer->QuantizeSize() * querySet->Count());
        querySet = std::make_shared<BasicVectorSet>(query_vector_array, VectorValueType::UInt8, quantizer->QuantizeSize(), querySet->Count());

        for (int i = 0; i < querySet->Count(); i++)
        {
            quantizer->QuantizeVector(queryset_raw->GetVector((std::uint32_t)i), (std::uint8_t*)querySet->GetVector((std::uint32_t)i));
        }
    }
    
    auto DBMapping = mappingReader->GetVectorSet();

    auto distMethod = SPTAG::DistCalcMethod::L2;
    
    auto before = std::chrono::high_resolution_clock::now();
    if (quantizer)
    {
        auto dist = quantizer->DistanceCalcSelector<std::uint8_t>(distMethod);
        for (int i = 0; i < options->m_numLoops; i++)
        {
            RunOneSearchLoop<std::uint8_t>(querySet, DBSet, DBMapping, dist);
        }
    }
    else
    {
        auto dist = SPTAG::COMMON::DistanceCalcSelector<std::int8_t>(distMethod);
        for (int i = 0; i < options->m_numLoops; i++)
        {

            RunOneSearchLoop<std::int8_t>(querySet, DBSet, DBMapping, dist);
        }
    }
    auto after = std::chrono::high_resolution_clock::now();

    auto duration = after - before;
    std::cout << "Finished " << options->m_numLoops << " iterations in " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << " ms" << std::endl;
    
    auto perCompare = duration / (options->m_numLoops * options->m_DBVecPerQuery);
    std::cout << "Average time per comparison = " << std::chrono::duration_cast<std::chrono::nanoseconds>(perCompare).count() << " ns" << std::endl;

    

}