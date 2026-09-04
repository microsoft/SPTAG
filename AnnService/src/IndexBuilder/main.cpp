// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/VectorSetReader.h"

#include <inc/Core/Common/DistanceUtils.h>
#include <exception>
#include <memory>

#ifdef RABITQ
#include "inc/Core/Common/RaBitQAutoTuner.h"
#endif

using namespace SPTAG;

class BuilderOptions : public Helper::ReaderOptions
{
  public:
    BuilderOptions() : Helper::ReaderOptions(VectorValueType::Float, 0, VectorFileType::TXT, "|", 32)
    {
        AddRequiredOption(m_outputFolder, "-o", "--outputfolder", "Output folder.");
        AddRequiredOption(m_indexAlgoType, "-a", "--algo", "Index Algorithm type.");
        AddOptionalOption(m_inputFiles, "-i", "--input", "Input raw data.");
        AddOptionalOption(m_builderConfigFile, "-c", "--config", "Config file for builder.");
        AddOptionalOption(m_quantizerFile, "-pq", "--quantizer", "Quantizer File");
        AddOptionalOption(m_metaMapping, "-m", "--metaindex", "Enable delete vectors through metadata");
    }

    ~BuilderOptions()
    {
    }

    std::string m_inputFiles;

    std::string m_outputFolder;

    SPTAG::IndexAlgoType m_indexAlgoType;

    std::string m_builderConfigFile;

    std::string m_quantizerFile;

    bool m_metaMapping = false;
};

int main(int argc, char *argv[])
{
    std::shared_ptr<BuilderOptions> options(new BuilderOptions);
    if (!options->Parse(argc - 1, argv + 1))
    {
        exit(1);
    }

    Helper::IniReader iniReader;
    if (!options->m_builderConfigFile.empty() &&
        iniReader.LoadIniFile(options->m_builderConfigFile) != ErrorCode::Success)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open index configure file!\n");
        return -1;
    }

    for (int i = 1; i < argc; i++)
    {
        std::string param(argv[i]);
        size_t idx = param.find("=");
        if (idx == std::string::npos)
            continue;

        std::string paramName = param.substr(0, idx);
        std::string paramVal = param.substr(idx + 1);
        std::string sectionName;
        idx = paramName.find(".");
        if (idx != std::string::npos)
        {
            sectionName = paramName.substr(0, idx);
            paramName = paramName.substr(idx + 1);
        }
        iniReader.SetParameter(sectionName, paramName, paramVal);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set [%s]%s = %s\n", sectionName.c_str(), paramName.c_str(),
                     paramVal.c_str());
    }

    if (options->m_inputFiles.empty() &&
        iniReader.DoesParameterExist("Base", "ValueType"))
    {
        options->m_inputValueType =
            iniReader.GetParameter("Base", "ValueType", VectorValueType::Undefined);
        if (options->m_inputValueType == VectorValueType::Undefined)
        {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error, "Invalid [Base] ValueType.\n");
            return 1;
        }
    }

    std::string quantizerFile = options->m_quantizerFile;
    if (quantizerFile.empty() && iniReader.GetParameter("RaBitQAutoTune", "isExecute", false))
    {
#ifdef RABITQ
        auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
        if (ErrorCode::Success != vectorReader->LoadFile(options->m_inputFiles.empty()? iniReader.GetParameter("Base", "VectorPath", "") : options->m_inputFiles))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read input file.\n");
            exit(1);
        }
        vecset = vectorReader->GetVectorSet(0, iniReader.GetParameter("RaBitQAutoTune", "TrainSamples", 10000));
        
        DistCalcMethod distCalcMethod = iniReader.GetParameter("Index", "DistCalcMethod", DistCalcMethod::Undefined);
        if (distCalcMethod == DistCalcMethod::Undefined)
        {
            distCalcMethod = iniReader.GetParameter("Base", "DistCalcMethod", DistCalcMethod::Undefined);
        }
        if (distCalcMethod == DistCalcMethod::Undefined)
        {
            distCalcMethod = DistCalcMethod::L2;
        }
        COMMON::RaBitQAutoTuneResult tuneResult;

        ErrorCode ret = COMMON::RaBitQAutoTuner::Run(
                    vecset, iniReader.GetParameter("RaBitQAutoTune", "TestQueries", 100), 
                    iniReader.GetParameter("RaBitQAutoTune", "ResultCount", 10), options->m_threads, 
                    iniReader.GetParameter("RaBitQAutoTune", "TargetRecall", 0.9F), distCalcMethod, ".", tuneResult);
        
        if (ret != ErrorCode::Success || !tuneResult.quantizer)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "RaBitQ auto-tuning failed.\n");
            return 1;
        }

        quantizerFile = tuneResult.quantizerPath;
        std::string newVectorPath;
        if (options->m_inputFiles.empty()) {
            newVectorPath = iniReader.GetParameter("Base", "VectorPath", "vectors.bin") + ".quan";         
            iniReader.SetParameter("Base", "VectorPath", newVectorPath);
            iniReader.SetParameter("Base", "VectorType", "Default");
            iniReader.SetParameter("Base", "ValueType", "UInt8");
            iniReader.SetParameter("Base", "Dim", "" + std::to_string(tuneResult.codeDimension));
        } else {
            auto oldVectorPath = SPTAG::Helper::StrUtils::SplitString(options->m_inputFiles, ",")[0];
            newVectorPath = oldVectorPath + ".quan";
            options->m_inputFiles = options->m_inputFiles.replace(0, oldVectorPath.length(), newVectorPath);
        }

        options->m_inputValueType = VectorValueType::UInt8;
        options->m_dim = tuneResult.codeDimension;
        options->m_dimension = tuneResult.codeDimension;
        {
            SizeType written = 0;
            auto vectorOutput = f_createIO();
            if (!vectorOutput ||
                !vectorOutput->Initialize(
                    newVectorPath.c_str(), std::ios::out | std::ios::binary) ||
                vectorOutput->WriteBinary(sizeof(written), reinterpret_cast<const char*>(&written)) !=
                    sizeof(written) ||
                vectorOutput->WriteBinary(
                    sizeof(tuneResult.codeDimension), reinterpret_cast<const char*>(&(tuneResult.codeDimension))) !=
                    sizeof(tuneResult.codeDimension)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to create encoded vector file: %s\n", newVectorPath.c_str());
                return 1;
            }
            p_quantizer->SetEnableADC(true);
            std::vector<std::uint8_t> code(static_cast<std::size_t>(tuneResult.codeDimension));
            SizeType kBatchSize = iniReader.GetParameter("RaBitQAutoTune", "BatchSize", (SizeType)1000000);
            for (SizeType start = 0; ; start += kBatchSize) {
                const auto batch = p_reader->GetVectorSet(start, start + kBatchSize);
                if (!batch) break;

                for (SizeType i = 0; i < batch->Count(); ++i) {
                    p_quantizer->QuantizeVector(batch->GetVector(i), code.data(), false);
                    if (vectorOutput->WriteBinary(
                            code.size(), reinterpret_cast<const char*>(code.data())) != code.size()) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write encoded vector file: %s\n", newVectorPath.c_str());
                        exit(1);
                    }
                }
                written += batch->Count();
                if (batch->Count() < kBatchSize) break;
            }
            vectorOutput->WriteBinary(sizeof(written), reinterpret_cast<const char*>(&written), 0);
        }

#else
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "[RaBitQAutoTune] isExecute=true requires a build configured with RABITQ=ON.\n");
        return 1;
#endif
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set QuantizerFile = %s\n", quantizerFile.c_str());

    auto indexBuilder = VectorIndex::CreateInstance(options->m_indexAlgoType, options->m_inputValueType);
    if (!indexBuilder)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot create index builder.\n");
        return 1;
    }
    if (!quantizerFile.empty())
    {
        if (indexBuilder->LoadQuantizer(quantizerFile) != ErrorCode::Success ||
            !indexBuilder->m_pQuantizer)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot load quantizer file.\n");
            return 1;
        }
    }

    std::string sections[] = {"Base", "SelectHead", "BuildHead", "BuildSSDIndex", "Index"};
    for (int i = 0; i < 5; i++)
    {
        if (!iniReader.DoesParameterExist(sections[i], "NumberOfThreads"))
        {
            iniReader.SetParameter(sections[i], "NumberOfThreads", std::to_string(options->m_threadNum));
        }
        for (const auto &iter : iniReader.GetParameters(sections[i]))
        {
            indexBuilder->SetParameter(iter.first.c_str(), iter.second.c_str(), sections[i]);
        }
    }

    ErrorCode code;
    std::shared_ptr<VectorSet> vecset;
    if (options->m_inputFiles != "")
    {
        if (options->m_dimension <= 0)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "--dimension is required when indexbuilder reads --input directly.\n");
            return 1;
        }
        auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
        if (ErrorCode::Success != vectorReader->LoadFile(options->m_inputFiles))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read input file.\n");
            exit(1);
        }
        vecset = vectorReader->GetVectorSet();
        code = indexBuilder->BuildIndex(vecset, vectorReader->GetMetadataSet(), options->m_metaMapping,
                                        options->m_normalized, true);
    }
    else
    {
        if (!quantizerFile.empty())
        {
            indexBuilder->SetQuantizerFileName(
                quantizerFile.substr(quantizerFile.find_last_of("/\\") + 1));
        }
        code = indexBuilder->BuildIndex(options->m_normalized);
    }
    if (code == ErrorCode::Success)
    {
        code = indexBuilder->SaveIndex(options->m_outputFolder);
        if (code != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to save index.\n");
            return 1;
        }
    }
    else
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to build index.\n");
        exit(1);
    }
    return 0;
}
