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
    BuilderOptions()
        : Helper::ReaderOptions(VectorValueType::Float, 0, VectorFileType::TXT, "|", 32)
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

    std::string quantizerFile = options->m_quantizerFile;
    VectorValueType builderValueType = options->m_inputValueType;
    if (options->m_inputFiles.empty() &&
        iniReader.DoesParameterExist("Base", "ValueType"))
    {
        builderValueType =
            iniReader.GetParameter("Base", "ValueType", VectorValueType::Undefined);
        if (builderValueType == VectorValueType::Undefined)
        {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error, "Invalid [Base] ValueType.\n");
            return 1;
        }
    }

    const bool autoTuneEnabled =
        iniReader.DoesSectionExist("RaBitQAutoTune") &&
        iniReader.GetParameter("RaBitQAutoTune", "isExecute", false);
    if (autoTuneEnabled)
    {
#ifdef RABITQ
        if (options->m_indexAlgoType != IndexAlgoType::SPANN)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "RaBitQ auto-tuning is supported only for SPANN index construction.\n");
            return 1;
        }
        if (!options->m_inputFiles.empty())
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "RaBitQ auto-tuning reads vectors from the INI; --input is not allowed.\n");
            return 1;
        }

        COMMON::RaBitQAutoTuneResult tuneResult;
        std::string tuneError;
        ErrorCode tuneStatus = ErrorCode::Fail;
        try
        {
            tuneStatus =
                COMMON::RaBitQAutoTuner::Run(
                    iniReader, options->m_outputFolder, tuneResult, tuneError);
        }
        catch (const std::exception& exception)
        {
            tuneError = exception.what();
        }
        if (tuneStatus != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "RaBitQ auto-tuning failed: %s\n", tuneError.c_str());
            return 1;
        }

        quantizerFile = tuneResult.quantizerPath;
        iniReader.SetParameter("Base", "VectorSize", std::to_string(tuneResult.vectorCount));
        iniReader.SetParameter("Base", "QuantizerFilePath", tuneResult.quantizerPath);
        iniReader.SetParameter("Base", "QuantizedVectorPath", tuneResult.vectorPath);
        iniReader.SetParameter("BuildSSDIndex", "EnableADC", "true");
#else
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "[RaBitQAutoTune] isExecute=true requires a build configured with RABITQ=ON.\n");
        return 1;
#endif
    }

    if (quantizerFile.empty() &&
        iniReader.DoesParameterExist("Base", "QuantizerFilePath"))
    {
        quantizerFile =
            iniReader.GetParameter("Base", "QuantizerFilePath", std::string());
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set QuantizerFile = %s\n", quantizerFile.c_str());

    auto indexBuilder = VectorIndex::CreateInstance(options->m_indexAlgoType, builderValueType);
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
        if (!indexBuilder->m_pQuantizer->QuantizeForIndexBuild())
        {
            const auto reconstructType = indexBuilder->m_pQuantizer->GetReconstructType();
            if (builderValueType != reconstructType)
            {
                if (!options->m_inputFiles.empty())
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "This quantizer requires raw reconstruct vectors for SPANN index build. "
                        "Set the input vector type to %s and keep pre-quantized codes in QuantizedVectorPath.\n",
                        Helper::Convert::ConvertToString<VectorValueType>(reconstructType).c_str());
                    return 1;
                }
                if (iniReader.DoesParameterExist("Base", "VectorPath") &&
                    !iniReader.DoesParameterExist("Base", "QuantizedVectorPath"))
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "This quantizer requires [Base] VectorPath to point to raw reconstruct vectors "
                        "and [Base] QuantizedVectorPath to point to pre-quantized codes.\n");
                    return 1;
                }

                builderValueType = reconstructType;
                iniReader.SetParameter(
                    "Base", "ValueType",
                    Helper::Convert::ConvertToString<VectorValueType>(builderValueType));
                indexBuilder = VectorIndex::CreateInstance(options->m_indexAlgoType, builderValueType);
                if (!indexBuilder ||
                    indexBuilder->LoadQuantizer(quantizerFile) != ErrorCode::Success ||
                    !indexBuilder->m_pQuantizer)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot recreate index builder for quantizer reconstruct type.\n");
                    return 1;
                }
            }
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
