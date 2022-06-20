// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/VectorSetReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common.h"
#include "inc/Helper/SimpleIniReader.h"

#include <memory>
#include <inc/Core/Common/DistanceUtils.h>

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

    ~BuilderOptions() {}

    std::string m_inputFiles;

    std::string m_outputFolder;

    SPTAG::IndexAlgoType m_indexAlgoType;

    std::string m_builderConfigFile;

    std::string m_quantizerFile;

    bool m_metaMapping = false;
};

int main(int argc, char* argv[])
{
    std::shared_ptr<BuilderOptions> options(new BuilderOptions);
    if (!options->Parse(argc - 1, argv + 1))
    {
        exit(1);
    }
    LOG(Helper::LogLevel::LL_Info, "Set QuantizerFile = %s\n", options->m_quantizerFile.c_str());

    auto indexBuilder = VectorIndex::CreateInstance(options->m_indexAlgoType, options->m_inputValueType);
    if (!options->m_quantizerFile.empty())
    {
        indexBuilder->LoadQuantizer(options->m_quantizerFile);
        if (!indexBuilder->m_pQuantizer)
        {
            exit(1);
        }
    }

    Helper::IniReader iniReader;
    if (!options->m_builderConfigFile.empty() && iniReader.LoadIniFile(options->m_builderConfigFile) != ErrorCode::Success)
    {
        LOG(Helper::LogLevel::LL_Error, "Cannot open index configure file!");
        return -1;
    }

    for (int i = 1; i < argc; i++)
    {
        std::string param(argv[i]);
        size_t idx = param.find("=");
        if (idx == std::string::npos) continue;

        std::string paramName = param.substr(0, idx);
        std::string paramVal = param.substr(idx + 1);
        std::string sectionName;
        idx = paramName.find(".");
        if (idx != std::string::npos) {
            sectionName = paramName.substr(0, idx);
            paramName = paramName.substr(idx + 1);
        }
        iniReader.SetParameter(sectionName, paramName, paramVal);
        LOG(Helper::LogLevel::LL_Info, "Set [%s]%s = %s\n", sectionName.c_str(), paramName.c_str(), paramVal.c_str());
    }

    std::string sections[] = { "Base", "SelectHead", "BuildHead", "BuildSSDIndex", "Index" };
    for (int i = 0; i < 5; i++) {
        if (!iniReader.DoesParameterExist(sections[i], "NumberOfThreads")) {
            iniReader.SetParameter(sections[i], "NumberOfThreads", std::to_string(options->m_threadNum));
        }
        for (const auto& iter : iniReader.GetParameters(sections[i]))
        {
            indexBuilder->SetParameter(iter.first.c_str(), iter.second.c_str(), sections[i]);
        }
    }
    
    ErrorCode code;
    std::shared_ptr<VectorSet> vecset;
    if (options->m_inputFiles != "") {
        auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
        if (ErrorCode::Success != vectorReader->LoadFile(options->m_inputFiles))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read input file.\n");
            exit(1);
        }
        vecset = vectorReader->GetVectorSet();
        code = indexBuilder->BuildIndex(vecset, vectorReader->GetMetadataSet(), options->m_metaMapping, options->m_normalized, true);
    }
    else {
        indexBuilder->SetQuantizerFileName(options->m_quantizerFile.substr(options->m_quantizerFile.find_last_of("/\\") + 1));
        code = indexBuilder->BuildIndex(options->m_normalized);    
    }
    if (code == ErrorCode::Success)
    {
        indexBuilder->SaveIndex(options->m_outputFolder);
    }
    else
    {
        LOG(Helper::LogLevel::LL_Error, "Failed to build index.\n");
        exit(1);
    }
    return 0;
}
