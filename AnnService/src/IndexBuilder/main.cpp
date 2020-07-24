// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/VectorSetReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common.h"
#include "inc/Helper/SimpleIniReader.h"

#include <memory>
#include <iostream>

using namespace SPTAG;

class BuilderOptions : public Helper::ReaderOptions
{
public:
    BuilderOptions() : Helper::ReaderOptions(VectorValueType::Float, 0, VectorFileType::TXT, "|", 32)
    {
        AddRequiredOption(m_inputFiles, "-i", "--input", "Input raw data.");
        AddRequiredOption(m_outputFolder, "-o", "--outputfolder", "Output folder.");
        AddRequiredOption(m_indexAlgoType, "-a", "--algo", "Index Algorithm type.");
        AddOptionalOption(m_builderConfigFile, "-c", "--config", "Config file for builder.");
    }

    ~BuilderOptions() {}

    std::string m_inputFiles;

    std::string m_outputFolder;

    SPTAG::IndexAlgoType m_indexAlgoType;

    std::string m_builderConfigFile;
};

int main(int argc, char* argv[])
{
    std::shared_ptr<BuilderOptions> options(new BuilderOptions);
    if (!options->Parse(argc - 1, argv + 1))
    {
        exit(1);
    }

    auto indexBuilder = VectorIndex::CreateInstance(options->m_indexAlgoType, options->m_inputValueType);

    Helper::IniReader iniReader;
    if (!options->m_builderConfigFile.empty())
    {
        iniReader.LoadIniFile(options->m_builderConfigFile);
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
        std::cout << "Set [" << sectionName << "]" << paramName << " = " << paramVal << std::endl;
    }

    if (!iniReader.DoesParameterExist("Index", "NumberOfThreads")) {
        iniReader.SetParameter("Index", "NumberOfThreads", std::to_string(options->m_threadNum));
    }
    for (const auto& iter : iniReader.GetParameters("Index"))
    {
        indexBuilder->SetParameter(iter.first.c_str(), iter.second.c_str());
    }

    auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
    if (ErrorCode::Success != vectorReader->LoadFile(options->m_inputFiles))
    {
        fprintf(stderr, "Failed to read input file.\n");
        exit(1);
    }
    ErrorCode code = indexBuilder->BuildIndex(vectorReader->GetVectorSet(), vectorReader->GetMetadataSet());
    indexBuilder->SaveIndex(options->m_outputFolder);

    if (ErrorCode::Success != code)
    {
        fprintf(stderr, "Failed to build index.\n");
        exit(1);
    }
    return 0;
}
