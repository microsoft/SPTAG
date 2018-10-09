#include "inc/IndexBuilder/ThreadPool.h"
#include "inc/IndexBuilder/Options.h"
#include "inc/IndexBuilder/VectorSetReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/SimpleIniReader.h"

#include <memory>
#include <iostream>

using namespace SPTAG;

int main(int argc, char* argv[])
{
    std::shared_ptr<IndexBuilder::BuilderOptions> options(new IndexBuilder::BuilderOptions);
    if (!options->Parse(argc - 1, argv + 1))
    {
        exit(1);
    }

    IndexBuilder::ThreadPool::Init(options->m_threadNum);

    auto vectorReader = IndexBuilder::VectorSetReader::CreateInstance(options);
    if (ErrorCode::Success != vectorReader->LoadFile(options->m_inputFiles))
    {
        fprintf(stderr, "Failed to read input file.\n");
        exit(1);
    }

    auto indexBuilder = VectorIndex::CreateInstance(options->m_indexAlgoType, options->m_inputValueType);
    if (!options->m_builderConfigFile.empty())
    {
        Helper::IniReader iniReader;
        iniReader.LoadIniFile(options->m_builderConfigFile);

        for (const auto& iter : iniReader.GetParameters("Index"))
        {
            indexBuilder->SetParameter(iter.first.c_str(), iter.second.c_str());
        }
    }

    if (ErrorCode::Success != indexBuilder->BuildIndex(vectorReader->GetVectorSet(),
                                                       vectorReader->GetMetadataSet()))
    {
        fprintf(stderr, "Failed to build index.\n");
        exit(1);
    }

    indexBuilder->SaveIndex(options->m_outputFolder);

    return 0;
}
