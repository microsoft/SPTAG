// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_INDEXBUILDER_OPTIONS_H_
#define _SPTAG_INDEXBUILDER_OPTIONS_H_

#include "inc/Core/Common.h"
#include "inc/Helper/VectorSetReader.h"

#include <string>
#include <vector>
#include <memory>

namespace SPTAG
{
namespace IndexBuilder
{

class BuilderOptions : public Helper::ReaderOptions
{
public:
    BuilderOptions() : Helper::ReaderOptions(VectorValueType::Float, 0, "|", 32)
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


} // namespace IndexBuilder
} // namespace SPTAG

#endif // _SPTAG_INDEXBUILDER_OPTIONS_H_
