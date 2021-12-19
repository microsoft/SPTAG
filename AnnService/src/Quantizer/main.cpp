// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/VectorSetReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common.h"
#include "inc/Helper/SimpleIniReader.h"

#include <memory>

using namespace SPTAG;

class QuantizerOptions : public Helper::ReaderOptions
{
public:
    QuantizerOptions() : Helper::ReaderOptions(VectorValueType::Float, 0, VectorFileType::TXT, "|", 32)
    {
        AddRequiredOption(m_inputFiles, "-i", "--input", "Input raw data.");
        AddRequiredOption(m_outputFile, "-o", "--output", "Output quantized vectors");
        AddRequiredOption(m_outputQuantizerFile, "-oq", "--outputquantizer", "Output quantizer");
        AddRequiredOption(m_quantizerType, "-qt", "--quantizer", "Quantizer type.");
    }

    ~QuantizerOptions() {}

    std::string m_inputFiles;

    std::string m_outputFile;

    SPTAG::Quantizer m_indexAlgoType;

    std::string m_builderConfigFile;
};

int main(int argc, char* argv[])
{

}