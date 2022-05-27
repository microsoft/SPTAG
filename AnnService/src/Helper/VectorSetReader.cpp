// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/VectorSetReaders/DefaultReader.h"
#include "inc/Helper/VectorSetReaders/TxtReader.h"
#include "inc/Helper/VectorSetReaders/XvecReader.h"

using namespace SPTAG;
using namespace SPTAG::Helper;


ReaderOptions::ReaderOptions(VectorValueType p_valueType, DimensionType p_dimension, VectorFileType p_fileType, std::string p_vectorDelimiter, std::uint32_t p_threadNum, bool p_normalized)
    :  m_inputValueType(p_valueType), m_dimension(p_dimension), m_inputFileType(p_fileType), m_vectorDelimiter(p_vectorDelimiter), m_threadNum(p_threadNum), m_normalized(p_normalized)
{
    AddOptionalOption(m_threadNum, "-t", "--thread", "Thread Number.");
    AddOptionalOption(m_vectorDelimiter, "-dl", "--delimiter", "Vector delimiter.");
    AddOptionalOption(m_normalized, "-norm", "--normalized", "Vector is normalized.");
    AddRequiredOption(m_dimension, "-d", "--dimension", "Dimension of vector.");
    AddRequiredOption(m_inputValueType, "-v", "--vectortype", "Input vector data type. Default is float.");
    AddRequiredOption(m_inputFileType, "-f", "--filetype", "Input file type (DEFAULT, TXT, XVEC). Default is DEFAULT.");
}


ReaderOptions::~ReaderOptions()
{
}


VectorSetReader::VectorSetReader(std::shared_ptr<ReaderOptions> p_options)
    : m_options(p_options)
{
}


VectorSetReader:: ~VectorSetReader()
{
}


std::shared_ptr<VectorSetReader>
VectorSetReader::CreateInstance(std::shared_ptr<ReaderOptions> p_options)
{
    if (p_options->m_inputFileType == VectorFileType::DEFAULT) {
        return std::make_shared<DefaultVectorReader>(p_options);
    }
    else if (p_options->m_inputFileType == VectorFileType::TXT) {
        return std::make_shared<TxtVectorReader>(p_options);
    }
    else if (p_options->m_inputFileType == VectorFileType::XVEC) {
        return std::make_shared<XvecVectorReader>(p_options);
    }
    return nullptr;
}


