// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/IndexBuilder/VectorSetReader.h"
#include "inc/IndexBuilder/VectorSetReaders/DefaultReader.h"


using namespace SPTAG;
using namespace SPTAG::IndexBuilder;

VectorSetReader::VectorSetReader(std::shared_ptr<BuilderOptions> p_options)
    : m_options(p_options)
{
}


VectorSetReader::VectorSetReader(VectorValueType p_valueType, DimensionType p_dimension, std::string p_vectorDelimiter, std::uint32_t p_threadNum)
    : m_options(new SPTAG::IndexBuilder::BuilderOptions)
{
    m_options->m_threadNum = p_threadNum;
    m_options->m_dimension = p_dimension;
    m_options->m_vectorDelimiter = p_vectorDelimiter;
    m_options->m_inputValueType = p_valueType;
}


VectorSetReader:: ~VectorSetReader()
{
}


std::shared_ptr<VectorSetReader>
VectorSetReader::CreateInstance(std::shared_ptr<BuilderOptions> p_options)
{
    return std::shared_ptr<VectorSetReader>(new DefaultReader(std::move(p_options)));
}


std::shared_ptr<VectorSetReader>
VectorSetReader::CreateInstance(VectorValueType p_valueType, DimensionType p_dimension, std::string p_vectorDelimiter, std::uint32_t p_threadNum)
{
    return std::shared_ptr<VectorSetReader>(new DefaultReader(p_valueType, p_dimension, p_vectorDelimiter, p_threadNum));
}

