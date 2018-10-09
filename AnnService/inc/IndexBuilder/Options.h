#ifndef _SPACEV_INDEXBUILDER_OPTIONS_H_
#define _SPACEV_INDEXBUILDER_OPTIONS_H_

#include "inc/Core/Common.h"
#include "inc/Helper/ArgumentsParser.h"

#include <string>
#include <vector>
#include <memory>

namespace SpaceV
{
namespace IndexBuilder
{

class BuilderOptions : public Helper::ArgumentsParser
{
public:
    BuilderOptions();

    ~BuilderOptions();

    std::uint32_t m_threadNum;

    std::uint32_t m_dimension;

    std::string m_vectorDelimiter;

    SpaceV::VectorValueType m_inputValueType;

    std::string m_inputFiles;

    std::string m_outputFolder;

    SpaceV::IndexAlgoType m_indexAlgoType;

    std::string m_builderConfigFile;
};


} // namespace IndexBuilder
} // namespace SpaceV

#endif // _SPACEV_INDEXBUILDER_OPTIONS_H_
