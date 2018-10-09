#include "inc/Aggregator/AggregatorSettings.h"

using namespace SpaceV;
using namespace SpaceV::Aggregator;

AggregatorSettings::AggregatorSettings()
    : m_searchTimeout(100),
      m_threadNum(8),
      m_socketThreadNum(8)
{
}
