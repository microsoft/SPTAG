#include "inc/Test.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Core/VectorIndex.h"

BOOST_AUTO_TEST_SUITE (BKTTest)

BOOST_AUTO_TEST_CASE(ParameterTest)
{
    SPTAG::Helper::IniReader reader;
    reader.SetParameter("Index", "DistCalcMethod", "Cosine");
    reader.SetParameter("Index", "BKTNumber", "2");
    reader.SetParameter("Index", "NeighborhoodSize", "16");

    auto index = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::BKT, SPTAG::VectorValueType::Float);
    
    for (const auto& iter : reader.GetParameters("Index"))
    {
        index->SetParameter(iter.first.c_str(), iter.second.c_str());
    }
    
    BOOST_CHECK(index->GetParameter("BKTNumber") == "2");
    BOOST_CHECK(index->GetParameter("NeighborhoodSize") == "16");
}

BOOST_AUTO_TEST_SUITE_END()