// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <bitset>
#include "inc/Test.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Core/Common/PQQuantizer.h"


BOOST_AUTO_TEST_SUITE(QuantizationTest)

BOOST_AUTO_TEST_CASE(TestReadTextAndDefault)
{
    SPTAG::DimensionType TEST_DIM = 10;
    // Two files with the same data, in the two formats
    std::string TXT_FILE = "testvectors-quantized.txt";
    std::string DEFAULT_FILE = "testvectors-quantized.bin";
    // Distances between vector 0 and vector i, in order L2, Cosine, L2, Cosine, etc.
    std::string TRUTH_FILE = "vector-distances-quantized.txt";

    SPTAG::Helper::ReaderOptions textOptions = SPTAG::Helper::ReaderOptions(SPTAG::VectorValueType::Float, TEST_DIM, SPTAG::VectorFileType::TXT, "|", 32, true);
    SPTAG::Helper::ReaderOptions defaultOptions = SPTAG::Helper::ReaderOptions(SPTAG::VectorValueType::Float, TEST_DIM, SPTAG::VectorFileType::DEFAULT, "|", 32, true);

    auto textVectorReader = SPTAG::Helper::VectorSetReader::CreateInstance(std::make_shared<SPTAG::Helper::ReaderOptions>(textOptions));
    textVectorReader->LoadFile(TXT_FILE);
    auto textVectorSet = textVectorReader->GetVectorSet();
    auto textQuantizer = SPTAG::COMMON::DistanceUtils::PQQuantizer;
    BOOST_ASSERT(textQuantizer != nullptr);

    auto defaultVectorReader = SPTAG::Helper::VectorSetReader::CreateInstance(std::make_shared<SPTAG::Helper::ReaderOptions>(defaultOptions));
    defaultVectorReader->LoadFile(DEFAULT_FILE);
    auto defaultVectorSet = defaultVectorReader->GetVectorSet();
    auto defaultQuantizer = SPTAG::COMMON::DistanceUtils::PQQuantizer;
    BOOST_ASSERT(defaultQuantizer != nullptr);

    BOOST_ASSERT(textQuantizer != defaultQuantizer);
    BOOST_ASSERT(textQuantizer->GetDimPerSubvector() == defaultQuantizer->GetDimPerSubvector());
    BOOST_ASSERT(textQuantizer->GetKsPerSubvector() == defaultQuantizer->GetKsPerSubvector());
    BOOST_ASSERT(textQuantizer->GetNumSubvectors() == defaultQuantizer->GetNumSubvectors());
    BOOST_ASSERT(textVectorSet->Count() == defaultVectorSet->Count());
    BOOST_ASSERT(textVectorSet->Dimension() == defaultVectorSet->Dimension());
    BOOST_ASSERT(textQuantizer->GetNumSubvectors() == textVectorSet->Dimension());
    BOOST_ASSERT(defaultQuantizer->GetNumSubvectors() == defaultVectorSet->Dimension());

    int M = textVectorSet->Dimension();
    int cnt = textVectorSet->Count();
    
    // check vectors match
    for (int i = 0; i < cnt; i++) {
        for (int j = 0; j < M; j++) {
            BOOST_ASSERT(((std::uint8_t*)textVectorSet->GetVector(i))[j] == ((std::uint8_t*)defaultVectorSet->GetVector(i))[j]);
        }
    }

    int Ks = textQuantizer->GetKsPerSubvector();
    int Ds = textQuantizer->GetDimPerSubvector();
    auto textCB = textQuantizer->GetCodebooks();
    auto defaultCB = defaultQuantizer->GetCodebooks();

    // check codebooks match
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < Ks; j++) {
            for (int k = 0; k < Ds; k++) {
                BOOST_CHECK_CLOSE_FRACTION(textCB[i][j][k], defaultCB[i][j][k], 1e-3);
            }
        }
    }


    std::ifstream verificationFile(TRUTH_FILE);
    // check distances match
    auto baseVector = defaultVectorSet->GetVector(0);
    for (int i = 0; i < cnt; i++) {
        std::string tmp;
        std::getline(verificationFile, tmp);
        float L2DistTarget = atof(tmp.c_str());
        std::getline(verificationFile, tmp);
        float CosineDistTarget = atof(tmp.c_str());

        auto CosineFn = SPTAG::COMMON::DistanceCalcSelector<std::uint8_t>(SPTAG::DistCalcMethod::Cosine);
        auto L2Fn = SPTAG::COMMON::DistanceCalcSelector<std::uint8_t>(SPTAG::DistCalcMethod::L2);

        float CosineDistReal = CosineFn((std::uint8_t*)baseVector, (std::uint8_t*)defaultVectorSet->GetVector(i), M);
        float L2DistReal = L2Fn((std::uint8_t*)baseVector, (std::uint8_t*)defaultVectorSet->GetVector(i), M);

        BOOST_CHECK_CLOSE_FRACTION(CosineDistTarget, CosineDistReal, 1e-3);
        BOOST_CHECK_CLOSE_FRACTION(L2DistTarget, L2DistReal, 1e-3);
    }

}



BOOST_AUTO_TEST_SUITE_END()