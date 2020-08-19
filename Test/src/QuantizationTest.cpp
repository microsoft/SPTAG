// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <bitset>
#include "inc/Test.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Core/Common/PQQuantizer.h"


BOOST_AUTO_TEST_SUITE(QuantizationTest)

BOOST_AUTO_TEST_CASE(TestReadTextAndDefault)
{
    SPTAG::DimensionType TEST_DIM = 5;
    // Two files with the same data, in the two formats
    std::string TXT_FILE = "testvectors-quantized.txt";
    std::string DEFAULT_FILE = "testvectors-quantized.bin";
    // Codebook file
    std::string CODEBOOK_FILE = "test-quantizer.bin";
    // Distances between vector 0 and vector i, in order L2, Cosine, L2, Cosine, etc.
    std::string TRUTH_FILE = "vector-distances-quantized.txt";

    SPTAG::Helper::ReaderOptions textOptions = SPTAG::Helper::ReaderOptions(SPTAG::VectorValueType::UInt8, TEST_DIM, SPTAG::VectorFileType::TXT, "|", 1);
    SPTAG::Helper::ReaderOptions defaultOptions = SPTAG::Helper::ReaderOptions(SPTAG::VectorValueType::UInt8, TEST_DIM, SPTAG::VectorFileType::DEFAULT, "|", 1);

    std::cout << "Loading TEXT file" << std::endl;

    auto textVectorReader = SPTAG::Helper::VectorSetReader::CreateInstance(std::make_shared<SPTAG::Helper::ReaderOptions>(textOptions));
    textVectorReader->LoadFile(TXT_FILE);
    auto textVectorSet = textVectorReader->GetVectorSet();


    std::cout << "Loading DEFAULT file" << std::endl;

    auto defaultVectorReader = SPTAG::Helper::VectorSetReader::CreateInstance(std::make_shared<SPTAG::Helper::ReaderOptions>(defaultOptions));
    defaultVectorReader->LoadFile(DEFAULT_FILE);
    auto defaultVectorSet = defaultVectorReader->GetVectorSet();

    std::cout << "Loading quantizer" << std::endl;
    SPTAG::COMMON::PQQuantizer::LoadQuantizer(CODEBOOK_FILE);
    std::cout << "Quantizer loaded" << std::endl;
    auto quantizer = SPTAG::COMMON::DistanceUtils::PQQuantizer;
    BOOST_ASSERT(quantizer != nullptr);

    std::cout << "Count (TXT/DEFAULT):" << textVectorSet->Count() << "/" << defaultVectorSet->Count() << std::endl;
    BOOST_ASSERT(textVectorSet->Count() == defaultVectorSet->Count());
    std::cout << "Dimension (TXT/DEFAULT):" << textVectorSet->Dimension() << "/" << defaultVectorSet->Dimension() << std::endl;
    BOOST_ASSERT(textVectorSet->Dimension() == defaultVectorSet->Dimension());
    BOOST_ASSERT(quantizer->GetNumSubvectors() == textVectorSet->Dimension());
    BOOST_ASSERT(quantizer->GetNumSubvectors() == defaultVectorSet->Dimension());

    int M = textVectorSet->Dimension();
    int cnt = textVectorSet->Count();

    // check vectors match
    for (int i = 0; i < cnt; i++) {
        for (int j = 0; j < M; j++) {
            //std::cout << "Entry (" << i << "," << j << ") - (TEXT/DEFAULT):" << (int) (((std::uint8_t*)textVectorSet->GetVector(i))[j]) << "/" << (int) (((std::uint8_t*)defaultVectorSet->GetVector(i))[j]) << std::endl;
            BOOST_ASSERT(((std::uint8_t*)textVectorSet->GetVector(i))[j] == ((std::uint8_t*)defaultVectorSet->GetVector(i))[j]);
        }
    }

    int Ks = quantizer->GetKsPerSubvector();
    int Ds = quantizer->GetDimPerSubvector();
    auto CB = quantizer->GetCodebooks();
    std::cout << "First entry of codebook:" << CB[0][0][0] << std::endl;
    std::cout << "Second entry of codebook:" << CB[0][0][1] << std::endl;

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
    std::cout << "Quantization Test complete" << std::endl;
    SPTAG::COMMON::DistanceUtils::PQQuantizer = nullptr;
}



BOOST_AUTO_TEST_SUITE_END()
