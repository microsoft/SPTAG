// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// #include <bitset>
#include <vector>
#include "inc/Test.h"
#include "inc/Core/Common/SIMDUtils.h"


template<typename T>
static void ComputeSum(T *pX, const T *pY, SPTAG::DimensionType length)
{
    const T* pEnd1 = pX + length;
    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

template<typename T>
T random(int high = RAND_MAX, int low = 0)   // Generates a random value.
{
    return (T)(low + float(high - low)*(std::rand()/static_cast<float>(RAND_MAX + 1.0)));
}

template<typename T>
void test(int high) {
    SPTAG::DimensionType dimension = random<SPTAG::DimensionType>(256, 2);
    T *X = new T[dimension], *Y = new T[dimension];
    BOOST_ASSERT(X != nullptr && Y != nullptr);
    for (SPTAG::DimensionType i = 0; i < dimension; i++) {
        X[i] = random<T>(high, -high);
        Y[i] = random<T>(high, -high);
    }
    T *X_copy = new T[dimension];
    for (SPTAG::DimensionType i = 0; i < dimension; i++) {
        X_copy[i] = X[i];
    }
    ComputeSum(X, Y, dimension);
    SPTAG::COMMON::SIMDUtils::ComputeSum(X_copy, Y, dimension);
    for (SPTAG::DimensionType i = 0; i < dimension; i++) {
        BOOST_CHECK_CLOSE_FRACTION(double(X[i]), double(X_copy[i]), 1e-5);
    }

    delete[] X;
    delete[] Y;
    delete[] X_copy;
}

BOOST_AUTO_TEST_SUITE(SIMDTest)

BOOST_AUTO_TEST_CASE(TestDistanceComputation)
{
    test<float>(1);
    test<std::int8_t>(127);
    test<std::int16_t>(32767);
}


BOOST_AUTO_TEST_SUITE_END()
