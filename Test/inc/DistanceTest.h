#ifndef _SPTAG_TEST_DISTANCETEST_H_
#define _SPTAG_TEST_DISTANCETEST_H_

#include <boost/test/included/unit_test.hpp>
#include "inc/Core/BKT/DistanceUtils.h"

BOOST_AUTO_TEST_SUITE(DistanceTest)

template<typename T>
static float ComputeCosineDistance(const T *pX, const T *pY, int length) {
    float diff = 0;
    const T* pEnd1 = pX + length;
    while (pX < pEnd1) diff += (*pX++) * (*pY++);
    return diff;
}

template<typename T>
static float ComputeL2Distance(const T *pX, const T *pY, int length)
{
    float diff = 0;
    const T* pEnd1 = pX + length;
    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

template<typename T>
T random(int high = RAND_MAX, int low = 0)   // Generates a random value.
{
    return (T)(low + float(high - low)*(std::rand() / static_cast<float>(RAND_MAX)));
}

template<typename T>
void test(int high) {
    int dimension = random<int>(256, 2);
    T *X = new T[dimension], *Y = new T[dimension];
    BOOST_ASSERT(X != nullptr && Y != nullptr);
    for (int i = 0; i < dimension; i++) {
        X[i] = random<T>(high, -high);
        Y[i] = random<T>(high, -high);
    }

    BOOST_CHECK_CLOSE_FRACTION(ComputeL2Distance(X, Y, dimension), SPTAG::BKT::DistanceUtils::ComputeL2Distance(X, Y, dimension), 1e-6);
    BOOST_CHECK_CLOSE_FRACTION(high*high - ComputeCosineDistance(X, Y, dimension), SPTAG::BKT::DistanceUtils::ComputeCosineDistance(X, Y, dimension), 1e-6);

    delete[] X;
    delete[] Y;
}

BOOST_AUTO_TEST_CASE(TestDistanceComputation)
{
    test<float>(1);
    test<std::int8_t>(127);
    test<std::int16_t>(32767);
}

BOOST_AUTO_TEST_SUITE_END()
#endif // _SPTAG_TEST_DISTANCETEST_H_