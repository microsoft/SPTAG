// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <bitset>
#include <ctime>
#include <thread>
#include <vector>
#include "inc/Test.h"
#include "inc/Core/Common/DistanceUtils.h"

template<typename T>
static float ComputeCosineDistance(const T *pX, const T *pY, SPTAG::DimensionType length) {
    float diff = 0;
    const T* pEnd1 = pX + length;
    while (pX < pEnd1) diff += (*pX++) * (*pY++);
    return diff;
}

template<typename T>
static float ComputeL2Distance(const T *pX, const T *pY, SPTAG::DimensionType length)
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
    BOOST_CHECK_CLOSE_FRACTION(ComputeL2Distance(X, Y, dimension), SPTAG::COMMON::DistanceUtils::ComputeDistance(X, Y, dimension, SPTAG::DistCalcMethod::L2), 1e-5);
    BOOST_CHECK_CLOSE_FRACTION(high * high - ComputeCosineDistance(X, Y, dimension), SPTAG::COMMON::DistanceUtils::ComputeDistance(X, Y, dimension, SPTAG::DistCalcMethod::Cosine), 1e-5);

    delete[] X;
    delete[] Y;
}

template <typename T>
void test_dist_calc_performance(
    int high, 
    SPTAG::DimensionType dimension = 256,
    SPTAG::SizeType size = 100,
    SPTAG::DistCalcMethod calc_method = SPTAG::DistCalcMethod::L2)
{
    T **X = new T *[size];
    T **Y = new T *[size];
    for (SPTAG::SizeType i = 0; i < size; i++)
    {
        X[i] = new T[dimension];
        Y[i] = new T[dimension];
        for (SPTAG::DimensionType j = 0; j < dimension; j++)
        {
            X[i][j] = random<T>(high, -high);
            Y[i][j] = random<T>(high, -high);
        }
    }

    double start, end;
    start = omp_get_wtime();
#pragma omp parallel for
    for (SPTAG::SizeType i = 0; i < size; i++)
    {
        SPTAG::COMMON::DistanceUtils::ComputeDistance(X[i], Y[i], dimension, calc_method);
    }
    end = omp_get_wtime();
    std::cout << "Time to calculate distance (ms): " << (end - start) * 1000 << std::endl;

    delete[] X;
    delete[] Y;
}

BOOST_AUTO_TEST_SUITE(DistanceTest)

BOOST_AUTO_TEST_CASE(TestDistanceComputation)
{
    test<float>(1);
    test<std::int8_t>(127);
    test<std::int16_t>(32767);
}

BOOST_AUTO_TEST_CASE(TestDistanceComputationPerformance)
{
    std::vector<SPTAG::DimensionType> dimensions{128, 256, 512, 1024};
    std::vector<int> nums_threads{1, 16, 40};
    std::vector<SPTAG::DistCalcMethod> calc_methods{SPTAG::DistCalcMethod::L2, SPTAG::DistCalcMethod::Cosine};
    SPTAG::SizeType size = 100000;
    std::cout << "Testing DistanceComputationPerformance..." << std::endl;
    for (int num_threads : nums_threads)
    {
        std::cout << "num_thread: " << num_threads << std::endl;
        omp_set_num_threads(num_threads);
        for (SPTAG::DistCalcMethod calc_method : calc_methods)
        {
            std::cout << "calc_method: " << (calc_method == SPTAG::DistCalcMethod::L2 ? "L2" : "Cosine") << std::endl;
            for (auto dimension : dimensions)
            {
                std::cout << "type: int8, dimension: " << dimension << ", size: " << size << std::endl;
                test_dist_calc_performance<std::int8_t>(127, dimension, size, calc_method);
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
