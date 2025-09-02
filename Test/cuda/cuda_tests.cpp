// #include "test_kernels.cu"

#include <chrono>
#include <cstdlib>

#include "inc/Test.h"
#include <boost/filesystem.hpp>
#include <iostream>

int GPUBuildKNNTest();

TEST(GPUTest, RandomTests)
{
    EXPECT_EQ(1, 1);

    int errors = GPUBuildKNNTest();
    printf("outside\n");
    EXPECT_EQ(errors, 0);
}

/*
int GPUTestDistance_All();

// TEST(GPUTest, DistanceTests) {
  int errs = GPUTestDistance_All();
  EXPECT_EQ(errs, 0);
// }

int GPUBuildTPTTest();

// TEST(GPUTest, TPTreeTests) {
  int errs = GPUBuildTPTTest();
  EXPECT_EQ(errs, 0);
// }

int GPUBuildSSDTest_All();

// TEST(GPUTest, BuildSSDTests) {
  int errs = GPUBuildSSDTest_All();
  EXPECT_EQ(errs, 0);
// }
*/
