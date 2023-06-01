//#include "test_kernels.cu"

#define BOOST_TEST_MODULE GPU

#include <cstdlib>
#include <chrono>

#include <iostream>
#include <boost/test/included/unit_test.hpp>
#include <boost/filesystem.hpp>

int GPUBuildKNNTest();

BOOST_AUTO_TEST_CASE(RandomTests) {
  BOOST_CHECK(1 == 1);

  int errors = GPUBuildKNNTest();
printf("outside\n");
  BOOST_CHECK(errors == 0);
}

/*
int GPUTestDistance_All(); 

BOOST_AUTO_TEST_CASE(DistanceTests) {
  int errs = GPUTestDistance_All();
  BOOST_CHECK(errs == 0);
}

int GPUBuildTPTTest();

BOOST_AUTO_TEST_CASE(TPTreeTests) {
  int errs = GPUBuildTPTTest();
  BOOST_CHECK(errs == 0);
}

int GPUBuildSSDTest_All();

BOOST_AUTO_TEST_CASE(BuildSSDTests) {
  int errs = GPUBuildSSDTest_All();
  BOOST_CHECK(errs == 0);
}
*/
