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
  BOOST_CHECK(errors == 0);
}

int GPUTestDistance_All(); 

BOOST_AUTO_TEST_CASE(DistanceTests) {
  int success = GPUTestDistance_All();
  BOOST_CHECK(success == 1);
}
