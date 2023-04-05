#include "common.hxx"


int GPUBuildSSDCosineTest_All() {

  int errors = 0;

  LOG(SPTAG::Helper::LogLevel::LL_Info, "Float datatype tests...\n");
  errors += GPUBuildKNNCosineTest<float, float, 10, 10>(1000);
  errors += GPUBuildKNNCosineTest<float, float, 100, 10>(1000);
  errors += GPUBuildKNNCosineTest<float, float, 200, 10>(1000);
  errors += GPUBuildKNNCosineTest<float, float, 384, 10>(1000);
  errors += GPUBuildKNNCosineTest<float, float, 1024, 10>(1000);

  LOG(SPTAG::Helper::LogLevel::LL_Info, "Int32 datatype tests...\n");
  errors += GPUBuildKNNCosineTest<int, int, 10, 10>(1000);
  errors += GPUBuildKNNCosineTest<int, int, 100, 10>(1000);
  errors += GPUBuildKNNCosineTest<int, int, 200, 10>(1000);
  errors += GPUBuildKNNCosineTest<int, int, 384, 10>(1000);
  errors += GPUBuildKNNCosineTest<int, int, 1024, 10>(1000);

  LOG(SPTAG::Helper::LogLevel::LL_Info, "Int8 datatype tests...\n");
  errors += GPUBuildKNNCosineTest<int8_t, int32_t, 100, 10>(1000);
  errors += GPUBuildKNNCosineTest<int8_t, int32_t, 200, 10>(1000);
  errors += GPUBuildKNNCosineTest<int8_t, int32_t, 384, 10>(1000);
  errors += GPUBuildKNNCosineTest<int8_t, int32_t, 1024, 10>(1000);
 
  return errors;
}


template<typename T, typename SUMTYPE, int dim, int K>
int GPUBuildSSDCosineTest(int rows) {
  int errors = 0;
  LOG(SPTAG::Helper::LogLevel::LL_Info, "Starting BuildSSD Cosine metric tests\n");

  T* data = create_dataset<T>(rows, dim);
  T* d_data;
  CUDA_CHECK(cudaMalloc(&d_data, dim*rows*sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_data, data, dim*rows*sizeof(T), cudaMemcpyHostToDevice));

  int* d_results;
  CUDA_CHECK(cudaMalloc(&d_results, rows*K*sizeof(int)));

  PointSet<T> h_ps;
  h_ps.dim = dim;
  h_ps.data = d_data;

  PointSet<T>* d_ps;
  
  CUDA_CHECK(cudaMalloc(&d_ps, sizeof(PointSet<T>)));
  CUDA_CHECK(cudaMemcpy(d_ps, &h_ps, sizeof(PointSet<T>), cudaMemcpyHostToDevice));

  return errors;
}

