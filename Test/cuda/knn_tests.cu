#include "common.hxx"
#include "inc/Core/Common/cuda/KNN.hxx"

template<typename T, typename SUMTYPE, int Dim, int metric>
__global__ void test_KNN(PointSet<T>* ps, int* results, int rows, int K) {

  extern __shared__ char sharememory[];
  DistPair<SUMTYPE>* threadList = (&((DistPair<SUMTYPE>*)sharememory)[K*threadIdx.x]);

  T query[Dim];
  T candidate_vec[Dim];

  DistPair<SUMTYPE> target;
  DistPair<SUMTYPE> candidate;
  DistPair<SUMTYPE> temp;

  bool good;
  SUMTYPE max_dist = INFTY<SUMTYPE>();
  int read_id, write_id;

  SUMTYPE (*dist_comp)(T*,T*) = &dist<T,SUMTYPE,Dim,metric>;

  for(size_t i=blockIdx.x*blockDim.x + threadIdx.x; i<rows; i+=blockDim.x*gridDim.x) {
    for(int j=0; j<Dim; ++j) {
      query[j] = ps->getVec(i)[j];
    }
    for(int k=0; k<K; k++) {
      threadList[k].dist=INFTY<SUMTYPE>();
    }

    for(size_t j=0; j<rows; ++j) {
      good = true;
      candidate.idx = j;
      candidate.dist = dist_comp(query, ps->getVec(j));

      if(max_dist > candidate.dist) {
        for(read_id=0; candidate.dist > threadList[read_id].dist && good; read_id++) {
          if(violatesRNG<T,SUMTYPE>(candidate_vec, ps->getVec(threadList[read_id].idx), candidate.dist, dist_comp)) {
            good = false;
          }
        }
        if(good) {
          target = threadList[read_id];
          threadList[read_id] = candidate;
          read_id++;
          for(write_id = read_id; read_id < K && threadList[read_id].idx != -1; read_id++) {
            if(!violatesRNG<T, SUMTYPE>(ps->getVec(threadList[read_id].idx), candidate_vec, threadList[read_id].dist, dist_comp)) {
              if(read_id == write_id) {
                temp = threadList[read_id];
                threadList[write_id] = target;
                target = temp;
              }
              else {
                threadList[write_id] = target;
                target = threadList[read_id];
              }
              write_id++;
            }
          }
          if(write_id < K) {
            threadList[write_id] = target;
            write_id++;
          }
          for(int k=write_id; k<K && threadList[k].idx != -1; k++) {
            threadList[k].dist = INFTY<SUMTYPE>();
            threadList[k].idx = -1;
          }
          max_dist = threadList[K-1].dist;
        }

      }

    }
    for(size_t j=0; j<K; j++) {
      results[(size_t)(i)*K+j] = threadList[j].idx;
    }
  }
}


template<typename T, typename SUMTYPE, int dim, int K>
int GPUBuildKNNCosineTest(int rows) {

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

  test_KNN<T,SUMTYPE,dim,(int)DistMetric::Cosine><<<1024, 64, K*64*sizeof(DistPair<SUMTYPE>)>>>(d_ps, d_results, rows, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  int* h_results = new int[K*rows];
  CUDA_CHECK(cudaMemcpy(h_results, d_results, rows*K*sizeof(int), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaDeviceSynchronize());

// Verify that the neighbor list of each vector is ordered correctly
  for(int i=0; i<rows; ++i) {
    for(int j=0; j<K-1; ++j) {
      int neighborId = h_results[i*K+j];
      int nextNeighborId = h_results[i*K+j+1];
      if(neighborId != -1 && nextNeighborId != -1) {
        SUMTYPE neighborDist = (SUMTYPE)(SPTAG::COMMON::DistanceUtils::ComputeCosineDistance<T>(&data[i*dim], &data[neighborId*dim], dim));
        SUMTYPE nextDist = (SUMTYPE)(SPTAG::COMMON::DistanceUtils::ComputeCosineDistance<T>(&data[i*dim], &data[nextNeighborId*dim], dim));
        if(neighborDist > nextDist) {
          SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Neighbor list not in ascending distance order. i:%d, neighbor:%d (dist:%f), next:%d (dist:%f)\n", i, neighborId, neighborDist, nextNeighborId, nextDist);
          return 1;
        }
      }
    }
  }

  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_results));
  CUDA_CHECK(cudaFree(d_ps));

  return 0;
}

int GPUBuildKNNCosineTest() {

  int errors = 0;

  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Float datatype tests...\n");
  errors += GPUBuildKNNCosineTest<float, float, 10, 10>(1000);
  errors += GPUBuildKNNCosineTest<float, float, 100, 10>(1000);
  errors += GPUBuildKNNCosineTest<float, float, 200, 10>(1000);
  errors += GPUBuildKNNCosineTest<float, float, 384, 10>(1000);
  errors += GPUBuildKNNCosineTest<float, float, 1024, 10>(1000);

  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Int32 datatype tests...\n");
  errors += GPUBuildKNNCosineTest<int, int, 10, 10>(1000);
  errors += GPUBuildKNNCosineTest<int, int, 100, 10>(1000);
  errors += GPUBuildKNNCosineTest<int, int, 200, 10>(1000);
  errors += GPUBuildKNNCosineTest<int, int, 384, 10>(1000);
  errors += GPUBuildKNNCosineTest<int, int, 1024, 10>(1000);

  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Int8 datatype tests...\n");
  errors += GPUBuildKNNCosineTest<int8_t, int32_t, 100, 10>(1000);
  errors += GPUBuildKNNCosineTest<int8_t, int32_t, 200, 10>(1000);
  errors += GPUBuildKNNCosineTest<int8_t, int32_t, 384, 10>(1000);
  errors += GPUBuildKNNCosineTest<int8_t, int32_t, 1024, 10>(1000);
 
  return errors;
}

template<typename T, typename SUMTYPE, int dim, int K>
int GPUBuildKNNL2Test(int rows) {

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

  test_KNN<T,SUMTYPE,dim,(int)DistMetric::L2><<<1024, 64, K*64*sizeof(DistPair<SUMTYPE>)>>>(d_ps, d_results, rows, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  int* h_results = new int[K*rows];
  CUDA_CHECK(cudaMemcpy(h_results, d_results, rows*K*sizeof(int), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaDeviceSynchronize());

// Verify that the neighbor list of each vector is ordered correctly
  for(int i=0; i<rows; ++i) {
    for(int j=0; j<K-1; ++j) {
      int neighborId = h_results[i*K+j];
      int nextNeighborId = h_results[i*K+j+1];
      if(neighborId != -1 && nextNeighborId != -1) {
        SUMTYPE neighborDist = (SUMTYPE)(SPTAG::COMMON::DistanceUtils::ComputeL2Distance<T>(&data[i*dim], &data[neighborId*dim], dim));
        SUMTYPE nextDist = (SUMTYPE)(SPTAG::COMMON::DistanceUtils::ComputeL2Distance<T>(&data[i*dim], &data[nextNeighborId*dim], dim));
        if(neighborDist > nextDist) {
          SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Neighbor list not in ascending distance order. i:%d, neighbor:%d (dist:%f), next:%d (dist:%f)\n", i, neighborId, neighborDist, nextNeighborId, nextDist);
          return 1;
        }
      }
    }
  }
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_results));
  CUDA_CHECK(cudaFree(d_ps));

  return 0;
}

int GPUBuildKNNL2Test() {

  int errors = 0;

  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Float datatype tests...\n");
  errors += GPUBuildKNNL2Test<float, float, 10, 10>(1000);
  errors += GPUBuildKNNL2Test<float, float, 100, 10>(1000);
  errors += GPUBuildKNNL2Test<float, float, 200, 10>(1000);
  errors += GPUBuildKNNL2Test<float, float, 384, 10>(1000);
  errors += GPUBuildKNNL2Test<float, float, 1024, 10>(1000);

  CHECK_ERRS(errors)

  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Int32 datatype tests...\n");
  errors += GPUBuildKNNL2Test<int, int, 10, 10>(1000);
  errors += GPUBuildKNNL2Test<int, int, 100, 10>(1000);
  errors += GPUBuildKNNL2Test<int, int, 200, 10>(1000);
  errors += GPUBuildKNNL2Test<int, int, 384, 10>(1000);
  errors += GPUBuildKNNL2Test<int, int, 1024, 10>(1000);

  CHECK_ERRS(errors)

  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Int8 datatype tests...\n");
  errors += GPUBuildKNNL2Test<int8_t, int32_t, 100, 10>(1000);
  errors += GPUBuildKNNL2Test<int8_t, int32_t, 200, 10>(1000);
  errors += GPUBuildKNNL2Test<int8_t, int32_t, 384, 10>(1000);
  errors += GPUBuildKNNL2Test<int8_t, int32_t, 1024, 10>(1000);

  CHECK_ERRS(errors)
}

int GPUBuildKNNTest() {
  int errors = 0;
  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Starting KNN Cosine metric tests\n");
  errors += GPUBuildKNNCosineTest();

  CHECK_ERRS(errors)

  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Starting KNN L2 metric tests\n");
  errors += GPUBuildKNNL2Test();

  CHECK_ERRS(errors)

  return errors;
}

int GPUBuildTPTreeTest();
