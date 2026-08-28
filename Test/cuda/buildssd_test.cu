#include "common.hxx"
#include "inc/Core/Common/cuda/TailNeighbors.hxx"

template<typename T, typename SUMTYPE, int dim, int K>
int GPUBuildSSDTest(int rows, int metric, int iters);

int GPUBuildSSDTest_All() {

  int errors = 0;

  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Starting Cosine BuildSSD tests\n");
  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Float datatype...\n");
  errors += GPUBuildSSDTest<float, float, 10, 8>(10000, (int)DistMetric::Cosine, 10); 
  errors += GPUBuildSSDTest<float, float, 100, 8>(10000, (int)DistMetric::Cosine, 10);
  errors += GPUBuildSSDTest<float, float, 200, 8>(10000, (int)DistMetric::Cosine, 10);
  errors += GPUBuildSSDTest<float, float, 384, 8>(10000, (int)DistMetric::Cosine, 10);
  errors += GPUBuildSSDTest<float, float, 1024, 8>(10000, (int)DistMetric::Cosine, 10);

  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Int datatype...\n");
  errors += GPUBuildSSDTest<int, int, 10, 8>(10000, (int)DistMetric::Cosine, 10); 
  errors += GPUBuildSSDTest<int, int, 100, 8>(10000, (int)DistMetric::Cosine, 10);
  errors += GPUBuildSSDTest<int, int, 200, 8>(10000, (int)DistMetric::Cosine, 10);
  errors += GPUBuildSSDTest<int, int, 384, 8>(10000, (int)DistMetric::Cosine, 10);

  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Int8 datatype...\n");
  errors += GPUBuildSSDTest<int8_t, int32_t, 100, 8>(10000, (int)DistMetric::Cosine, 10);
  errors += GPUBuildSSDTest<int8_t, int32_t, 200, 8>(10000, (int)DistMetric::Cosine, 10);
  errors += GPUBuildSSDTest<int8_t, int32_t, 384, 8>(10000, (int)DistMetric::Cosine, 10);
  errors += GPUBuildSSDTest<int8_t, int32_t, 1024, 8>(10000, (int)DistMetric::Cosine, 10);


  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Starting L2 BuildSSD tests\n");
  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Float datatype...\n");
  errors += GPUBuildSSDTest<float, float, 100, 8>(10000, (int)DistMetric::L2, 10);
  errors += GPUBuildSSDTest<float, float, 200, 8>(10000, (int)DistMetric::L2, 10);
  errors += GPUBuildSSDTest<float, float, 384, 8>(10000, (int)DistMetric::L2, 10);
  errors += GPUBuildSSDTest<float, float, 1024, 8>(10000, (int)DistMetric::L2, 10);

  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Int datatype...\n");
  errors += GPUBuildSSDTest<int, int, 10, 8>(10000, (int)DistMetric::L2, 10); 
  errors += GPUBuildSSDTest<int, int, 100, 8>(10000, (int)DistMetric::L2, 10);
  errors += GPUBuildSSDTest<int, int, 200, 8>(10000, (int)DistMetric::L2, 10);
  errors += GPUBuildSSDTest<int, int, 384, 8>(10000, (int)DistMetric::L2, 10);

  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Int8 datatype...\n");
  errors += GPUBuildSSDTest<int8_t, int32_t, 100, 8>(10000, (int)DistMetric::L2, 10);
  errors += GPUBuildSSDTest<int8_t, int32_t, 200, 8>(10000, (int)DistMetric::L2, 10);
  errors += GPUBuildSSDTest<int8_t, int32_t, 384, 8>(10000, (int)DistMetric::L2, 10);
  errors += GPUBuildSSDTest<int8_t, int32_t, 1024, 8>(10000, (int)DistMetric::L2, 10);

  return errors;
}

template<typename T, typename SUMTYPE, int dim, int K>
int GPUBuildSSDTest(int rows, int metric, int iters) {
  int errors = 0;
  int num_heads = rows/10;

  // Create random data for head vectors
  T* head_data = create_dataset<T>(num_heads, dim);
  T* d_head_data;
  CUDA_CHECK(cudaMalloc(&d_head_data, dim*num_heads*sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_head_data, head_data, dim*num_heads*sizeof(T), cudaMemcpyHostToDevice));
  PointSet<T> h_head_ps;
  h_head_ps.dim = dim;
  h_head_ps.data = d_head_data;
  PointSet<T>* d_head_ps;
  CUDA_CHECK(cudaMalloc(&d_head_ps, sizeof(PointSet<T>)));
  CUDA_CHECK(cudaMemcpy(d_head_ps, &h_head_ps, sizeof(PointSet<T>), cudaMemcpyHostToDevice));

  // Create random data for tail vectors
  T* tail_data = create_dataset<T>(rows, dim);
  T* d_tail_data;
  CUDA_CHECK(cudaMalloc(&d_tail_data, dim*rows*sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_tail_data, tail_data, dim*rows*sizeof(T), cudaMemcpyHostToDevice));
  PointSet<T> h_tail_ps;
  h_tail_ps.dim = dim;
  h_tail_ps.data = d_tail_data;
  PointSet<T>* d_tail_ps;
  CUDA_CHECK(cudaMalloc(&d_tail_ps, sizeof(PointSet<T>)));
  CUDA_CHECK(cudaMemcpy(d_tail_ps, &h_tail_ps, sizeof(PointSet<T>), cudaMemcpyHostToDevice));

  DistPair<SUMTYPE>* d_results;
  CUDA_CHECK(cudaMalloc(&d_results, rows*K*sizeof(DistPair<SUMTYPE>)));
  int TPTlevels = (int)std::log2(num_heads/100);

  TPtree* h_tree = new TPtree;
  h_tree->initialize(num_heads, TPTlevels, dim);  
  TPtree* d_tree;
  CUDA_CHECK(cudaMalloc(&d_tree, sizeof(TPtree)));

  // Alloc memory for QuerySet structure
  int* d_queryMem;
  CUDA_CHECK(cudaMalloc(&d_queryMem, rows*sizeof(int) + 2*h_tree->num_leaves*sizeof(int)));
  QueryGroup* d_queryGroups;
  CUDA_CHECK(cudaMalloc(&d_queryGroups, sizeof(QueryGroup)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  for(int i=0; i<iters; ++i) {
    create_tptree_multigpu<T>(&h_tree, &d_head_ps, num_heads, TPTlevels, 1, &stream, 2, NULL);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_tree, h_tree, sizeof(TPtree), cudaMemcpyHostToDevice));

    get_query_groups<T,SUMTYPE>(d_queryGroups, d_tree, d_tail_ps, rows, (int)h_tree->num_leaves, d_queryMem, 1024, 32, dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    if(metric == (int)DistMetric::Cosine) {
      findTailNeighbors_selector<T,SUMTYPE,(int)DistMetric::Cosine><<<1024, 32, sizeof(DistPair<SUMTYPE>)*32*K>>>(d_head_ps, d_tail_ps, d_tree, K, d_results, rows, num_heads, d_queryGroups, dim);
    }
    else {
      findTailNeighbors_selector<T,SUMTYPE,(int)DistMetric::L2><<<1024, 32, sizeof(DistPair<SUMTYPE>)*32*K>>>(d_head_ps, d_tail_ps, d_tree, K, d_results, rows, num_heads, d_queryGroups, dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); 
  }

  CUDA_CHECK(cudaFree(d_head_data));
  CUDA_CHECK(cudaFree(d_head_ps));
  CUDA_CHECK(cudaFree(d_tail_data));
  CUDA_CHECK(cudaFree(d_tail_ps));
  CUDA_CHECK(cudaFree(d_results));
  h_tree->destroy();
  CUDA_CHECK(cudaFree(d_tree));

  return errors;
}

