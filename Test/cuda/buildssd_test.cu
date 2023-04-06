#include "common.hxx"
#include "inc/Core/Common/cuda/TailNeighbors.hxx"
//#include "inc/Core/Common/cuda/TPtree.hxx"

template<typename T, typename SUMTYPE, int dim, int K>
int GPUBuildSSDTest(int rows, int metric, int iters);

int GPUBuildSSDTest_All() {

  int errors = 0;

  LOG(SPTAG::Helper::LogLevel::LL_Info, "Starting Cosine BuildSSD tests\n");
  LOG(SPTAG::Helper::LogLevel::LL_Info, "Float datatype...\n");
  errors += GPUBuildSSDTest<float, float, 10, 8>(1000, (int)DistMetric::Cosine, 10); 
  errors += GPUBuildSSDTest<float, float, 100, 8>(1000, (int)DistMetric::Cosine, 10);
  errors += GPUBuildSSDTest<float, float, 200, 8>(1000, (int)DistMetric::Cosine, 10);
  errors += GPUBuildSSDTest<float, float, 384, 8>(1000, (int)DistMetric::Cosine, 10);
  errors += GPUBuildSSDTest<float, float, 1024, 8>(1000, (int)DistMetric::Cosine, 10);

  return errors;
}

template<typename T, typename SUMTYPE, int dim, int K>
int GPUBuildSSDTest(int rows, int metric, int iters) {
  int errors = 0;
  LOG(SPTAG::Helper::LogLevel::LL_Info, "Starting BuildSSD Cosine metric tests\n");

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
  int TPTlevels = (int)std::log2(num_heads/200);

  TPtree* h_tree = new TPtree;
  h_tree->initialize(num_heads, TPTlevels, dim);  
  TPtree* d_tree;
  CUDA_CHECK(cudaMalloc(&d_tree, sizeof(TPtree)));

  for(int i=0; i<iters; ++i) {
    create_tptree_multigpu<T>(&h_tree, &d_head_ps, num_heads, TPTlevels, 1, NULL, 2, NULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check any host side values are correct in h_tree

    CUDA_CHECK(cudaMemcpy(d_tree, h_tree, sizeof(TPtree), cudaMemcpyHostToDevice));

    if(metric == (int)DistMetric::Cosine) {
      findTailNeighbors_selector<T,SUMTYPE,(int)DistMetric::Cosine><<<1024, 32, sizeof(DistPair<SUMTYPE>)*32*K>>>(d_head_ps, d_tail_ps, d_tree, K, d_results, rows, num_heads, NULL, dim);
    }
    else {
      findTailNeighbors_selector<T,SUMTYPE,(int)DistMetric::L2><<<1024, 32, sizeof(DistPair<SUMTYPE>)*32*K>>>(d_head_ps, d_tail_ps, d_tree, K, d_results, rows, num_heads, NULL, dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); 
 
    // Check results of iteration
  }
  // Check final results of buildSSD

  CUDA_CHECK(cudaFree(d_head_data));
  CUDA_CHECK(cudaFree(d_head_ps));
  CUDA_CHECK(cudaFree(d_tail_data));
  CUDA_CHECK(cudaFree(d_tail_ps));
  CUDA_CHECK(cudaFree(d_results));
  d_tree->destroy();
  CUDA_CHECK(cudaFree(d_tree));

  return errors;
}

