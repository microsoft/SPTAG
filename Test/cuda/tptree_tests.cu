#include "common.hxx"
#include "inc/Core/Common/cuda/TPtree.hxx"

template<typename T, typename SUMTYPE, int dim>
int TPTKernelsTest(int rows) {


  int errs = 0;

  T* data = create_dataset<T>(rows, dim);
  T* d_data;
  CUDA_CHECK(cudaMalloc(&d_data, dim*rows*sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_data, data, dim*rows*sizeof(T), cudaMemcpyHostToDevice));

  PointSet<T> h_ps;
  h_ps.dim = dim;
  h_ps.data = d_data;

  PointSet<T>* d_ps;
  
  CUDA_CHECK(cudaMalloc(&d_ps, sizeof(PointSet<T>)));
  CUDA_CHECK(cudaMemcpy(d_ps, &h_ps, sizeof(PointSet<T>), cudaMemcpyHostToDevice));

  int levels = (int)std::log2(rows/100); // TPT levels
  TPtree* tptree = new TPtree;
  tptree->initialize(rows, levels, dim);

  // Check that tptree structure properly initialized
  CHECK_VAL(tptree->Dim,dim,errs)
  CHECK_VAL(tptree->levels,levels,errs)
  CHECK_VAL(tptree->N,rows,errs)
  CHECK_VAL(tptree->num_leaves,pow(2,levels),errs)  

  // Create TPT structure and random weights
  KEYTYPE* h_weights = new KEYTYPE[tptree->levels*tptree->Dim];
  for(int i=0; i<tptree->levels*tptree->Dim; ++i) {
    h_weights[i] = ((rand()%2)*2)-1;
  }

  tptree->reset();
  CUDA_CHECK(cudaMemcpy(tptree->weight_list, h_weights, tptree->levels*tptree->Dim*sizeof(KEYTYPE), cudaMemcpyHostToDevice));


  curandState* states;
  CUDA_CHECK(cudaMalloc(&states, 1024*32*sizeof(curandState)));
  initialize_rands<<<1024,32>>>(states, 0);


  int nodes_on_level=1;
  for(int i=0; i<tptree->levels; ++i) {

    find_level_sum<T><<<1024,32>>>(d_ps, tptree->weight_list, dim, tptree->node_ids, tptree->split_keys, tptree->node_sizes, rows, nodes_on_level, i, rows);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* split_key_sum = new float[nodes_on_level];
    CUDA_CHECK(cudaMemcpy(split_key_sum, &tptree->split_keys[nodes_on_level-1], nodes_on_level*sizeof(float), cudaMemcpyDeviceToHost)); // Copy the sum to compare with mean computed later
    int* node_sizes = new int[nodes_on_level];
    CUDA_CHECK(cudaMemcpy(node_sizes, &tptree->node_sizes[nodes_on_level-1], nodes_on_level*sizeof(int), cudaMemcpyDeviceToHost));

    compute_mean<<<1024, 32>>>(tptree->split_keys, tptree->node_sizes, tptree->num_nodes);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check the mean values for each node
    for(int j=0; j<nodes_on_level; ++j) {
      GPU_CHECK_VAL(&tptree->split_keys[nodes_on_level-1+j],split_key_sum[j]/(float)(node_sizes[j]),float,errs)
    }

    update_node_assignments<T><<<1024, 32>>>(d_ps, tptree->weight_list, tptree->node_ids, tptree->split_keys, tptree->node_sizes, rows, i, dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    nodes_on_level *= 2;
  }
  count_leaf_sizes<<<1024, 32>>>(tptree->leafs, tptree->node_ids, rows, tptree->num_nodes - tptree->num_leaves);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Check that total leaf node sizes equals total vectors
  int total_leaf_sizes=0;
  for(int j=0; j<tptree->num_leaves; ++j) {
    LeafNode temp_leaf;
    CUDA_CHECK(cudaMemcpy(&temp_leaf, &tptree->leafs[j], sizeof(LeafNode), cudaMemcpyDeviceToHost));
    total_leaf_sizes+=temp_leaf.size;
  }
  CHECK_VAL_LT(total_leaf_sizes,rows,errs)

  assign_leaf_points_out_batch<<<1024, 32>>>(tptree->leafs, tptree->leaf_points, tptree->node_ids, rows, tptree->num_nodes - tptree->num_leaves, 0, rows);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Check that points were correctly assigned to leaf nodes
  int* h_leaf_points = new int[rows];
  CUDA_CHECK(cudaMemcpy(h_leaf_points, tptree->leaf_points, rows*sizeof(int), cudaMemcpyDeviceToHost));
  for(int j=0; j<rows; ++j) {
    CHECK_VAL_LT(h_leaf_points[j],tptree->num_leaves,errs)
  }

  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_ps));
  CUDA_CHECK(cudaFree(states));
  tptree->destroy();

  return errs;
}

int GPUBuildTPTTest() {

  int errors = 0;

  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Starting TPTree Kernel tests\n");
  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Float datatype...\n");
  errors += TPTKernelsTest<float, float, 100>(1000);
  errors += TPTKernelsTest<float, float, 200>(1000);
  errors += TPTKernelsTest<float, float, 384>(1000);
  errors += TPTKernelsTest<float, float, 1024>(1000);

//  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "int32 datatype...\n");
//  errors += TPTKernelsTest<int, int, 100>(1000);
//  errors += TPTKernelsTest<int, int, 200>(1000);
//  errors += TPTKernelsTest<int, int, 384>(1000);
//  errors += TPTKernelsTest<int, int, 1024>(1000);

  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "int8 datatype...\n");
  errors += TPTKernelsTest<int8_t, int32_t, 100>(1000);
  errors += TPTKernelsTest<int8_t, int32_t, 200>(1000);
  errors += TPTKernelsTest<int8_t, int32_t, 384>(1000);
  errors += TPTKernelsTest<int8_t, int32_t, 1024>(1000);

  return errors;
}
