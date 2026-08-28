//#include "inc/Core/Common/cuda/KNN.hxx"
#include <cstdlib>
#include <chrono>

#define CHECK_ERRS(errs) \
  if(errs > 0) {          \
    SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "%d errors found\n", errs); \
  }         

#define CHECK_VAL(val,exp,errs) \
  if(val != exp) { \
    errs++;        \
    SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "%s != %s\n",#val,#exp);   \
  }

#define CHECK_VAL_LT(val,exp,errs) \
  if(val > exp) { \
    errs++;        \
    SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "%s > %s\n",#val,#exp);   \
  }

#define GPU_CHECK_VAL(val,exp,dtype,errs) \
  dtype temp; \
  CUDA_CHECK(cudaMemcpy(&temp, val, sizeof(dtype), cudaMemcpyDeviceToHost)); \
  float eps = 0.01; \
  if((float)temp>0.0 && ((float)temp*(1.0+eps) < (float)(exp) || (float)temp*(1.0-eps) > (float)(exp))) { \
    errs++;        \
    SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "%s != %s\n",#val,#exp);   \
  }


template<typename T>
T* create_dataset(size_t rows, int dim) {

  srand(0);
  T* h_data = new T[rows*dim];
  for(size_t i=0; i<rows*dim; ++i) {
    if(std::is_same<T,float>::value) {
      h_data[i] = (rand()/(float)RAND_MAX);
    }
    else if(std::is_same<T,int>::value) {
      h_data[i] = static_cast<T>((rand()%INT_MAX));
    }
    else if(std::is_same<T,uint8_t>::value) {
      h_data[i] = static_cast<T>((rand()%127));
    }
    else if(std::is_same<T,int8_t>::value) {
      h_data[i] = static_cast<T>((rand()%127));
    }
  } 
  return h_data;
}
/*
__global__ void count_leaf_sizes(LeafNode* leafs, int* node_ids, int N, int internal_nodes);
__global__ void assign_leaf_points_in_batch(LeafNode* leafs, int* leaf_points, int* node_ids, int N, int internal_nodes, int min_id, int max_id);
__global__ void assign_leaf_points_out_batch(LeafNode* leafs, int* leaf_points, int* node_ids, int N, int internal_nodes, int min_id, int max_id);
__global__ void compute_mean(KEYTYPE* split_keys, int* node_sizes, int num_nodes);
__global__ void initialize_rands(curandState* states, int iter);
*/
