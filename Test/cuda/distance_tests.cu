#include "common.hxx"
#include "inc/Core/Common/cuda/KNN.hxx"

#define GPU_CHECK_CORRECT(a,b,msg) \
  if(a != (SUMTYPE)b) {              \
    printf(msg);                     \
    errs = 0;                     \
    return;                          \
  }                                  


template<typename T, typename SUMTYPE, int Dim>
__global__ void GPUTestDistancesKernelStatic(PointSet<T>* ps, int errs) {
  SUMTYPE ab, ac, bc;
// Expected results
  SUMTYPE l2_res[3] = {Dim, 4*Dim, Dim};
  SUMTYPE cosine_res[3] = {BASE<SUMTYPE>(), BASE<SUMTYPE>(), BASE<SUMTYPE>()-2*Dim};

  // l2 dist
  ab = dist<T, SUMTYPE, Dim, (int)DistMetric::L2>(ps->getVec(0), ps->getVec(1));
  ac = dist<T, SUMTYPE, Dim, (int)DistMetric::L2>(ps->getVec(0), ps->getVec(2));
  bc = dist<T, SUMTYPE, Dim, (int)DistMetric::L2>(ps->getVec(1), ps->getVec(2));
  GPU_CHECK_CORRECT(ab,l2_res[0],"Static L2 distance check failed\n");
  GPU_CHECK_CORRECT(ac,l2_res[1], "Static L2 distance check failed\n");
  GPU_CHECK_CORRECT(bc,l2_res[2], "Static L2 distance check failed\n");

  // cosine dist
  ab = dist<T, SUMTYPE, Dim, (int)DistMetric::Cosine>(ps->getVec(0), ps->getVec(1));
  ac = dist<T, SUMTYPE, Dim, (int)DistMetric::Cosine>(ps->getVec(0), ps->getVec(2));
  bc = dist<T, SUMTYPE, Dim, (int)DistMetric::Cosine>(ps->getVec(1), ps->getVec(2));
  GPU_CHECK_CORRECT(ab,cosine_res[0],"Static Cosine distance check failed\n");
  GPU_CHECK_CORRECT(ac,cosine_res[1],"Static Cosine distance check failed\n");
  GPU_CHECK_CORRECT(bc,cosine_res[2],"Static Cosine distance check failed\n");

}

template<typename T, typename SUMTYPE, int dim>
int GPUTestDistancesSimple() {
  T* h_data = new T[dim*3];
  for(int i=0; i<3; ++i) {
    for(int j=0; j<dim; ++j) {
      h_data[i*dim+j] = (T)(i);
    }
  }

  T* d_data;
  CUDA_CHECK(cudaMalloc(&d_data, dim*3*sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_data, h_data, dim*3*sizeof(T), cudaMemcpyHostToDevice));

  PointSet<T> h_ps;
  h_ps.dim = dim;
  h_ps.data = d_data;
  PointSet<T>* d_ps;
  CUDA_CHECK(cudaMalloc(&d_ps, sizeof(PointSet<T>)));
  CUDA_CHECK(cudaMemcpy(d_ps, &h_ps, sizeof(PointSet<T>), cudaMemcpyHostToDevice));

 
  int errs=0;
  GPUTestDistancesKernelStatic<T,SUMTYPE,dim><<<1, 1>>>(d_ps, errs); // TODO - make sure result is correct returned
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_ps));
  return errs;
}


template<typename T, typename SUMTYPE, int dim>
__global__ void GPUTestDistancesRandomKernel(PointSet<T>* ps, int vecs, SUMTYPE* cosine_dists, SUMTYPE* l2_dists, int errs) {
  SUMTYPE cos, l2;
  SUMTYPE diff;
  float eps = 0.001; // exact distances are not expected, but should be within an epsilon
  for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<vecs; i+= gridDim.x*blockDim.x) {
    for(int j=0; j<vecs; ++j) {
      cos = dist<T,SUMTYPE,dim, (int)DistMetric::Cosine>(ps->getVec(i), ps->getVec(j));
      if( (((float)(cos - (SUMTYPE)cosine_dists[i*vecs+j]))/(float)cos) > eps) {
        printf("Error, cosine calculations differ by too much i:%d, j:%d - GPU:%u, CPU:%u, diff:%f\n", i, j, cos, (SUMTYPE)cosine_dists[i*vecs+j], (((float)(cos - (SUMTYPE)cosine_dists[i*vecs+j]))/(float)cos));
return;
      }

      l2 = dist<T,SUMTYPE,dim, (int)DistMetric::L2>(ps->getVec(i), ps->getVec(j));
      if( (((float)(l2 - (SUMTYPE)l2_dists[i*vecs+j]))/(float)l2) > eps) {
        printf("Error, l2 calculations differ by too much i:%d, j:%d - GPU:%d, CPU:%d\n", i, j, l2, (int)l2_dists[i*vecs+j]);
return;
      }
    }
  }
}

template<typename T, typename SUMTYPE, int dim>
int GPUTestDistancesComplex(int vecs) {

  srand(time(NULL));
  T* h_data = new T[vecs*dim];
  for(int i=0; i<vecs; ++i) {
    for(int j=0; j<dim; ++j) {
      if(std::is_same<T,float>::value) {
        h_data[i*dim+j] = (rand()/(float)RAND_MAX);
      }
      else if(std::is_same<T,int>::value) {
        h_data[i*dim+j] = static_cast<T>((rand()%INT_MAX));
      }
      else if(std::is_same<T,uint8_t>::value) {
        h_data[i*dim+j] = static_cast<T>((rand()%127));
      }
      else if(std::is_same<T,int8_t>::value) {
        h_data[i*dim+j] = static_cast<T>((rand()%127));
      }
    }
  }

 // Compute CPU distances to verify with GPU metric
  SUMTYPE* cpu_cosine_dists = new SUMTYPE[vecs*vecs];
  SUMTYPE* cpu_l2_dists = new SUMTYPE[vecs*vecs];
  for(int i=0; i<vecs; ++i) {
    for(int j=0; j<vecs; ++j) {
      cpu_cosine_dists[i*vecs+j] = (SUMTYPE)(SPTAG::COMMON::DistanceUtils::ComputeCosineDistance<T>(&h_data[i*dim], &h_data[j*dim], dim));
      cpu_l2_dists[i*vecs+j] = (SUMTYPE)(SPTAG::COMMON::DistanceUtils::ComputeL2Distance<T>(&h_data[i*dim], &h_data[j*dim], dim));
    }
  }
  int errs=0;

  T* d_data;
  CUDA_CHECK(cudaMalloc(&d_data, dim*vecs*sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_data, h_data, dim*vecs*sizeof(T), cudaMemcpyHostToDevice));

  PointSet<T> h_ps;
  h_ps.dim = dim;
  h_ps.data = d_data;
  PointSet<T>* d_ps;
  CUDA_CHECK(cudaMalloc(&d_ps, sizeof(PointSet<T>)));
  CUDA_CHECK(cudaMemcpy(d_ps, &h_ps, sizeof(PointSet<T>), cudaMemcpyHostToDevice));


  SUMTYPE* d_cosine_dists;
  CUDA_CHECK(cudaMalloc(&d_cosine_dists, vecs*vecs*sizeof(SUMTYPE)));
  CUDA_CHECK(cudaMemcpy(d_cosine_dists, cpu_cosine_dists, vecs*vecs*sizeof(SUMTYPE), cudaMemcpyHostToDevice));
  SUMTYPE* d_l2_dists;
  CUDA_CHECK(cudaMalloc(&d_l2_dists, vecs*vecs*sizeof(SUMTYPE)));
  CUDA_CHECK(cudaMemcpy(d_l2_dists, cpu_l2_dists, vecs*vecs*sizeof(SUMTYPE), cudaMemcpyHostToDevice));

 
  GPUTestDistancesRandomKernel<T,SUMTYPE,dim><<<1024, 32>>>(d_ps, vecs, d_cosine_dists, d_l2_dists, errs); // TODO - make sure result is correct returned
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_ps));
  CUDA_CHECK(cudaFree(d_cosine_dists));
  CUDA_CHECK(cudaFree(d_l2_dists));

  return errs;
}

int GPUTestDistance_All() {

  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Static distance tests...\n");
  // Test Distances with float datatype
  int errs = 0;
  errs += GPUTestDistancesSimple<float, float, 10>();
  errs += GPUTestDistancesSimple<float, float, 100>();
  errs += GPUTestDistancesSimple<float, float, 200>();
  errs += GPUTestDistancesSimple<float, float, 384>();
  errs += GPUTestDistancesSimple<float, float, 1024>();

  // Test distances with int datatype
  errs += GPUTestDistancesSimple<int, int, 10>();
  errs += GPUTestDistancesSimple<int, int, 100>();
  errs += GPUTestDistancesSimple<int, int, 200>();
  errs += GPUTestDistancesSimple<int, int, 384>();
  errs += GPUTestDistancesSimple<int, int, 1024>();

  // Test distances with int8 datatype
  errs += GPUTestDistancesSimple<int8_t, int32_t, 100>();
  errs += GPUTestDistancesSimple<int8_t, int32_t, 200>();
  errs += GPUTestDistancesSimple<int8_t, int32_t, 384>();
  errs += GPUTestDistancesSimple<int8_t, int32_t, 1024>();

  CHECK_ERRS(errs)

  SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Randomized vector distance tests...\n");
  // Test distances between random vectors and compare with CPU calculation
  errs += GPUTestDistancesComplex<float, float, 10>(100);
  errs += GPUTestDistancesComplex<float, float, 100>(100);
  errs += GPUTestDistancesComplex<float, float, 200>(100);
  errs += GPUTestDistancesComplex<float, float, 384>(100);
  errs += GPUTestDistancesComplex<float, float, 1024>(100);

  errs += GPUTestDistancesComplex<int, int, 10>(100);
  errs += GPUTestDistancesComplex<int, int, 100>(100);
  errs += GPUTestDistancesComplex<int, int, 200>(100);
  errs += GPUTestDistancesComplex<int, int, 384>(100);
  errs += GPUTestDistancesComplex<int, int, 1024>(100);

  errs += GPUTestDistancesComplex<int8_t, int32_t, 100>(100);
  errs += GPUTestDistancesComplex<int8_t, int32_t, 200>(100);
  errs += GPUTestDistancesComplex<int8_t, int32_t, 384>(100);
  errs += GPUTestDistancesComplex<int8_t, int32_t, 1024>(100);

  CHECK_ERRS(errs)

  return errs;
}
