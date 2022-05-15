// GPU implementation of OPQ matrix update, based on NVIDIA cusolver sample code: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/Xgesvd/cusolver_Xgesvd_example.cu
#include "inc/Quantizer/OPQUpdate.hxx"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "inc/Quantizer/cusolver_utils.h"
#include <cublas_v2.h>

int OPQRotationUpdate(float* svd_mat, float* rotation, SPTAG::SizeType dim)
{
  cusolverDnHandle_t cusolverH = NULL;
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;

  std::vector<float> U(dim*dim);
  std::vector<float> VT(dim*dim);
  std::vector<float> S(dim);
  std::vector<float> S_exact(dim);

  
  float *d_svd = nullptr;
  float *d_S = nullptr;
  float *d_U = nullptr;
  float *d_VT = nullptr;
  int *d_info = nullptr;
  float *d_work = nullptr;
  float *d_rwork = nullptr;
  float *d_W = nullptr; // W = S*VT

  
  int lwork = 0;
  int info = 0;
  const float h_one = 1;
  const float h_zero = 0;
  const float h_minus_one = -1;

  /* step 1: create cusolver handle, bind a stream */
  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
  CUBLAS_CHECK(cublasCreate(&cublasH));

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  
  /* step 2: copy A to device */
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_svd), sizeof(float) * dim * dim));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(float) * S.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(float) * U.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_VT), sizeof(float) * VT.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(float) * dim * dim));

  CUDA_CHECK(cudaMemcpyAsync(d_svd, svd_mat, sizeof(float) * dim * dim, cudaMemcpyHostToDevice, stream));

  /* step 3: query working space of SVD */
  CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(cusolverH, dim, dim, &lwork));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(float) * lwork));

  
  /* step 4: compute SVD */
  signed char jobu = 'A';  // all dim columns of U
  signed char jobvt = 'A'; // all dim columns of VT

  CUSOLVER_CHECK(cusolverDnSgesvd(cusolverH, jobu, jobvt, dim, dim, d_svd, dim, d_S, d_U,
								  dim, 
								  d_VT,
								  dim,
								  d_work, lwork, d_rwork, d_info));

  
  CUDA_CHECK(cudaMemcpyAsync(U.data(), d_U, sizeof(float) * U.size(), cudaMemcpyDeviceToHost,
							 stream));
  CUDA_CHECK(cudaMemcpyAsync(VT.data(), d_VT, sizeof(float) * VT.size(),
							 cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(S.data(), d_S, sizeof(float) * S.size(), cudaMemcpyDeviceToHost,
							 stream));
  CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // rotation matrix = U * VT
  float *d_rot = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_rot), sizeof(float) * dim * dim));
  
  CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &h_one, d_U, dim, d_VT, dim, &h_zero, d_rot, dim));

  CUDA_CHECK(cudaMemcpyAsync(rotation, d_rot, sizeof(float)*dim*dim, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  /* free resources */
  CUDA_CHECK(cudaFree(d_svd));
  CUDA_CHECK(cudaFree(d_S));
  CUDA_CHECK(cudaFree(d_U));
  CUDA_CHECK(cudaFree(d_VT));
  CUDA_CHECK(cudaFree(d_info));
  CUDA_CHECK(cudaFree(d_work));
  CUDA_CHECK(cudaFree(d_rwork));
  CUDA_CHECK(cudaFree(d_W));
  CUDA_CHECK(cudaFree(d_rot));

  
  CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
  CUBLAS_CHECK(cublasDestroy(cublasH));

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaDeviceReset());

  return EXIT_SUCCESS;
}