
#include "common.hxx"
#include "inc/Core/Common/cuda/KNN.hxx"

template<typename T, int Dim, int metric>
__global__ void top1_nopq_kernel(T* data, int* res_idx, float* results, int datasize) {
  float temp;
  for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<datasize; i+=blockDim.x*gridDim.x) {
    results[i] = INFTY<T>();
    for(int j=0; j<datasize; ++j) {
      if(j != i) {
        temp = dist<T,T,Dim,metric>(&data[i*Dim], &data[j*Dim]);
        if(temp < results[i]) {
          results[i] = temp;
          res_idx[i] = j;
        }
      }
    }
  }
}

#define TEST_BLOCKS 512
#define TEST_THREADS 32

template<typename R>
void GPU_top1_nopq(std::shared_ptr<VectorSet>& real_vecset, DistCalcMethod distMethod, int* res_idx, float* res_dist) {

  R* d_data;
  CUDA_CHECK(cudaMalloc(&d_data, real_vecset->Count()*real_vecset->Dimension()*sizeof(R)));
  CUDA_CHECK(cudaMemcpy(d_data, reinterpret_cast<R*>(real_vecset->GetVector(0)), real_vecset->Count()*real_vecset->Dimension()*sizeof(R), cudaMemcpyHostToDevice));

//  float* nearest = new float[real_vecset->Count()];
  float* d_res_dist;
  CUDA_CHECK(cudaMalloc(&d_res_dist, real_vecset->Count()*sizeof(float)));
  int* d_res_idx;
  CUDA_CHECK(cudaMalloc(&d_res_idx, real_vecset->Count()*sizeof(int)));

  // Run kernel that performs 
  // Create options for different dims and metrics
  if(real_vecset->Dimension() == 256) {
    top1_nopq_kernel<R,256,(int)DistMetric::L2><<<TEST_BLOCKS,TEST_THREADS>>>(d_data, d_res_idx, d_res_dist, real_vecset->Count());
  }
  else {
    printf("Add support for testing with %d dimensions\n", real_vecset->Dimension());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(res_idx, d_res_idx, real_vecset->Count()*sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(res_dist, d_res_dist, real_vecset->Count()*sizeof(float), cudaMemcpyDeviceToHost));
  // Optional check results for debugging

}

void GPU_nopq_alltype(std::shared_ptr<VectorSet>& real_vecset, DistCalcMethod distMethod, int* res_idx, float* res_dist) {
  GPU_top1_nopq<float>(real_vecset, distMethod, res_idx, res_dist);
}


template<typename T, int Dim>
__global__ void top1_pq_kernel(uint8_t* data, int* res_idx, float* res_dist, int datasize, GPU_Quantizer* quantizer) {

  if(threadIdx.x==0 && blockIdx.x==0) {
    printf("Quantizer numSub:%d, KsPerSub:%ld, BlockSize:%ld, dimPerSub:%ld\n", quantizer->m_NumSubvectors, quantizer->m_KsPerSubvector, quantizer->m_BlockSize, quantizer->m_DimPerSubvector);
    printf("dataSize:%d, data: %u, %u\n", datasize, data[0] & 255, data[1] & 255);
  }
  float temp;
  for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<datasize; i+=blockDim.x*gridDim.x) {
    res_dist[i] = INFTY<T>();
    for(int j=0; j<datasize; ++j) {
      if(i != j) {
        temp = quantizer->dist(&data[i*Dim], &data[j*Dim]);
        if(temp < res_dist[i]) {
          res_dist[i] = temp;
          res_idx[i] = j;
          if(i==0) printf("j:%d, dist:%f\n", j, temp);
        }
      }
    }
  }
}

template<typename R>
void GPU_top1_pq(std::shared_ptr<VectorSet>& real_vecset, std::shared_ptr<VectorSet>& quan_vecset,DistCalcMethod distMethod, std::shared_ptr<COMMON::IQuantizer>& quantizer, int* res_idx, float* res_dist) {

  printf("Running GPU PQ - PQ dims:%d\n", quan_vecset->Dimension());

  std::shared_ptr<VectorIndex> vecIndex = SPTAG::VectorIndex::CreateInstance(IndexAlgoType::BKT, SPTAG::GetEnumValueType<uint8_t>());
  vecIndex->SetQuantizer(quantizer);
  vecIndex->SetParameter("DistCalcMethod", SPTAG::Helper::Convert::ConvertToString(distMethod));

  GPU_Quantizer* d_quantizer = NULL;
  GPU_Quantizer* h_quantizer = NULL;

printf("QTYPE:%d\n", (int)(quantizer->GetQuantizerType()));

printf("Creating GPU_Quantizer\n");
  h_quantizer = new GPU_Quantizer(quantizer, DistMetric::L2); // TODO - add other metric option
  CUDA_CHECK(cudaMalloc(&d_quantizer, sizeof(GPU_Quantizer)));
  CUDA_CHECK(cudaMemcpy(d_quantizer, h_quantizer, sizeof(GPU_Quantizer), cudaMemcpyHostToDevice));

  uint8_t* d_data;
  CUDA_CHECK(cudaMalloc(&d_data, quan_vecset->Count()*quan_vecset->Dimension()*sizeof(uint8_t)));
  CUDA_CHECK(cudaMemcpy(d_data, reinterpret_cast<uint8_t*>(quan_vecset->GetVector(0)), quan_vecset->Count()*quan_vecset->Dimension()*sizeof(uint8_t), cudaMemcpyHostToDevice));

//  float* nearest = new float[quan_vecset->Count()];
//  float* d_nearest;
//  CUDA_CHECK(cudaMalloc(&d_nearest, quan_vecset->Count()*sizeof(float)));

  float* d_res_dist;
  CUDA_CHECK(cudaMalloc(&d_res_dist, quan_vecset->Count()*sizeof(float)));
  int* d_res_idx;
  CUDA_CHECK(cudaMalloc(&d_res_idx, quan_vecset->Count()*sizeof(int)));

  // Run kernel that performs 
  // Create options for different dims and metrics
  if(quan_vecset->Dimension() == 128) {
    top1_pq_kernel<R,128><<<TEST_BLOCKS,TEST_THREADS>>>(d_data, d_res_idx, d_res_dist, quan_vecset->Count(), d_quantizer);
  }
  else {
    printf("Add support for testing with %d PQ dimensions\n", quan_vecset->Dimension());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(res_idx, d_res_idx, quan_vecset->Count()*sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(res_dist, d_res_dist, quan_vecset->Count()*sizeof(float), cudaMemcpyDeviceToHost));
  // Optional check results for debugging

}


void GPU_pq_alltype(std::shared_ptr<VectorSet>& real_vecset, std::shared_ptr<VectorSet>& quan_vecset, DistCalcMethod distMethod, std::shared_ptr<COMMON::IQuantizer>& quantizer, int* res_idx, float* res_dist) {
  GPU_top1_pq<float>(real_vecset, quan_vecset, distMethod, quantizer, res_idx, res_dist);

}

