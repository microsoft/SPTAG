#include "../Distance.hxx"
#include "../KNN.hxx"
#include "../params.h"
#include <cstdlib>
#include <chrono>


float* create_dataset(size_t rows, int dim) {

  srand(0);
  float* data = new float[rows*dim];
  for(size_t i=0; i<rows*dim; ++i) {
    data[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  } 
  return data;
}

template<typename T, typename SUMTYPE, int Dim>
__global__ void test_KNN(PointSet<T>* ps, int* results, int rows, int K) {

  extern __shared__ char sharememory[];
  DistPair<SUMTYPE>* threadList = (&((DistPair<SUMTYPE>*)sharememory)[K*threadIdx.x]);

  T query[Dim];
  T candidate_vec[Dim];

  DistPair<SUMTYPE> target;
  DistPair<SUMTYPE> candidate;
  DistPair<SUMTYPE> temp;

  bool good;
  float max_dist = INFTY<float>();
  int read_id, write_id;

  float (*dist_comp)(T*,T*) = cosine<T,SUMTYPE,Dim>;

  for(size_t i=blockIdx.x*blockDim.x + threadIdx.x; i<rows; i+=blockDim.x*gridDim.x) {
    for(int j=0; j<Dim; ++j) {
      query[j] = ps->getVec(i)[j];

      for(int k=0; k<K; k++) {
        threadList[k].dist=INFTY<float>();
      }
    }

    for(size_t j=0; j<rows; ++j) {
      good = true;
      candidate.idx = j;
      candidate.dist = dist_comp(query, ps->getVec(i));

      
      if(candidate.dist < max_dist) {
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

int main() {

  int dim=100;
  int rows = 100000;
  int K = 10;

  float* data = create_dataset(rows, dim);
  float* d_data;
for(int i=0; i<100; i++) printf("%f, ", data[i]);
printf("\n");
  CUDA_CHECK(cudaMalloc(&d_data, dim*rows*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_data, data, dim*rows*sizeof(float), cudaMemcpyHostToDevice));

  int* d_results;
  CUDA_CHECK(cudaMalloc(&d_results, rows*K*sizeof(int)));



  PointSet<float> h_ps;
  h_ps.dim = dim;
  h_ps.data = d_data;

  PointSet<float>* d_ps;
  
  CUDA_CHECK(cudaMalloc(&d_ps, sizeof(PointSet<float>)));
  CUDA_CHECK(cudaMemcpy(d_ps, &h_ps, sizeof(PointSet<float>), cudaMemcpyHostToDevice));

  auto start_t = std::chrono::high_resolution_clock::now();

  test_KNN<float,float,100><<<1024, 64, K*THREADS*sizeof(DistPair<float>)>>>(d_ps, d_results, rows, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  auto end_t = std::chrono::high_resolution_clock::now();

  printf("Total time: %0.2lf\n", GET_CHRONO_TIME(start_t, end_t));
  
  int* h_results = new int[K*rows];
  CUDA_CHECK(cudaMemcpy(h_results, d_results, rows*K*sizeof(int), cudaMemcpyDeviceToHost));
  for(int i=0; i<100; ++i) printf("%d, ", h_results[i*100]);
  printf("%d, %d, ..., %d\n", h_results[0], h_results[1], h_results[rows*K-1]);

  return 0;
}
