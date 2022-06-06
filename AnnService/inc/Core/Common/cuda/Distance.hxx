/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Licensed under the MIT License.
 */

#ifndef _SPTAG_COMMON_CUDA_DISTANCE_H_
#define _SPTAG_COMMON_CUDA_DISTANCE_H_

#include<cuda.h>
#include<cstdint>
#include<vector>
#include<climits>
#include<float.h>
#include<unordered_set>
#include<chrono>

#include "params.h"
#include "inc/Core/VectorIndex.h"
#include "GPUQuantizer.hxx"

using namespace std;

using namespace SPTAG;


// Templated infinity value
template<typename T> __host__ __device__ T INFTY() {}
template<> __forceinline__ __host__ __device__ int INFTY<int>() {return INT_MAX;}
template<> __forceinline__ __host__ __device__ long long int INFTY<long long int>() {return LLONG_MAX;}
template<> __forceinline__ __host__ __device__ float INFTY<float>() {return FLT_MAX;}
//template<> __forceinline__ __host__ __device__ __half INFTY<__half>() {return FLT_MAX;}
template<> __forceinline__ __host__ __device__ uint8_t INFTY<uint8_t>() {return 255;}

template<typename T> __device__ T BASE() {}
template<> __forceinline__ __device__ float BASE<float>() {return 1;}
template<> __forceinline__ __device__ uint32_t BASE<uint32_t>() {return 16384;}

template<typename T, typename SUMTYPE, int Dim>
__forceinline__ __device__ SUMTYPE cosine(T* a, T* b) {
    SUMTYPE total[2]={0,0};
    for(int i=0; i<Dim; i+=2) {
      total[0] += ((SUMTYPE)(a[i] * b[i]));
      total[1] += ((SUMTYPE)(a[i+1] * b[i+1]));
    }
    return BASE<SUMTYPE>()-(total[0]+total[1]);
}

template<typename T, typename SUMTYPE, int Dim>
__forceinline__ __device__ SUMTYPE l2(T* aVec, T* bVec) {
  SUMTYPE total[2]={0,0};
  for(int i=0; i<Dim; i+=2) {
    total[0] += (aVec[i]-bVec[i])*(aVec[i]-bVec[i]);
    total[1] += (aVec[i+1]-bVec[i+1])*(aVec[i+1]-bVec[i+1]);
  }
  return total[0]+total[1];

}

template<typename T, typename SUMTYPE, int Dim, int metric>
__device__ SUMTYPE dist(T* a, T* b) {
  if(metric == (int)DistMetric::Cosine) {
    return cosine<T,SUMTYPE,Dim>(a, b);
  }
  else {
    return l2<T,SUMTYPE,Dim>(a,b);
  }
}

template<typename T, typename SUMTYPE>
__forceinline__ __device__ bool violatesRNG(T* a, T* b, SUMTYPE dist, SUMTYPE (*comp)(T*, T*)) {
  SUMTYPE between;
  between = comp(a, b);
  return between <= dist;
}

template<typename T>
class PointSet {
  public:
    int dim;
    T* data;
    DistMetric metric;

  __forceinline__ __device__ T* getVec(size_t idx) {
    return &(data[idx*dim]);
  }

};

#define COPY_BATCH_SIZE 10000
template<typename T>
__host__ void copyRawDataToMultiGPU(SPTAG::VectorIndex* index, T** d_data, size_t dataSize, int dim, int NUM_GPUS, cudaStream_t* streams) {
  T* samplePtr;
  T* temp = new T[COPY_BATCH_SIZE*dim];
  size_t copy_size=COPY_BATCH_SIZE;

  for(size_t batch_start = 0; batch_start < dataSize; batch_start += COPY_BATCH_SIZE) {
    if(batch_start + copy_size > dataSize) {
      copy_size = dataSize - batch_start;
    }
    for(int i=0; i<copy_size; ++i) {
      samplePtr = (T*)(index->GetSample(batch_start + i));
      for(int j=0; j<dim; ++j) {
        temp[i*dim+j] = samplePtr[j];
      }
    }
    for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
      CUDA_CHECK(cudaSetDevice(gpuNum));
      CUDA_CHECK(cudaMemcpy(d_data[gpuNum]+(batch_start*dim), temp, copy_size*dim * sizeof(T), cudaMemcpyHostToDevice));
    }
  }
  for(int gpuNum=0; gpuNum < NUM_GPUS; ++gpuNum) {
    CUDA_CHECK(cudaStreamSynchronize(streams[gpuNum]));
  }
  delete temp;
}

/*********************************************************************
* DEPRICATED CODE - Old data structures used before code restructure
*********************************************************************/

/*********************************************************************
* Object representing a Dim-dimensional point, with each coordinate
* represented by a element of datatype T
* NOTE: Dim must be templated so that we can store coordinate values in registers
*********************************************************************/
/*
template<typename T, typename SUMTYPE, int Dim>
class Point {
  public:
    int id;
    T* coords;

  __host__ void load(vector<T> data) {
    for(int i=0; i<Dim; i++) {
      coords[i] = data[i];
    }
  }

  __host__ __device__ void loadChunk(T* data, int exact_dim) {
    for(int i=0; i<exact_dim; i++) {
      coords[i] = data[i];
    }
    for(int i=exact_dim; i<Dim; i++) {
      coords[i] = 0;
    }
  }

  __host__ __device__ Point& operator=( const Point& other ) {
    for(int i=0; i<Dim; i++) {
      coords[i] = other.coords[i];
    }
    id = other.id;
    return *this;
  }

  // Computes euclidean dist.  Uses 2 registers to increase pipeline efficiency and ILP
  __device__ __host__ SUMTYPE l2(Point<T,SUMTYPE,Dim>* other) {
    SUMTYPE total[2]={0,0};

    for(int i=0; i<Dim; i+=2) {
      total[0] += (coords[i]-other->coords[i])*(coords[i]-other->coords[i]);
      total[1] += (coords[i+1]-other->coords[i+1])*(coords[i+1]-other->coords[i+1]);
    }
    return total[0]+total[1];
  }

  __device__ SUMTYPE l2_block(Point<T,SUMTYPE,Dim>* other) {
    SUMTYPE total=0;
    __shared__ SUMTYPE final_val;
    final_val=0;
    __syncthreads();

    for(int i=threadIdx.x; i<Dim; i+=blockDim.x) {
      total += (coords[i]-other->coords[i])*(coords[i]-other->coords[i]);
    }
    atomicAdd(&final_val, total);
    __syncthreads();
    return final_val;
  }


  // Computes Cosine dist.  Uses 2 registers to increase pipeline efficiency and ILP
  // Assumes coordinates are normalized so each vector is of unit length.  This lets us
  // perform a dot-product instead of the full cosine distance computation.
//  __device__ SUMTYPE cosine(Point<T,SUMTYPE,Dim>* other, bool test) {return NULL;}
  __device__ SUMTYPE cosine(Point<T,SUMTYPE,Dim>* other) {
    SUMTYPE total[2]={0,0};

    for(int i=0; i<Dim; i+=2) {
      total[0] += ((SUMTYPE)((SUMTYPE)coords[i] * (SUMTYPE)other->coords[i]));
      total[1] += ((SUMTYPE)((SUMTYPE)coords[i+1] * (SUMTYPE)other->coords[i+1]));
    }
    return (SUMTYPE)1.0 - (total[0]+total[1]);
  }

  __forceinline__ __device__ SUMTYPE dist(Point<T,SUMTYPE,Dim>* other, int metric) {
    if(metric == 0) return l2(other);
    else return cosine(other);
  }
};

// Less-than operator between two points.
template<typename T, typename SUMTYPE, int Dim>
__host__ __device__ bool operator<(const Point<T,SUMTYPE,Dim>& first, const Point<T,SUMTYPE,Dim>& other) {
  return first.id < other.id;
}


// Specialized version of Point structure for 1-byte datatype (int8)
// Packs coordinates into Dim/4 total integer values, and functions un-pack as needed
template<typename SUMTYPE, int Dim>
class Point<uint8_t, SUMTYPE, Dim> {
  public:
    int id;
//    uint32_t* coords;
    uint8_t* coords;

  __host__ void load(vector<uint8_t> data) {
    for(int i=0; i<Dim/4; i++) {
      coords[i] = 0;
      for(int j=0; j<4; j++) {
        coords[i] += (data[i*4 + j] << (j*8));
      }
    }
  }

  __host__ __device__ void loadChunk(uint8_t* data, int exact_dims) {
    for(int i=0; i<exact_dims/4; i++) {
      coords[i] = 0;
      for(int j=0; j<4; j++) {
        coords[i] += (data[i*4 + j] << (j*8));
      }
    }
    for(int i=exact_dims/4; i<Dim/4; i++) {
      coords[i]=0;
    }
  }


  __host__ __device__ Point& operator=( const Point& other ) {
    for(int i=0; i<Dim/4; i++) {
      coords[i] = other.coords[i];
    }
    id = other.id;
    return *this;
  }

  __host__ uint8_t getVal(int idx) {
    if(idx % 4 == 0) {
      return (uint8_t)(coords[idx/4] & 0x000000FF);
    }
    else if(idx % 4 == 1) {
      return (uint8_t)((coords[idx/4] & 0x0000FF00)>>8);
    }
    else if(idx % 4 == 2) {
      return (uint8_t)((coords[idx/4] & 0x00FF0000)>>16);
    }
    else if(idx % 4 == 3) {
      return (uint8_t)((coords[idx/4])>>24);
    }
    return 0;
  }

  __device__ __host__ SUMTYPE l2_block(Point<uint8_t,SUMTYPE,Dim>* other) {return 0;}
  __device__ __host__ SUMTYPE l2(Point<uint8_t,SUMTYPE,Dim>* other) {

    SUMTYPE totals[4] = {0,0,0,0};
    SUMTYPE temp[4];
    SUMTYPE temp_other[4];

    for(int i=0; i<Dim/4; ++i) {
      temp[0] = (coords[i] & 0x000000FF);
      temp_other[0] = (other->coords[i] & 0x000000FF);

      temp[1] = (coords[i] & 0x0000FF00) >> 8;
      temp_other[1] = (other->coords[i] & 0x0000FF00) >> 8;

      temp[2] = (coords[i] & 0x00FF0000) >> 16;
      temp_other[2] = (other->coords[i] & 0x00FF0000) >> 16;

      temp[3] = (coords[i]) >> 24;
      temp_other[3] = (other->coords[i])>> 24;

      totals[0] += (temp[0]-temp_other[0])*(temp[0]-temp_other[0]);
      totals[1] += (temp[1]-temp_other[1])*(temp[1]-temp_other[1]);
      totals[2] += (temp[2]-temp_other[2])*(temp[2]-temp_other[2]);
      totals[3] += (temp[3]-temp_other[3])*(temp[3]-temp_other[3]);
    }
    return totals[0]+totals[1]+totals[2]+totals[3];
  }

#if __CUDA_ARCH__ > 610  // Use intrinsics if available for GPU being compiled for

  // With int8 datatype, values are packed into integers so they need to be
  // unpacked while computing distance
  __device__ SUMTYPE cosine(Point<uint8_t,SUMTYPE,Dim>* other) {
    uint32_t prod=0;
    uint32_t src=0;
    uint32_t target=0;

    for(int i=0; i<Dim/4; ++i) {
      src = coords[i];
      target = other->coords[i];
      prod = __dp4a(src, target, prod);
    }

    return ((SUMTYPE)65536) - prod;
  }

#else
  __device__ SUMTYPE cosine(Point<uint8_t,SUMTYPE,Dim>* other) {
    SUMTYPE prod[4];
    prod[0]=0;

    for(int i=0; i<Dim/4; ++i) {
      prod[0] += (coords[i] & 0x000000FF)*(other->coords[i] & 0x000000FF);
      prod[1] = ((coords[i] & 0x0000FF00) >> 8)*((other->coords[i] & 0x0000FF00) >> 8);
      prod[2] = ((coords[i] & 0x00FF0000) >> 16)*((other->coords[i] & 0x00FF0000) >> 16);
      prod[3] = ((coords[i]) >> 24)*((other->coords[i]) >> 24);

      prod[0] += prod[1]+prod[2]+prod[3];
    }

    return ((SUMTYPE)65536) - prod[0];
  }
#endif

  __forceinline__ __device__ SUMTYPE dist(Point<uint8_t,SUMTYPE,Dim>* other, int metric) {
    if(metric == 0) return l2(other);
    else return cosine(other);
  }

};

// Specialized version of Point structure for SIGNED 1-byte datatype (int8)
// Packs coordinates into Dim/4 total integer values, and functions un-pack as needed
template<typename SUMTYPE, int Dim>
class Point<int8_t, SUMTYPE, Dim> {
  public:
    int id;
//    uint32_t* coords;
    int8_t* coords;

  __host__ void load(vector<int8_t> data) {

    uint8_t* test = reinterpret_cast<uint8_t*>(data.data());
    for(int i=0; i<Dim/4; i++) {
      coords[i] = 0;
      for(int j=0; j<4; j++) {
        coords[i] += ((test[i*4 + j]) << (j*8));
      }
    }
  }

  __host__ __device__ void loadChunk(int8_t* data, int exact_dims) {

    uint8_t* test = reinterpret_cast<uint8_t*>(data);
    for(int i=0; i<exact_dims/4; i++) {
      coords[i] = 0;
      for(int j=0; j<4; j++) {
        coords[i] += (test[i*4 + j] << (j*8));
      }
    }
    for(int i=exact_dims/4; i<Dim/4; i++) {
      coords[i]=0;
    }
  }

  __host__ int8_t getVal(int idx) {
    if(idx % 4 == 0) {
      return (int8_t)(coords[idx/4] & 0x000000FF);
    }
    else if(idx % 4 == 1) {
      return (int8_t)((coords[idx/4] & 0x0000FF00)>>8);
    }
    else if(idx % 4 == 2) {
      return (int8_t)((coords[idx/4] & 0x00FF0000)>>16);
    }
    else if(idx % 4 == 3) {
      return (int8_t)((coords[idx/4])>>24);
    }
    return 0;
  }

  __host__ __device__ Point& operator=( const Point& other ) {
    for(int i=0; i<Dim/4; i++) {
      coords[i] = other.coords[i];
    }
    id = other.id;
    return *this;
  }
  __device__ __host__ SUMTYPE l2_block(Point<int8_t,SUMTYPE,Dim>* other) {return 0;}
  __device__ __host__ SUMTYPE l2(Point<int8_t,SUMTYPE,Dim>* other) {

    SUMTYPE totals[4] = {0,0,0,0};
    int32_t temp[4];
    int32_t temp_other[4];

    for(int i=0; i<Dim/4; ++i) {
      temp[0] = (int8_t)(coords[i] & 0x000000FF);
      temp_other[0] = (int8_t)(other->coords[i] & 0x000000FF);

      temp[1] = (int8_t)((coords[i] & 0x0000FF00) >> 8);
      temp_other[1] = (int8_t)((other->coords[i] & 0x0000FF00) >> 8);

      temp[2] = (int8_t)((coords[i] & 0x00FF0000) >> 16);
      temp_other[2] = (int8_t)((other->coords[i] & 0x00FF0000) >> 16);

      temp[3] = (int8_t)((coords[i]) >> 24);
      temp_other[3] = (int8_t)((other->coords[i])>> 24);

      totals[0] += (temp[0]-temp_other[0])*(temp[0]-temp_other[0]);
      totals[1] += (temp[1]-temp_other[1])*(temp[1]-temp_other[1]);
      totals[2] += (temp[2]-temp_other[2])*(temp[2]-temp_other[2]);
      totals[3] += (temp[3]-temp_other[3])*(temp[3]-temp_other[3]);
    }
    return totals[0]+totals[1]+totals[2]+totals[3];
  }


#if __CUDA_ARCH__ > 610  // Use intrinsics if available for GPU being compiled for
  // With int8 datatype, values are packed into integers so they need to be
  // unpacked while computing distance
  __device__ SUMTYPE cosine(Point<int8_t,SUMTYPE,Dim>* other) {
    int32_t prod=0;
    int32_t src=0;
    int32_t target=0;

    for(int i=0; i<Dim/4; ++i) {
      src = coords[i];
      target = other->coords[i];
      prod = __dp4a(src, target, prod);
    }

    return ((SUMTYPE)16384) - (SUMTYPE)prod;
  }

#else
  __device__ SUMTYPE cosine(Point<int8_t,SUMTYPE,Dim>* other) {
    SUMTYPE prod[4];
    prod[0]=0;

    for(int i=0; i<Dim/4; ++i) {
      prod[0] += ((int8_t)(coords[i] & 0x000000FF))*((int8_t)(other->coords[i] & 0x000000FF));
      prod[1] = ((int8_t)((coords[i] & 0x0000FF00) >> 8))*((int8_t)((other->coords[i] & 0x0000FF00) >> 8));
      prod[2] = ((int8_t)((coords[i] & 0x00FF0000) >> 16))*((int8_t)((other->coords[i] & 0x00FF0000) >> 16));
      prod[3] = ((int8_t)((coords[i]) >> 24))*((int8_t)((other->coords[i]) >> 24));
      prod[0] += prod[1]+prod[2]+prod[3];
    }

    return ((SUMTYPE)1) - prod[0];
  }
#endif

  __forceinline__ __device__ SUMTYPE dist(Point<int8_t,SUMTYPE,Dim>* other, int metric) {
    if(metric == 0) return l2(other);
    else return cosine(other);
  }

};
*/

/*********************************************************************
 * Create an array of Point structures out of an input array
 ********************************************************************/
/*
template<typename T, typename SUMTYPE, int Dim>
__host__ Point<T, SUMTYPE, Dim>* assignCoords(T* d_data, int rows, int dim) {
  Point<T,SUMTYPE,Dim>* pointArray = new Point<T,SUMTYPE,Dim>[rows];

  for(size_t i=0; i<rows; ++i) {
    pointArray[i].coords = d_data+(i*dim);  // d_data points to data in device memory
    pointArray[i].id = i;
  }
  return pointArray;
}


template<typename T, typename SUMTYPE, int Dim>
__host__ Point<T, SUMTYPE, Dim>* convertMatrix(T* data, int rows, int exact_dim) {
//  Point<T,SUMTYPE,Dim>* pointArray = (Point<T,SUMTYPE,Dim>*)malloc(rows*sizeof(Point<T,SUMTYPE,Dim>));
  Point<T,SUMTYPE,Dim>* pointArray = new Point<T,SUMTYPE,Dim>[rows];
  for(int i=0; i<rows; i++) {
    pointArray[i].loadChunk(&data[i*exact_dim], exact_dim);
  }
  return pointArray;
} 

template<typename T, typename SUMTYPE, int Dim>
__host__ Point<T, SUMTYPE, Dim>* convertMatrix(SPTAG::VectorIndex* index, size_t rows, int exact_dim) {
//  Point<T,SUMTYPE,Dim>* pointArray = (Point<T,SUMTYPE,Dim>*)malloc(rows*sizeof(Point<T,SUMTYPE,Dim>));
  Point<T,SUMTYPE,Dim>* pointArray = new Point<T,SUMTYPE,Dim>[rows];
  printf("allocated %d points in convertMatrix\n");

  T* data;

  for(int i=0; i<rows; i++) {
    data = (T*)index->GetSample(i);

    pointArray[i].loadChunk(data, exact_dim);
  }
  return pointArray;
}

template<typename T, typename SUMTYPE, int Dim>
__host__ void extractHeadPoints(T* data, Point<T,SUMTYPE,Dim>* headPoints, size_t totalRows, std::unordered_set<int> headVectorIDS, int exact_dim) {
    int headIdx=0;
    for(size_t i=0; i<totalRows; i++) {
        if(headVectorIDS.count(i) != 0) {
            headPoints[headIdx].loadChunk(&data[i*exact_dim], exact_dim);
            headPoints[headIdx].id = i;
            headIdx++;
        }
    }
}

template<typename T, typename SUMTYPE, int Dim>
__host__ void extractTailPoints(T* data, Point<T,SUMTYPE,Dim>* tailPoints, int totalRows, std::unordered_set<int> headVectorIDS, int exact_dim) {
    int tailIdx=0;
    for(size_t i=0; i<totalRows; i++) {
        if(headVectorIDS.count(i) == 0) {
            tailPoints[tailIdx].loadChunk(&data[i*exact_dim], exact_dim);
            tailPoints[tailIdx].id = i;
            tailIdx++;
        }
    }
}

template<typename T, typename SUMTYPE, int Dim>
__host__ void extractFullVectorPoints(T* data, Point<T,SUMTYPE,Dim>* tailPoints, size_t totalRows, int exact_dim) {

    for(size_t i=0; i<totalRows; ++i) {
        tailPoints[i].loadChunk(&data[i*exact_dim], exact_dim);
        tailPoints[i].id = i;
    }
}

template<typename T, typename SUMTYPE, int Dim>
__host__ void extractHeadPointsFromIndex(T* data, SPTAG::VectorIndex* headIndex, Point<T,SUMTYPE,Dim>* headPoints, int exact_dim) {
    size_t headRows = headIndex->GetNumSamples();

    for(size_t i=0; i<headRows; ++i) {
        headPoints[i].loadChunk((T*)headIndex->GetSample(i), exact_dim);
        headPoints[i].id = i;
    }
}

*/

#endif
