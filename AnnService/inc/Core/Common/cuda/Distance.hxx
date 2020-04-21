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

#include<cstdint>
#include<vector>
#include<climits>
#include<float.h>

using namespace std;

// Templated infinity value
template<typename T> __host__ __device__ T INFTY() {}
template<> __forceinline__ __host__ __device__ int INFTY<int>() {return INT_MAX;}
template<> __forceinline__ __host__ __device__ long long int INFTY<long long int>() {return LLONG_MAX;}
template<> __forceinline__ __host__ __device__ float INFTY<float>() {return FLT_MAX;}

/*********************************************************************
* Object representing a Dim-dimensional point, with each coordinate
* represented by a element of datatype T
* NOTE: Dim must be templated so that we can store coordinate values in registers
*********************************************************************/
template<typename T, typename SUMTYPE, int Dim>
class Point {
  public:
    int id;
    T coords[Dim];

 __host__ void load(vector<T> data) {
    for(int i=0; i<Dim; i++) {
      coords[i] = data[i];
    }
  }

  __host__ void loadChunk(T* data, int exact_dim) {
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

  // Computes Cosine dist.  Uses 2 registers to increase pipeline efficiency and ILP
  // Assumes coordinates are normalized so each vector is of unit length.  This lets us
  // perform a dot-product instead of the full cosine distance computation.
  __device__ SUMTYPE cosine(Point<T,SUMTYPE,Dim>* other) {
    SUMTYPE total[2]={0,0};

    for(int i=0; i<Dim; i+=2) {
      total[0] += ((float)(coords[i] * other->coords[i]));
      total[1] += ((float)(coords[i+1] * other->coords[i+1]));
    }
    return 1.0 - (total[0]+total[1]);
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
    uint32_t coords[Dim/4];

  __host__ void load(vector<uint8_t> data) {
    for(int i=0; i<Dim/4; i++) {
      coords[i] = 0;
      for(int j=0; j<4; j++) {
        coords[i] += (data[i*4 + j] << (j*8));
      }
    }
  }

  __host__ void loadChunk(uint8_t* data, int exact_dims) {
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

  // Cosine distance measure for uint8_t datatype is more complex because it cannot be normalized.
  // So, we have to perform the formal computation, not just dot-product
  __device__ SUMTYPE cosine(Point<uint8_t,SUMTYPE,Dim>* other) {
    SUMTYPE prod[4];
    SUMTYPE a[4];
    SUMTYPE b[4];
    prod[0]=0; a[0]=0; b[0]=0;

    for(int i=0; i<Dim/4; ++i) {
      a[0] += (coords[i] & 0x000000FF)*(coords[i] & 0x000000FF);
      a[1] = ((coords[i] & 0x0000FF00) >> 8)*((coords[i] & 0x0000FF00) >> 8);
      a[2] = ((coords[i] & 0x00FF0000) >> 16)*((coords[i] & 0x00FF0000) >> 16);
      a[3] = ((coords[i]) >> 24)*((coords[i]) >> 24);

      b[0] += (other->coords[i] & 0x000000FF)*(other->coords[i] & 0x000000FF);
      b[1] = ((other->coords[i] & 0x0000FF00) >> 8)*((other->coords[i] & 0x0000FF00) >> 8);
      b[2] = ((other->coords[i] & 0x00FF0000) >> 16)*((other->coords[i] & 0x00FF0000) >> 16);
      b[3] = ((other->coords[i]) >> 24)*((other->coords[i]) >> 24);

      prod[0] += (coords[i] & 0x000000FF)*(other->coords[i] & 0x000000FF);
      prod[1] = ((coords[i] & 0x0000FF00) >> 8)*((other->coords[i] & 0x0000FF00) >> 8);
      prod[2] = ((coords[i] & 0x00FF0000) >> 16)*((other->coords[i] & 0x00FF0000) >> 16);
      prod[3] = ((coords[i]) >> 24)*((other->coords[i]) >> 24);

      a[0] += a[1]+a[2]+a[3];
      b[0] += b[1]+b[2]+b[3];
      prod[0] += prod[1]+prod[2]+prod[3];
    }

    return 1-(prod[0] / (sqrt((float)a[0]*b[0])));
  }

};

/*********************************************************************
 * Create an array of Point structures out of an input array
 ********************************************************************/
template<typename T, typename SUMTYPE, int Dim>
__host__ Point<T, SUMTYPE, Dim>* convertMatrix(T* data, int rows, int exact_dim) {
  Point<T,SUMTYPE,Dim>* pointArray = (Point<T,SUMTYPE,Dim>*)malloc(rows*sizeof(Point<T,SUMTYPE,Dim>));
  for(int i=0; i<rows; i++) {
    pointArray[i].loadChunk(&data[i*exact_dim], exact_dim);
  }
  return pointArray;
} 


#endif
