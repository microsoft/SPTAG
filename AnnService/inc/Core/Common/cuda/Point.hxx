/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
 */

#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <assert.h>
#include<cuda.h>

#include "log.hxx"
#include "gpu_ThreadHeap.hxx"

// file deliminator
constexpr char sep = '|';

/************************************************************************************
 Read in a data file made up of rows and columns.  Entries separated by 'sep' described
 above and each new row is terminated by a newline.
************************************************************************************/
template<typename T>
int read_tsv(vector<std::string>* labels, vector<vector<T>>* data, const char* fname, const char sep) {
  constexpr size_t MAX_LINE_LEN=1000000; // Max elements per line
  char line[MAX_LINE_LEN];

  FILE* fp = fopen(fname, "r");
  if (fp == NULL) {

    std::cerr<<"ERROR: opening file "<<fname<<"!"<<std::endl;

    return 1;
  }

  while(fgets(line, MAX_LINE_LEN, fp)) {
    vector<T> newRow;
    std::stringstream row(line);

    std::string seg;

    std::getline(row,seg,'\t');
    (*labels).push_back(seg);

    while(std::getline(row,seg,sep)) {
      newRow.push_back(std::stof(seg));
    }
    (*data).push_back(newRow);
  }
  return 0;
}


#define SizeType int
#define DimensionType int
/***************************************************************************************
 * Write KNN graph to a binary file.  
 * Writes the Ids of the K nearest neighbor for each point.
***************************************************************************************/
void write_result_to_binary(int* results, int KVAL, long long int N, std::string outname) {
  ofstream out_file;
  out_file.open (outname.c_str(), std::ios::binary);

  if(!out_file.is_open()) {
    LOG_CRIT("Failed to open output file\n");
    exit(1);
  }

  out_file.write((char*)&N, sizeof(SizeType));
  out_file.write((char*)&KVAL, sizeof(DimensionType));
  out_file.write((char*)results, (long long int)sizeof(int)*N*KVAL);

  out_file.close();
}

/***************************************************************************************
 * Loads a KNN graph from a binary file.
 * Expects N, K, followed by a sequence of integers representing the KNN graph
***************************************************************************************/
void load_result_from_binary(int* graph, int KVAL, long long int N, string file_name) {

  std::ifstream input(file_name, std::ios::binary);
  if(!input.is_open()) {
    LOG_CRIT("Failed to open input file\n");
    exit(1);
  }

  // Check to make sure that N and K are what we expect
  SizeType rows;
  DimensionType cols;
  input.read((char*)&rows, sizeof(SizeType));
  if(rows != N) {
    LOG_CRIT("Rows listed in graph file incorrect.  Expected:%lld, found:%d\n", N, rows);
    exit(1);
  }
  input.read((char*)&cols, sizeof(DimensionType));
  if(cols != KVAL) {
    LOG_CRIT("K listed in graph file incorrect.  Expected:%d, found:%d\n", KVAL, cols);
    exit(1);
  }

  input.read((char*)graph, (long long int)N*KVAL*sizeof(int)); 
  input.close();
}


/*********************************************************************
* Object representing a Dim-dimensional point, with each coordinate
* represented by a element of datatype T
*********************************************************************/
template<typename T, int Dim>
class Point {
  public:
    int id;
    T coords[Dim];


 __host__ void load(vector<T> data) {
    for(int i=0; i<Dim; i++) {
      coords[i] = data[i];
    }
  }

  __host__ __device__ Point& operator=( const Point& other ) {
    for(int i=0; i<Dim; i++) {
      coords[i] = other.coords[i]; 
    } 
    id = other.id;
    return *this;
  }


  __host__ __device__ Point& operator>(const Point& other) {
    return id > other.id;
  }

  /*********************************************************************
  * Methods to compute the distance between two points.  NOT used by
  * optimized versions that use transposed points (class and methods below).
  *********************************************************************/
  __device__ __host__ float l2_dist(Point* other) {
    float total=0.0;
    for(int i=0; i<Dim; i++) {
      total += (coords[i]-other->coords[i])*(coords[i]-other->coords[i]);
    }
    return sqrt(total);
  }

  __device__ __host__ SUMTYPE l2_dist_sq(Point* other) {
    SUMTYPE total=0.0;

    if(other->id == id) {
      return INFTY;
    }
    for(int i=0; i<Dim; i++) {
      total += (coords[i]-other->coords[i])*(coords[i]-other->coords[i]);
    }
    return total;
  }

  __device__ __host__ SUMTYPE l2_partial(Point* other, int laneId) {
    SUMTYPE total=0.0;

    for(int i=laneId; i<Dim; i+=32) {
      total += (coords[i]-other->coords[i])*(coords[i]-other->coords[i]);
    }
    return total;
  }
};

template<typename T, int Dim>
__host__ __device__ bool operator<(const Point<T,Dim>& first, const Point<T,Dim>& other) {
  return first.id < other.id;
}

  /*********************************************************************
  * Convert a vector of vectors (matrix) into an array of Points
  *********************************************************************/
template<typename T, int Dim>
__host__ Point<T, Dim>* convertMatrix(std::vector<std::vector<T>> mtx) {
  Point<T,Dim>* pointArray = (Point<T,Dim>*)malloc(mtx.size()*sizeof(Point<T,Dim>));
  for(int i=0; i<mtx.size(); i++) {
    pointArray[i].load(mtx[i]);
  }
  return pointArray;
} 


/************************************************************************
 * Wrapper around shared memory holding transposed points to avoid 
 * bank conflicts. 
 ************************************************************************/
template<typename T, int Dim, int Stride>
class TransposePoint {
  public:
    T* dataPtr;

  __device__ void setMem(T* ptr) {
    dataPtr = ptr;
  }

  // Load regular point into memory transposed
  __device__ void loadPoint(Point<T,Dim> p) {
    for(int i=0; i<Dim; i++) {
      dataPtr[i*Stride] = p.coords[i];
    }
  }

  // Return the idx-th coordinate of the transposed vector
  __forceinline__ __device__ T getCoord(int idx) {
    return dataPtr[idx*Stride];
  }

  /******************************************************************************************
  * Various methods used to compute euclidean distance between two Dim-dimensional vectors. 
  *
  * Vector a is transposed so that consecutive entries are Stride distance apart.  Target being
  * compared with is NOT transposed...
  * NOTE: Avoid using squareroots, allows us to use integers rather than floating point
  ******************************************************************************************/

  /******************************************************************************************
  * Main L2 distance metric used by approx-KNN application.
  ******************************************************************************************/
  __forceinline__ __device__ __host__ SUMTYPE l2(Point<T,Dim>* other) {
    SUMTYPE total=0;
#pragma unroll
    for(int i=0; i<Dim; ++i) {
      total += ((getCoord(i)-other->coords[i])*(getCoord(i)-other->coords[i]));
    }

    return total;
  }
  
  /******************************************************************************************
  * Version of L2 distance metric that uses ILP to try and improve performance
  ******************************************************************************************/
  __forceinline__ __device__ __host__ SUMTYPE l2_ILP(Point<T,Dim>* other) {
    SUMTYPE totals[ILP];
  #pragma unroll
    for(int i=0; i<ILP; ++i)
      totals[i]=0.0;

    for(int i=0; i<Dim-ILP+1; i+=ILP) {

  #pragma unroll
      for(int l=0; l<ILP; ++l) {
      totals[l] += ((getCoord(i+l)-other->coords[i+l])*(getCoord(i+l)-other->coords[i+l]));
      }
    }

  #pragma unroll
    for(int i=1; i<ILP; ++i)
      totals[0]+=totals[i];
  
    return totals[0];
  }


  /******************************************************************************************
  * Cosine distance metric comparison operation.  Requires that the SUMTYPE is floating point,
  * regardless of the datatype T, because it requires squareroot.
  ******************************************************************************************/
  __forceinline__ __device__ __host__ SUMTYPE cosine(Point<T,Dim>* other) {
    SUMTYPE prod = 0;
    SUMTYPE a = 0;
    SUMTYPE b=0;
#pragma unroll
    for(int i=0; i<Dim; ++i) {
      a += getCoord(i)*getCoord(i);
      b += other->coords[i]*other->coords[i];
      prod += getCoord(i)*other->coords[i];
    }

    return 1-(prod / (sqrt(a*b)));
  }


  /******************************************************************************************
  * Depricated versions of L2 distance metrics below
  ******************************************************************************************/
  __forceinline__ __device__ __host__ SUMTYPE l2_partial_dist(Point<T,Dim>* other) {
    SUMTYPE total=0;
#pragma unroll
    for(int i=0; i<Dim; ++i) {
      total += ((getCoord(i)-other->coords[i])*(getCoord(i)-other->coords[i]));
    }

    return total;
  }


  __forceinline__ __device__ __host__ SUMTYPE cosine_no_unroll(Point<T,Dim>* other) {
    SUMTYPE prod = 0;
    SUMTYPE a = 0;
    SUMTYPE b=0;
    for(int i=0; i<Dim; ++i) {
      a += getCoord(i)*getCoord(i);
      b += other->coords[i]*other->coords[i];
      prod += getCoord(i)*other->coords[i];
    }

    // 1- because its similarity, not distance?!

    return 1-(prod / (sqrt(a*b)));
  }
};

  
  /******************************************************************************************
  * Simple euclidean distance calculation used for correctness check
  ******************************************************************************************/
template<typename T, int Dim>
__host__ float l2_dist(Point<T,Dim> a, Point<T,Dim> b) {
  float total=0.0;
  float dist;
  for(int i=0; i<Dim; i++) {
    dist = __half2float(a.coords[i])-__half2float(b.coords[i]);
    total += dist*dist;
//    total += __half2float((a.coords[i]-b.coords[i])*(a.coords[i]-b.coords[i]));
  }
  return sqrt((total));
}


// Depricated - Used by previous brute-force implementation.
template<typename T, int Dim>
__forceinline__ __device__ __host__ SUMTYPE l2(T* query, Point<T,Dim>* other) {
  SUMTYPE total=0;

  for(int i=0; i<Dim; ++i) {
    total += (query[i]-other->coords[i])*(query[i]-other->coords[i]);
  }

  return total;
}
