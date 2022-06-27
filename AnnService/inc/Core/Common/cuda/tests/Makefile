ARCH=sm_70

CUDA_HOME=/usr/local/cuda
FLAGS=-std=c++14 -Xptxas -O3

all: test_dists

test_dists: cuda_unit_tests.cu ../KNN.hxx ../Distance.hxx
	nvcc ${FLAGS} -lineinfo --use_fast_math --gpu-architecture=${ARCH} -I${CUDA_HOME}/include -I ../../../../../  -o test_dists cuda_unit_tests.cu
