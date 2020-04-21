ARCH=sm_70

CUDA_HOME=/usr/local/cuda
FLAGS=-std=c++14 -Xptxas -O3

all: cuda_unit_tests

cuda_unit_tests: cuda_unit_tests.cu Refine.hxx
	nvcc ${FLAGS} -lineinfo --use_fast_math --gpu-architecture=${ARCH} -I../../../../ -I../../../../../../cub-1.8.0 -I${CUDA_HOME}/include -o cuda_unit_tests cuda_unit_tests.cu
