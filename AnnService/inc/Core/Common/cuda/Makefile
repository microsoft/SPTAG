ARCH=sm_70

CUDA_HOME='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1'
FLAGS=-std=c++14 -Xptxas -O3
#for windows
COMMON+= -DGPU -I 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include'
LDFLAGS+= -L 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64' -lcuda -lcudart -lcublas -lcurand

all: cuda_unit_tests

cuda_unit_tests: cuda_unit_tests.cu Refine.hxx
	nvcc ${FLAGS} -lineinfo --use_fast_math --gpu-architecture=${ARCH} -I../../../../ -I../../../../../../cub-1.8.0 -I${CUDA_HOME}\\include -o cuda_unit_tests cuda_unit_tests.cu
