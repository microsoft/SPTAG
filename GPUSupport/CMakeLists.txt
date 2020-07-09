# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

find_package(CUDA)
if (CUDA_FOUND)
    message (STATUS "Found cuda.")
    message (STATUS "Include Path:" ${CUDA_INCLUDE_DIRS})
    message (STATUS "Library Path:" ${CUDA_LIBRARIES})
    set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp -std=c++14 -Xptxas -O3 --use_fast_math 
    -gencode arch=compute_70,code=sm_70 
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_50,code=sm_50" )
   
    message (STATUS "CUDA_NVCC_FLAGS:" ${CUDA_NVCC_FLAGS})

    file(GLOB GPUBUILDER_HDR_FILES ${PROJECT_SOURCE_DIR}/AnnService/inc/Core/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Core/Common/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Core/BKT/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Core/KDT/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Helper/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Helper/VectorSetReaders/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/IndexBuilder/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Core/Common/cuda/*)
    file(GLOB GPUBUILDER_FILES ${PROJECT_SOURCE_DIR}/AnnService/src/Core/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Core/Common/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Core/BKT/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Core/KDT/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Helper/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Helper/VectorSetReaders/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/IndexBuilder/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Core/Common/Kernel.cu)

    include_directories(${PROJECT_SOURCE_DIR}/GPUSupport)

    include_directories(${PROJECT_SOURCE_DIR}/AnnService)
    set_source_files_properties(${GPUBUILDER_FILES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

    CUDA_ADD_EXECUTABLE(gpuindexbuilder ${GPUBUILDER_FILES} ${GPUBUILDER_HDR_FILES})
    target_link_libraries(gpuindexbuilder ${Boost_LIBRARIES} ${CUDA_LIBRARIES})
    if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_definitions(gpuindexbuilder PRIVATE -DGPU -DDEBUG)
    else()
        target_compile_definitions(gpuindexbuilder PRIVATE -DGPU)
    endif()
else()
    message (STATUS "Could not find cuda.")
endif()
