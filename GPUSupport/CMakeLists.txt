# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

find_package(CUDA)

if (NOT GPU)
    set (CUDA_FOUND false)
endif()

if (CUDA_FOUND)
    message (STATUS "Found cuda.")
    message (STATUS "Include Path:" ${CUDA_INCLUDE_DIRS})
    message (STATUS "Library Path:" ${CUDA_LIBRARIES})
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC -Xcompiler -fopenmp -std=c++14 -Xptxas -O3 --use_fast_math --disable-warnings -lineinfo
        -gencode arch=compute_70,code=sm_70
        -gencode arch=compute_61,code=sm_61
        -gencode arch=compute_60,code=sm_60" )
    elseif(WIN32)
        set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler /openmp -Xcompiler /std:c++14 -Xcompiler /Zc:__cplusplus -Xcompiler /FS --use_fast_math
        -gencode arch=compute_70,code=sm_70
        -gencode arch=compute_61,code=sm_61
        -gencode arch=compute_60,code=sm_60" )
    endif()

    message (STATUS "CUDA_NVCC_FLAGS:" ${CUDA_NVCC_FLAGS})

    set(AnnService ${PROJECT_SOURCE_DIR}/AnnService)

    include_directories(${AnnService})

    file(GLOB_RECURSE GPU_HDR_FILES ${AnnService}/inc/Core/*.h  ${AnnService}/inc/Helper/*.h ${AnnService}/inc/Core/Common/cuda/*)
    file(GLOB_RECURSE GPU_SRC_FILES ${AnnService}/src/Core/*.cpp ${AnnService}/src/Helper/*.cpp ${AnnService}/src/Core/Common/Kernel.cu)

    list(REMOVE_ITEM GPU_HDR_FILES
        ${AnnService}/inc/Core/Common/DistanceUtils.h
        ${AnnService}/inc/Core/Common/InstructionUtils.h
        ${AnnService}/inc/Core/Common/CommonUtils.h
        ${AnnService}/inc/Core/Common/cuda/cuda_unit_tests.cu
    )
    list(REMOVE_ITEM GPU_SRC_FILES
        ${AnnService}/src/Core/Common/DistanceUtils.cpp
        ${AnnService}/src/Core/Common/InstructionUtils.cpp
    )

    set_source_files_properties(${GPU_SRC_FILES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        set(Definition -DGPU -DDEBUG)
    else()
        set(Definition -DGPU)
    endif()

    CUDA_ADD_LIBRARY(GPUSPTAGLib SHARED ${GPU_SRC_FILES} ${GPU_HDR_FILES})
    target_link_libraries(GPUSPTAGLib DistanceUtils ${Boost_LIBRARIES} ${CUDA_LIBRARIES})
    target_compile_definitions(GPUSPTAGLib PRIVATE ${Definition})

    CUDA_ADD_LIBRARY(GPUSPTAGLibStatic STATIC ${GPU_SRC_FILES} ${GPU_HDR_FILES})
    target_link_libraries(GPUSPTAGLibStatic DistanceUtils ${Boost_LIBRARIES} ${CUDA_LIBRARIES})
    target_compile_definitions(GPUSPTAGLibStatic PRIVATE ${Definition})
    add_dependencies(GPUSPTAGLibStatic GPUSPTAGLib)

    CUDA_ADD_EXECUTABLE(gpuindexbuilder ${AnnService}/src/IndexBuilder/main.cpp)
    target_link_libraries(gpuindexbuilder GPUSPTAGLibStatic ${Boost_LIBRARIES} ${CUDA_LIBRARIES})
    target_compile_definitions(gpuindexbuilder PRIVATE ${Definition})

    set(VECTORSEARCH_INC_DIR ${AnnService}/inc/SSDServing/VectorSearch)
    set(VECTORSEARCH_IMP_DIR ${AnnService}/src/SSDServing/VectorSearch)
    file(GLOB_RECURSE SSD_SERVING_HDR_FILES ${AnnService}/inc/SSDServing/*.h)
    file(GLOB_RECURSE SSD_SERVING_FILES ${AnnService}/src/SSDServing/*.cpp)
    if(NOT WIN32)
        list(REMOVE_ITEM SSD_SERVING_HDR_FILES 
            ${VECTORSEARCH_INC_DIR}/AsyncFileReader.h
            )
    elseif(WIN32)
        list(REMOVE_ITEM SSD_SERVING_HDR_FILES
            ${VECTORSEARCH_INC_DIR}/AsyncFileReaderLinux.h
            )
    endif()

    CUDA_ADD_EXECUTABLE(gpussdserving ${SSD_SERVING_HDR_FILES} ${SSD_SERVING_FILES})
    target_link_libraries(gpussdserving GPUSPTAGLibStatic ${Boost_LIBRARIES} ${CUDA_LIBRARIES})
    target_compile_definitions(gpussdserving PRIVATE ${Definition} _exe)
else()
    message (STATUS "Could not find cuda.")
endif()
