cmake_minimum_required(VERSION 3.28)
project(SPTAGConsumer LANGUAGES C CXX)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../.." "${CMAKE_BINARY_DIR}/sptag" EXCLUDE_FROM_ALL)

add_executable(sptag_sdk_example main.cpp)
target_link_libraries(sptag_sdk_example PRIVATE SPTAG::SPTAG)
