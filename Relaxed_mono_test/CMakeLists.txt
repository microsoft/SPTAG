# Copyright (c) Microsoft
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.15)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === Boost dependencies ===
find_package(Boost 1.66 REQUIRED COMPONENTS system filesystem)

if (Boost_FOUND)
    message(STATUS "Found Boost.")
    message(STATUS "Include Path: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Library Path: ${Boost_LIBRARY_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
else()
    message(FATAL_ERROR "Boost 1.66+ is required.")
endif()

# === SPTAG includes and libraries ===
include_directories(
    ${PROJECT_SOURCE_DIR}/AnnService 
    ${PROJECT_SOURCE_DIR}/ThirdParty/spdk/build/include
)

# === Sources ===
file(GLOB RELAXED_MONO_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB RELAXED_MONO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB MAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

# Validate presence
if(NOT MAIN_FILE)
    message(FATAL_ERROR "main.cpp not found in Relaxedmono/")
endif()

add_executable(Relaxedmono
    ${MAIN_FILE}
    ${RELAXED_MONO_HEADERS}
    ${RELAXED_MONO_SOURCES}
)

target_link_libraries(
    Relaxedmono
    SPTAGLibStatic 
    ssdservingLib 
    ${Boost_LIBRARIES})

target_include_directories(
    Relaxedmono 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# === Optional Install Step ===
install(TARGETS Relaxedmono
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
