cmake_minimum_required (VERSION 2.8)

project (SPTAGLib)

function(CXX_COMPILER_DUMPVERSION _OUTPUT_VERSION)

  exec_program(${CMAKE_CXX_COMPILER}
    ARGS ${CMAKE_CXX_COMPILER_ARG1} -dumpversion
    OUTPUT_VARIABLE COMPILER_VERSION
  )

  set(${_OUTPUT_VERSION} ${COMPILER_VERSION} PARENT_SCOPE)
endfunction()

if(NOT WIN32)
    CXX_COMPILER_DUMPVERSION(CXX_COMPILER_VERSION)
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    # require at least gcc 4.7
    if (CXX_COMPILER_VERSION VERSION_LESS 4.7)
        message(FATAL_ERROR "GCC version must be at least 4.7!")
    endif()
    set (CMAKE_CXX_FLAGS_RELEASE "-Wall -Wunreachable-code -Wno-reorder -Wno-sign-compare -Wno-unknown-pragmas -Wcast-align -lm -lrt -DNDEBUG -std=c++11 -fopenmp -march=native")
    set (CMAKE_CXX_FLAGS_DEBUG   "-Wall -Wunreachable-code -Wno-reorder -Wno-sign-compare -Wno-unknown-pragmas -Wcast-align -ggdb -lm -lrt -DNDEBUG -std=c++11 -fopenmp -march=native")

elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    if (CXX_COMPILER_VERSION VERSION_LESS 14.0.1)
        message(FATAL_ERROR "Intel version must be at least 14.0.1!")
    endif()
    set (CMAKE_CXX_FLAGS_RELEASE "-Wall -Wunreachable-code -lrt -DNDEBUG -std=c++11 -fopenmp -march=native")
    set (CMAKE_CXX_FLAGS_DEBUG   "-Wall -Wunreachable-code -ggdb  -lrt -DNDEBUG -std=c++11 -fopenmp -march=native")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    if (CXX_COMPILER_VERSION VERSION_LESS 4.2.1)
        message(FATAL_ERROR "Clang version must be at least 3.4 (GCC >= 4.2.1 equivalent)!")
    endif()
    if (CMAKE_SYSTEM_NAME MATCHES Darwin)
        # MACOSX
        set (CMAKE_CXX_FLAGS_RELEASE "-Wall -Wunreachable-code -Wcast-align -DNDEBUG -std=c++11 -fopenmp -march=native")
        set (CMAKE_CXX_FLAGS_DEBUG   "-Wall -Wunreachable-code -Wcast-align -ggdb  -DNDEBUG -std=c++11 -fopenmp -march=native")
    else()
        set (CMAKE_CXX_FLAGS_RELEASE "-Wall -Wunreachable-code -Wcast-align -DNDEBUG -std=c++11 -fopenmp -march=native")
        set (CMAKE_CXX_FLAGS_DEBUG   "-Wall -Wunreachable-code -Wcast-align -ggdb  -DNDEBUG -std=c++11 -fopenmp -march=native")
    endif()
elseif(WIN32)
    if(NOT MSVC14)
         message(FATAL_ERROR "On Windows, only MSVC version 14 are supported!") 
    endif()
else ()
    message(FATAL_ERROR "Unrecognized compiler (use GCC, Clang, Intel compiler, or MSVC)!")
endif()

if (WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
else()
    if (NOT CMAKE_BUILD_TYPE)
        set (CMAKE_BUILD_TYPE Release CACHE STRING "Build types: Release Debug" FORCE)
    endif (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory (AnnService)
add_subdirectory (PythonWrapper)
