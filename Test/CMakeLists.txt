# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if (NOT LIBRARYONLY)
  find_package(Boost 1.66 COMPONENTS system thread serialization wserialization regex filesystem)
  if (Boost_FOUND)
    include_directories (${Boost_INCLUDE_DIR})
    link_directories (${Boost_LIBRARY_DIR})
    message (STATUS "Found Boost.")
    message (STATUS "Include Path: ${Boost_INCLUDE_DIRS}")
    message (STATUS "Library Path: ${Boost_LIBRARY_DIRS}")
    message (STATUS "Library: ${Boost_LIBRARIES}")
  else()
    message (FATAL_ERROR "Could not find Boost 1.67!")
  endif()

  # GoogleTest
  include(FetchContent)
  find_package(GTest QUIET)
  if (NOT GTest_FOUND)
    message(STATUS "GTest not found; fetching googletest via FetchContent...")
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    # For Windows: prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

    include_directories(${PROJECT_SOURCE_DIR}/AnnService ${PROJECT_SOURCE_DIR}/Test ${PROJECT_SOURCE_DIR}/ThirdParty/spdk/build/include)

    file(GLOB TEST_HDR_FILES ${PROJECT_SOURCE_DIR}/Test/inc/Test.h)
    file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/Test/src/*.cpp)
    add_executable(SPTAGTest ${TEST_SRC_FILES} ${TEST_HDR_FILES})
  target_link_libraries(SPTAGTest PRIVATE SPTAGLibStatic ssdservingLib ${Boost_LIBRARIES} GTest::gtest GTest::gtest_main)

    install(TARGETS SPTAGTest
      RUNTIME DESTINATION bin  
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib)
endif()
