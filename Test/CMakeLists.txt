find_package(OpenMP)
if (OpenMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message (STATUS "Found openmp.")
else()
    message (FATAL_ERROR "Could no find openmp!")
endif()

if(NOT WIN32)
  ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
  message (STATUS "BOOST_TEST_DYN_LINK")
endif()

find_package(Boost 1.67 COMPONENTS system thread serialization wserialization regex filesystem unit_test_framework)
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

include_directories(${PYTHON_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/AnnService ${PROJECT_SOURCE_DIR}/PythonWrapper ${PROJECT_SOURCE_DIR}/Test)

file(GLOB TEST_HDR_FILES ${PROJECT_SOURCE_DIR}/Test/inc/Test.h)
file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/Test/src/*.cpp)
add_executable (test ${TEST_SRC_FILES} ${TEST_HDR_FILES})
target_link_libraries(test SPTAGLib ${Boost_LIBRARIES})

install(TARGETS test
  RUNTIME DESTINATION bin  
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

