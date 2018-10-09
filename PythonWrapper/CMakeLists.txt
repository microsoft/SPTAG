if (WIN32)
    set(PYTHON_INCLUDE_PATH C:\Python27\include)
    set(PY_SUFFIX .pyd)
else()
    set(PYTHON_INCLUDE_PATH /usr/include/python2.7)
    set(PY_SUFFIX .so)
endif()

include_directories(${PYTHON_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/AnnService ${PROJECT_SOURCE_DIR}/PythonWrapper)
message(STATUS "Header files: ${HDR_FILES}")
message(STATUS "Source files: ${SRC_FILES}")

find_package(OpenMP)
if (OpenMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message (STATUS "Found openmp.")
else()
    message (FATAL_ERROR "Could no find openmp!")
endif()

find_package(Boost 1.67 COMPONENTS system thread serialization wserialization regex)
if (Boost_FOUND)
    include_directories (${Boost_INCLUDE_DIR})
    link_directories (${Boost_LIBRARY_DIR} "/usr/lib")
    message (STATUS "Found Boost.")
    message (STATUS "Include Path: ${Boost_INCLUDE_DIRS}")
    message (STATUS "Library Path: ${Boost_LIBRARY_DIRS}")
    message (STATUS "Library: ${Boost_LIBRARIES}")
else()
    message (FATAL_ERROR "Could not find Boost 1.67!")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set (LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/release/")
    set (EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/release/")
else ()
    set (LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/debug/")
    set (EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/debug/")
endif ()

execute_process(COMMAND swig -l${PROJECT_SOURCE_DIR}/PythonWrapper/inc/PyByteArray.i -python -c++ ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/CoreInterface.h)
execute_process(COMMAND mv ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/SPTAG.py ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/CoreInterface_wrap.cxx ${LIBRARY_OUTPUT_PATH})

execute_process(COMMAND swig -l${PROJECT_SOURCE_DIR}/PythonWrapper/inc/PyByteArray.i -python -c++ ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/ClientInterface.h)
execute_process(COMMAND mv ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/SPTAGClient.py ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/ClientInterface_wrap.cxx ${LIBRARY_OUTPUT_PATH})

set(CMAKE_SHARED_LIBRARY_PREFIX "")
file(GLOB CORE_HDR_FILES ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/CoreInterface.h ${OTH_HDR_FILES})
file(GLOB CORE_SRC_FILES ${PROJECT_SOURCE_DIR}/PythonWrapper/src/CoreInterface.cpp ${LIBRARY_OUTPUT_PATH}/CoreInterface_wrap.cxx)
add_library (_SPTAG SHARED ${CORE_SRC_FILES} ${CORE_HDR_FILES})
target_link_libraries(_SPTAG SPTAGLib python2.7)

file(GLOB CLIENT_HDR_FILES ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/ClientInterface.h ${OTH_HDR_FILES} ${PROJECT_SOURCE_DIR}/AnnService/inc/Socket/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Client/*.h)
file(GLOB CLIENT_SRC_FILES ${PROJECT_SOURCE_DIR}/PythonWrapper/src/ClientInterface.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Socket/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Client/*.cpp ${LIBRARY_OUTPUT_PATH}/ClientInterface_wrap.cxx)
add_library (_SPTAGClient SHARED ${CLIENT_SRC_FILES} ${CLIENT_HDR_FILES})
target_link_libraries(_SPTAGClient SPTAGLib python2.7 ${Boost_LIBRARIES})

string(LENGTH ${PROJECT_SOURCE_DIR} PREFIX_LEN) 
MATH(EXPR PREFIX_LEN "${PREFIX_LEN}+1")
foreach(F ${PROJ_HDR_FILES}) 
  GET_FILENAME_COMPONENT(FP ${F} PATH)
  string(SUBSTRING ${FP} ${PREFIX_LEN} -1 FS)
  message(${PREFIX_LEN} ":" ${FS})
  install(FILES "${F}" DESTINATION ${FS})
endforeach(F)

install(TARGETS _SPTAG  
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  COMPONENT library
)

install(FILES ${PROJECT_SOURCE_DIR}/PythonWrapper/src/SPTAG.py DESTINATION lib)
