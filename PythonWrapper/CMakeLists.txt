# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

find_package(Python2 COMPONENTS Development)
if (Python2_FOUND)    
    include_directories (${Python2_INCLUDE_DIRS})
    link_directories (${Python2_LIBRARY_DIRS})
    message (STATUS "Found Python.")
    message (STATUS "Include Path: ${Python2_INCLUDE_DIRS}")
    message (STATUS "Library Path: ${Python2_LIBRARIES}")
    set (Python_LIBRARIES ${Python2_LIBRARIES})
else()
    message (STATUS "Could not find Python 2.7!")
    find_package(Python3 COMPONENTS Development)
    if (Python3_FOUND)
        include_directories (${Python3_INCLUDE_DIRS})
        link_directories (${Python3_LIBRARY_DIRS})
        message (STATUS "Found Python.")
        message (STATUS "Include Path: ${Python3_INCLUDE_DIRS}")
        message (STATUS "Library Path: ${Python3_LIBRARIES}")
        set (Python_LIBRARIES ${Python3_LIBRARIES})
    else ()
        message (FATAL_ERROR "Could not find python2 or python3!")
    endif()
endif()

if (WIN32)
    set(PY_SUFFIX .pyd)
else()
	set(PY_SUFFIX .so)
endif()

execute_process(COMMAND swig -l${PROJECT_SOURCE_DIR}/PythonWrapper/inc/PyByteArray.i -python -c++ ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/CoreInterface.h)
execute_process(COMMAND swig -l${PROJECT_SOURCE_DIR}/PythonWrapper/inc/PyByteArray.i -python -c++ ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/ClientInterface.h)

include_directories(${PYTHON_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/AnnService ${PROJECT_SOURCE_DIR}/PythonWrapper)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

file(GLOB CORE_HDR_FILES ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/CoreInterface.h)
file(GLOB CORE_SRC_FILES ${PROJECT_SOURCE_DIR}/PythonWrapper/src/CoreInterface.cpp ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/CoreInterface_wrap.cxx)
add_library (_SPTAG SHARED ${CORE_SRC_FILES} ${CORE_HDR_FILES})
set_target_properties(_SPTAG PROPERTIES SUFFIX ${PY_SUFFIX})
target_link_libraries(_SPTAG SPTAGLib ${Python_LIBRARIES} ${TBB_LIBRARIES})
add_custom_command(TARGET _SPTAG POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/SPTAG.py ${EXECUTABLE_OUTPUT_PATH})

file(GLOB CLIENT_HDR_FILES ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/ClientInterface.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Socket/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Client/*.h)
file(GLOB CLIENT_SRC_FILES ${PROJECT_SOURCE_DIR}/PythonWrapper/src/ClientInterface.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Socket/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Client/*.cpp ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/ClientInterface_wrap.cxx)
add_library (_SPTAGClient SHARED ${CLIENT_SRC_FILES} ${CLIENT_HDR_FILES})
set_target_properties(_SPTAGClient PROPERTIES SUFFIX ${PY_SUFFIX})
target_link_libraries(_SPTAGClient SPTAGLib ${Python_LIBRARIES} ${Boost_LIBRARIES} ${TBB_LIBRARIES})
add_custom_command(TARGET _SPTAGClient POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/SPTAGClient.py ${EXECUTABLE_OUTPUT_PATH})

install(TARGETS _SPTAG _SPTAGClient
  RUNTIME DESTINATION bin  
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(FILES ${PROJECT_SOURCE_DIR}/PythonWrapper/src/SPTAG.py ${PROJECT_SOURCE_DIR}/PythonWrapper/inc/SPTAGClient.py DESTINATION bin)
