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

file(GLOB HDR_FILES ${PROJECT_SOURCE_DIR}/AnnService/inc/Core/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Core/BKT/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Helper/*.h)
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/AnnService/src/Core/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Core/BKT/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Helper/*.cpp)

include_directories(${PROJECT_SOURCE_DIR}/AnnService)

add_library (SPTAGLib SHARED ${SRC_FILES} ${HDR_FILES})
add_library (SPTAGLibStatic STATIC ${SRC_FILES} ${HDR_FILES})
set_target_properties(SPTAGLibStatic PROPERTIES OUTPUT_NAME SPTAGLib)

file(GLOB SERVER_HDR_FILES ${HDR_FILES} ${PROJECT_SOURCE_DIR}/AnnService/inc/Server/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Socket/*.h)
file(GLOB SERVER_FILES ${SRC_FILES} ${PROJECT_SOURCE_DIR}/AnnService/src/Server/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Socket/*.cpp)
add_executable (server ${SERVER_FILES} ${SERVER_HDR_FILES})
target_link_libraries(server ${Boost_LIBRARIES})

file(GLOB CLIENT_HDR_FILES ${HDR_FILES} ${PROJECT_SOURCE_DIR}/AnnService/inc/Client/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Socket/*.h)
file(GLOB CLIENT_FILES ${SRC_FILES} ${PROJECT_SOURCE_DIR}/AnnService/src/Client/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Socket/*.cpp)
add_executable (client ${CLIENT_FILES} ${CLIENT_HDR_FILES})
target_link_libraries(client ${Boost_LIBRARIES})

file(GLOB AGG_HDR_FILES ${HDR_FILES} ${PROJECT_SOURCE_DIR}/AnnService/inc/Aggregator/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Socket/*.h)
file(GLOB AGG_FILES ${SRC_FILES} ${PROJECT_SOURCE_DIR}/AnnService/src/Aggregator/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Socket/*.cpp)
add_executable (aggregator ${AGG_FILES} ${AGG_HDR_FILES})
target_link_libraries(aggregator ${Boost_LIBRARIES})

file(GLOB BUILDER_HDR_FILES ${HDR_FILES} ${PROJECT_SOURCE_DIR}/AnnService/inc/IndexBuilder/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/IndexBuilder/VectorSetReaders/*.h)
file(GLOB BUILDER_FILES ${SRC_FILES} ${PROJECT_SOURCE_DIR}/AnnService/src/IndexBuilder/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/IndexBuilder/VectorSetReaders/*.cpp)
add_executable (indexbuilder ${BUILDER_FILES} ${BUILDER_HDR_FILES})
target_link_libraries(indexbuilder ${Boost_LIBRARIES})

install(TARGETS SPTAGLib SPTAGLibStatic server client aggregator indexbuilder
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)