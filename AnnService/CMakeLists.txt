# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(AnnService ${PROJECT_SOURCE_DIR}/AnnService)

include_directories(${AnnService})

file(GLOB_RECURSE HDR_FILES ${AnnService}/inc/Core/*.h  ${AnnService}/inc/Helper/*.h)
file(GLOB_RECURSE SRC_FILES ${AnnService}/src/Core/*.cpp ${AnnService}/src/Helper/*.cpp)

list(REMOVE_ITEM HDR_FILES
    ${AnnService}/inc/Core/Common/DistanceUtils.h 
    ${AnnService}/inc/Core/Common/InstructionUtils.h
    ${AnnService}/inc/Core/Common/CommonUtils.h
    ${AnnService}/inc/Core/Common/PQQuantizer.h
    )

list(REMOVE_ITEM SRC_FILES
    ${AnnService}/src/Core/Common/DistanceUtils.cpp
    ${AnnService}/src/Core/Common/InstructionUtils.cpp
    ${AnnService}/src/Core/Common/PQQuantizer.cpp
    )

add_library (DistanceUtils STATIC 
    inc/Core/Common/DistanceUtils.h 
    inc/Core/Common/InstructionUtils.h
    inc/Core/Common/CommonUtils.h
    inc/Core/Common/PQQuantizer.h
    src/Core/Common/DistanceUtils.cpp
    src/Core/Common/InstructionUtils.cpp
    src/Core/Common/PQQuantizer.cpp
    )

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(DistanceUtils PRIVATE -mavx2 -mavx -msse -msse2 -fPIC)
endif()

add_library (SPTAGLib SHARED ${SRC_FILES} ${HDR_FILES})
target_link_libraries (SPTAGLib DistanceUtils)
add_library (SPTAGLibStatic STATIC ${SRC_FILES} ${HDR_FILES})
target_link_libraries (SPTAGLibStatic DistanceUtils)
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(SPTAGLibStatic PRIVATE -fPIC)
endif()

file(GLOB SERVER_HDR_FILES ${AnnService}/inc/Server/*.h ${AnnService}/inc/Socket/*.h)
file(GLOB SERVER_FILES ${AnnService}/src/Server/*.cpp ${AnnService}/src/Socket/*.cpp)
add_executable (server ${SERVER_FILES} ${SERVER_HDR_FILES})
target_link_libraries(server ${Boost_LIBRARIES} SPTAGLibStatic)

file(GLOB CLIENT_HDR_FILES ${AnnService}/inc/Client/*.h ${AnnService}/inc/Socket/*.h)
file(GLOB CLIENT_FILES ${AnnService}/src/Client/*.cpp ${AnnService}/src/Socket/*.cpp)
add_executable (client ${CLIENT_FILES} ${CLIENT_HDR_FILES})
target_link_libraries(client ${Boost_LIBRARIES} SPTAGLibStatic)

file(GLOB AGG_HDR_FILES ${AnnService}/inc/Aggregator/*.h ${AnnService}/inc/Socket/*.h ${AnnService}/inc/Server/QueryParser.h)
file(GLOB AGG_FILES ${AnnService}/src/Aggregator/*.cpp ${AnnService}/src/Socket/*.cpp ${AnnService}/src/Server/QueryParser.cpp)
add_executable (aggregator ${AGG_FILES} ${AGG_HDR_FILES})
target_link_libraries(aggregator ${Boost_LIBRARIES} SPTAGLibStatic)

file(GLOB BUILDER_FILES ${AnnService}/src/IndexBuilder/*.cpp)
add_executable (indexbuilder ${BUILDER_FILES})
target_link_libraries(indexbuilder ${Boost_LIBRARIES} SPTAGLibStatic)

file(GLOB SEARCHER_FILES ${AnnService}/src/IndexSearcher/*.cpp)
add_executable (indexsearcher ${SEARCHER_FILES})
target_link_libraries(indexsearcher ${Boost_LIBRARIES} SPTAGLibStatic)

install(TARGETS SPTAGLib SPTAGLibStatic server client aggregator indexbuilder indexsearcher
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
