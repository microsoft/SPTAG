# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB HDR_FILES ${PROJECT_SOURCE_DIR}/AnnService/inc/Core/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Core/Common/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Core/BKT/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Core/KDT/*.h ${PROJECT_SOURCE_DIR}/AnnService/inc/Helper/*.h)
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/AnnService/src/Core/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Core/Common/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Core/BKT/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Core/KDT/*.cpp ${PROJECT_SOURCE_DIR}/AnnService/src/Helper/*.cpp)

include_directories(${PROJECT_SOURCE_DIR}/AnnService)

file(GLOB SEARCH_FILES ${SRC_FILES} ${PROJECT_SOURCE_DIR}/Search/*.cpp)
add_executable (search ${SEARCH_FILES} ${HDR_FILES})
target_link_libraries(search ${Boost_LIBRARIES} ${TBB_LIBRARIES})

install(TARGETS search
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

