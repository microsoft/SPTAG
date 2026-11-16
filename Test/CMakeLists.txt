# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if (NOT LIBRARYONLY)
    if(NOT WIN32)
      ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
      message (STATUS "BOOST_TEST_DYN_LINK")
    endif()

    find_package(Boost 1.66 COMPONENTS system thread serialization wserialization regex filesystem unit_test_framework)
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

    include_directories(${PROJECT_SOURCE_DIR}/AnnService ${PROJECT_SOURCE_DIR}/Test ${PROJECT_SOURCE_DIR}/ThirdParty/spdk/build/include)

    file(GLOB TEST_HDR_FILES ${PROJECT_SOURCE_DIR}/Test/inc/Test.h)
    file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/Test/src/*.cpp)
    list(REMOVE_ITEM TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/Test/src/TiKVPostingClientTest.cpp)

    if (NOT RABITQ)
        list(REMOVE_ITEM TEST_SRC_FILES
            ${PROJECT_SOURCE_DIR}/Test/src/RaBitQQuantizerTest.cpp)
    endif()

    add_executable(SPTAGTest ${TEST_SRC_FILES} ${TEST_HDR_FILES})
    target_link_libraries(SPTAGTest SPTAGLibStatic ssdservingLib ${Boost_LIBRARIES})
    if (TIKV)
        find_package(absl CONFIG REQUIRED)
        target_link_libraries(SPTAGTest absl::synchronization)
    endif()

    if (MULTITENANT)
        add_executable(tenantperf tenant_perf.cpp)
        target_compile_definitions(tenantperf PRIVATE TENANT_BENCH_CURRENT)
        target_link_libraries(tenantperf SPTAGLibStatic ${Boost_LIBRARIES})
        add_executable(postingbackendperf posting_backend_perf.cpp)
        target_link_libraries(postingbackendperf SPTAGLibStatic ${Boost_LIBRARIES})
        set(TENANT_SERVER_SOURCES
            ${PROJECT_SOURCE_DIR}/AnnService/src/Server/ServiceContext.cpp
            ${PROJECT_SOURCE_DIR}/AnnService/src/Server/ServiceSettings.cpp
            ${PROJECT_SOURCE_DIR}/AnnService/src/Server/SearchExecutor.cpp
            ${PROJECT_SOURCE_DIR}/AnnService/src/Server/SearchExecutionContext.cpp
            ${PROJECT_SOURCE_DIR}/AnnService/src/Server/QueryParser.cpp
            ${PROJECT_SOURCE_DIR}/AnnService/src/Socket/RemoteSearchQuery.cpp)
        target_sources(SPTAGTest PRIVATE ${TENANT_SERVER_SOURCES})
        add_executable(SPTAGTenantTest src/main.cpp src/MultiTenantTest.cpp src/PostingBackendTest.cpp ${TENANT_SERVER_SOURCES})
        target_link_libraries(SPTAGTenantTest SPTAGLibStatic ${Boost_LIBRARIES})
        if (TIKV)
            target_link_libraries(SPTAGTenantTest absl::synchronization)
            add_executable(SPTAGTiKVClientTest src/main.cpp src/TiKVPostingClientTest.cpp)
            target_link_libraries(SPTAGTiKVClientTest SPTAGLibStatic ${Boost_LIBRARIES} absl::synchronization)
            add_test(NAME TiKVPostingClient COMMAND SPTAGTiKVClientTest --run_test=TiKVPostingClient)
        endif()
        add_test(NAME MultiTenant COMMAND SPTAGTenantTest --run_test=MultiTenant)
        add_test(NAME PostingBackends COMMAND SPTAGTenantTest --run_test=PostingBackends)
    endif()

    install(TARGETS SPTAGTest
      RUNTIME DESTINATION bin  
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib)
endif()
