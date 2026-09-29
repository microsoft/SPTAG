# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(AnnService ${PROJECT_SOURCE_DIR}/AnnService)
set(Zstd ${PROJECT_SOURCE_DIR}/ThirdParty/zstd)

include_directories(${AnnService})
include_directories(${Zstd}/lib)

file(GLOB_RECURSE HDR_FILES ${AnnService}/inc/Core/*.h  ${AnnService}/inc/Helper/*.h)
file(GLOB_RECURSE SRC_FILES ${AnnService}/src/Core/*.cpp ${AnnService}/src/Helper/*.cpp)

set(SPDK_LIBRARIES "")
if (SPDK)
    set(Spdk ${PROJECT_SOURCE_DIR}/ThirdParty/spdk/build)
    set(Dpdk ${PROJECT_SOURCE_DIR}/ThirdParty/spdk/dpdk/build)
    set(IsalLCrypto ${PROJECT_SOURCE_DIR}/ThirdParty/isal-l_crypto/.libs/libisal_crypto.a)
    set(SpdkLibPrefix ${Spdk}/lib/libspdk_)
    set(DpdkLibPrefix ${Dpdk}/lib/librte_)
    if ((NOT EXISTS ${SpdkLibPrefix}bdev_nvme.a) OR (NOT EXISTS ${DpdkLibPrefix}pci.a) OR  (NOT EXISTS ${IsalLCrypto}))
        set (SPDK_LIBRARIES "")
        message (FATAL_ERROR "Cound not find SPDK, DPDK and IsalLCrypto!")
        list(REMOVE_ITEM HDR_FILES
            ${AnnService}/inc/Core/SPANN/ExtraSPDKController.h
        )

        list(REMOVE_ITEM SRC_FILES
            ${AnnService}/src/Core/SPANN/ExtraSPDKController.cpp
        )
    else()
        include_directories(${Spdk}/include)
        add_definitions(-DSPDK)
        set(SPDK_LIBRARIES -Wl,--whole-archive ${SpdkLibPrefix}bdev_nvme.a ${SpdkLibPrefix}bdev.a ${SpdkLibPrefix}nvme.a ${SpdkLibPrefix}vfio_user.a ${SpdkLibPrefix}sock.a ${SpdkLibPrefix}dma.a ${SpdkLibPrefix}notify.a ${SpdkLibPrefix}accel.a ${SpdkLibPrefix}event_bdev.a ${SpdkLibPrefix}event_accel.a ${SpdkLibPrefix}vmd.a ${SpdkLibPrefix}event_vmd.a ${SpdkLibPrefix}event_sock.a ${SpdkLibPrefix}event_iobuf.a ${SpdkLibPrefix}event.a ${SpdkLibPrefix}env_dpdk.a ${SpdkLibPrefix}log.a ${SpdkLibPrefix}thread.a ${SpdkLibPrefix}rpc.a ${SpdkLibPrefix}init.a ${SpdkLibPrefix}jsonrpc.a ${SpdkLibPrefix}json.a ${SpdkLibPrefix}trace.a ${SpdkLibPrefix}util.a ${DpdkLibPrefix}mempool.a ${DpdkLibPrefix}mempool_ring.a ${DpdkLibPrefix}eal.a ${DpdkLibPrefix}ring.a ${DpdkLibPrefix}telemetry.a ${DpdkLibPrefix}bus_pci.a ${DpdkLibPrefix}kvargs.a ${DpdkLibPrefix}pci.a -Wl,--no-whole-archive dl rt isal ${IsalLCrypto} uuid)
        message (STATUS "Found SPDK:${SPDK_LIBRARIES}")
    endif()
else()
    list(REMOVE_ITEM HDR_FILES
        ${AnnService}/inc/Core/SPANN/ExtraSPDKController.h
    )

    list(REMOVE_ITEM SRC_FILES
        ${AnnService}/src/Core/SPANN/ExtraSPDKController.cpp
    )
endif()

list(REMOVE_ITEM HDR_FILES
    ${AnnService}/inc/Core/Common/DistanceUtils.h
    ${AnnService}/inc/Core/Common/SIMDUtils.h 
    ${AnnService}/inc/Core/Common/InstructionUtils.h
    ${AnnService}/inc/Core/Common/CommonUtils.h
    )

list(REMOVE_ITEM SRC_FILES
    ${AnnService}/src/Core/Common/DistanceUtils.cpp
    ${AnnService}/src/Core/Common/SIMDUtils.cpp
    ${AnnService}/src/Core/Common/InstructionUtils.cpp
    )

add_library (DistanceUtils STATIC 
    inc/Core/Common/DistanceUtils.h
    inc/Core/Common/SIMDUtils.h
    inc/Core/Common/InstructionUtils.h
    inc/Core/Common/CommonUtils.h
    src/Core/Common/DistanceUtils.cpp
    src/Core/Common/SIMDUtils.cpp
    src/Core/Common/InstructionUtils.cpp
    )

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(DistanceUtils PRIVATE -mavx2 -mavx -msse -msse2 -mavx512f -mavx512bw -mavx512dq -fPIC)
endif()

add_library (SPTAGLib SHARED ${SRC_FILES} ${HDR_FILES})
target_link_libraries (SPTAGLib DistanceUtils ${RocksDB_LIBRARIES} ${uring_LIBRARIES} libzstd_shared ${NUMA_LIBRARY} ${TBB_LIBRARIES} ${SPDK_LIBRARIES})
add_library (SPTAGLibStatic STATIC ${SRC_FILES} ${HDR_FILES})
target_link_libraries (SPTAGLibStatic DistanceUtils ${RocksDB_LIBRARIES} ${uring_LIBRARIES} libzstd_static ${NUMA_LIBRARY_STATIC} ${TBB_LIBRARIES} ${SPDK_LIBRARIES})

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(SPTAGLibStatic PRIVATE -fPIC)
endif()

install(TARGETS SPTAGLib SPTAGLibStatic
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

if (NOT LIBRARYONLY)
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
    
    file(GLOB QUANTIZER_HDR_FILES ${AnnService}/inc/Quantizer/*.h)
    file(GLOB QUANTIZER_FILES ${AnnService}/src/Quantizer/*.cpp)
    add_executable (quantizer ${QUANTIZER_FILES} ${QUANTIZER_HDR_FILES})
    target_link_libraries(quantizer ${Boost_LIBRARIES} SPTAGLibStatic)

    install(TARGETS server client aggregator indexbuilder indexsearcher quantizer
      RUNTIME DESTINATION bin
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib)
endif()

file(GLOB_RECURSE SSD_SERVING_HDR_FILES ${AnnService}/inc/SSDServing/*.h)
file(GLOB_RECURSE SSD_SERVING_FILES ${AnnService}/src/SSDServing/*.cpp)

file(GLOB_RECURSE SPFRESH_HDR_FILES ${AnnService}/inc/SPFresh/*.h)
file(GLOB_RECURSE SPFRESH_FILES ${AnnService}/src/SPFresh/*.cpp)

file(GLOB_RECURSE KEYVALUETEST_HDR_FILES ${AnnService}/inc/KeyValueTest/*.h)
file(GLOB_RECURSE KEYVALUETEST_FILES ${AnnService}/src/KeyValueTest/*.cpp)

file(GLOB_RECURSE USEFULTOOL_FILES ${AnnService}/src/UsefulTool/*.cpp)

add_executable(ssdserving ${SSD_SERVING_HDR_FILES} ${SSD_SERVING_FILES})
add_executable(spfresh ${SPFRESH_HDR_FILES} ${SPFRESH_FILES})
add_executable(keyvaluetest ${KEYVALUETEST_HDR_FILES} ${KEYVALUETEST_FILES})
add_executable(usefultool ${USEFULTOOL_FILES})
target_link_libraries(ssdserving SPTAGLibStatic ${Boost_LIBRARIES} ${RocksDB_LIBRARIES} ${uring_LIBRARIES} ${TBB_LIBRARIES} ${SPDK_LIBRARIES})
target_link_libraries(spfresh SPTAGLibStatic ${Boost_LIBRARIES} ${RocksDB_LIBRARIES} ${uring_LIBRARIES} ${TBB_LIBRARIES} ${SPDK_LIBRARIES})
target_link_libraries(keyvaluetest SPTAGLibStatic ${Boost_LIBRARIES} ${RocksDB_LIBRARIES} ${uring_LIBRARIES} ${TBB_LIBRARIES} ${SPDK_LIBRARIES})
target_link_libraries(usefultool SPTAGLibStatic ${Boost_LIBRARIES} ${RocksDB_LIBRARIES} ${uring_LIBRARIES} ${TBB_LIBRARIES} ${SPDK_LIBRARIES})
target_compile_definitions(ssdserving PRIVATE _exe)
target_compile_definitions(spfresh PRIVATE _exe)
target_compile_definitions(keyvaluetest PRIVATE _exe)
target_compile_definitions(usefultool PRIVATE _exe)

# for Test
add_library(ssdservingLib ${SSD_SERVING_HDR_FILES} ${SSD_SERVING_FILES})
target_link_libraries(ssdservingLib SPTAGLibStatic ${Boost_LIBRARIES} ${RocksDB_LIBRARIES} ${uring_LIBRARIES} ${TBB_LIBRARIES} ${SPDK_LIBRARIES})

find_package(MPI)
if (MPI_FOUND)
    message (STATUS "Found MPI.")
    message (STATUS "MPI Include Path: ${MPI_CXX_INCLUDE_PATH}")
    message (STATUS "MPI Libraries: ${MPI_CXX_LIBRARIES}")
    include_directories (${MPI_CXX_INCLUDE_PATH})
    file(GLOB PARTITION_FILES ${AnnService}/src/BalancedDataPartition/*.cpp)
    add_executable(balanceddatapartition ${PARTITION_FILES})
    target_link_libraries(balanceddatapartition SPTAGLibStatic ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif()

install(TARGETS ssdservingLib ssdserving
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
