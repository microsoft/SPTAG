message (STATUS "git apply ${CMAKE_CURRENT_LIST_DIR}/diskann.patch WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/DiskANN/")

execute_process(COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/diskann.patch 
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/DiskANN/
)

set(DISKANN_SOURCE "${CMAKE_CURRENT_LIST_DIR}/DiskANN")
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/DiskANN")

# thirdparty_dependent(target)
#   Add dependence for third-party libraries for "target"
function(thirdparty_dependent target)
    target_link_libraries("${target}" ${Boost_LIBRARIES})

    target_include_directories("${target}" BEFORE PUBLIC "${DISKANN_SOURCE}/include")
    #target_link_libraries("${target}" SPTAGLibStatic)
endfunction()
