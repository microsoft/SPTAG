message (STATUS "git apply ${CMAKE_CURRENT_LIST_DIR}/diskann.patch WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/DiskANN/")
execute_process(COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/diskann.patch 
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/DiskANN/
)

set(DISKANN_SOURCE "${CMAKE_CURRENT_LIST_DIR}/DiskANN")
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/DiskANN")

#   thirdparty_dependent(target)
#   Add dependence for third-party libraries for "target"
#
function(thirdparty_dependent target)
    if (MSVC)
        checkEnvAndSetLocalVar("INTEL_ROOT" "Please install Intel MKL libraries and set the env variable INTEL_ROOT to the intel software directory. Should be similar to: C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries\\windows\\. " "INTEL_ROOT")
        set(MKL_ROOT ${INTEL_ROOT}/mkl)
        add_compile_options(/arch:AVX2 /Qpar)
        link_libraries("${INTEL_ROOT}/mkl/lib/intel64/mkl_core_dll.lib" "${INTEL_ROOT}/mkl/lib/intel64/mkl_rt.lib"  "${INTEL_ROOT}/mkl/lib/intel64/mkl_intel_thread_dll.lib" "${INTEL_ROOT}/compiler/lib/intel64/libiomp5md.lib"  "${INTEL_ROOT}/mkl/lib/intel64/mkl_intel_ilp64_dll.lib" "${INTEL_ROOT}/mkl/lib/intel64/mkl_sequential_dll.lib")
    else()
        set(INTEL_ROOT /opt/intel/compilers_and_libraries/linux)
        set(MKL_ROOT ${INTEL_ROOT}/mkl)
        add_compile_options(-m64 -Wl,--no-as-needed) 
        link_libraries(mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl)
        link_directories(${INTEL_ROOT}/lib/intel64 ${MKL_ROOT}/lib/intel64)
    endif()

    target_link_libraries("${target}" ${Boost_LIBRARIES} diskann)
endfunction()
