# Created by SwiftWare Lab on 9/24.
# CE 4SP4 - High Performance Programming
# Copyright (c) 2024 SwiftWare Lab

cmake_minimum_required(VERSION 3.18)


option(PROFILING_ENABLED "Enables Modules for Profiling, needs LIBFPM4" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#include(FetchContent)
#
#FetchContent_Declare(
#        argparse
#        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
#)
#FetchContent_MakeAvailable(argparse)
#
#


include(FetchContent)
FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.0
)
# enable BENCHMARK_DOWNLOAD_DEPENDENCIES
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
if (PROFILING_ENABLED)
  # enable PFM
  set(BENCHMARK_ENABLE_LIBPFM ON)
endif()



FetchContent_MakeAvailable(googlebenchmark)



include_directories(
        ${CMAKE_SOURCE_DIR}/include
)

project(lab2)

set(CMAKE_CXX_STANDARD 17)
add_executable(lab2_spmm
        ${CMAKE_SOURCE_DIR}/spmm_driver.cpp
        ${CMAKE_SOURCE_DIR}/src/gemm.cpp
        ${CMAKE_SOURCE_DIR}/src/spmm.cpp
        ${CMAKE_SOURCE_DIR}/src/utils.cpp
        )

target_link_libraries(lab2_spmm
        benchmark::benchmark )

add_executable(lab2_spmv
        ${CMAKE_SOURCE_DIR}/spmv_driver.cpp
        ${CMAKE_SOURCE_DIR}/src/gemv.cpp
        ${CMAKE_SOURCE_DIR}/src/spmv.cpp
        ${CMAKE_SOURCE_DIR}/src/utils.cpp
)

target_link_libraries(lab2_spmv
        benchmark::benchmark )

add_executable(lab2_sparseNN
        ${CMAKE_SOURCE_DIR}/sparseNN_driver.cpp
        ${CMAKE_SOURCE_DIR}/src/utils.cpp
        ${CMAKE_SOURCE_DIR}/src/sparse_nn.cpp
        ${CMAKE_SOURCE_DIR}/src/gemm.cpp
)
target_link_libraries(lab2_sparseNN
        benchmark::benchmark )

add_subdirectory(test)