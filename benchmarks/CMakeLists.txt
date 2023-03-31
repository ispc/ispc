#
#  Copyright (c) 2020-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.13)

set(BENCHMARKS_PROJECT_NAME ispc_benchmarks)
project(${BENCHMARKS_PROJECT_NAME}
        DESCRIPTION "ISPC micro benchmarks"
        LANGUAGES CXX)

#################### Add Google Benchmark to the project #####################

# Benchmarks depend on Google Benchmark submodule
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/benchmarks/vendor/google/benchmark/CMakeLists.txt")
    message(STATUS "Google Beanchmark submodule update")
    execute_process(COMMAND ${GIT_BINARY} submodule update --init --recursive
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Build google benchmark (target: benchmark)
# Do not build tests of benchmarking lib
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
# Do not install benchmarking lib, we don't need it, as we link statically..
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Suppressing benchmark's install" FORCE)
add_subdirectory(vendor/google/benchmark)

# Suppress warnings, as benchmarks are built with -Werror and trunk version of clang may trigger
# new warnings, which are not yet fixed in Google Benchmark sources.
target_compile_options(benchmark PRIVATE "-w")

# Warn if build type is not release.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # NDEBUG may be excluded from build options globally, if LLVM build doesn't have that.
    # But building Google Benchmark, we need it to let it know that optimized build is used.
    target_compile_definitions(benchmark PRIVATE NDEBUG)
else()
    message(WARNING "Build type ${CMAKE_BUILD_TYPE} is used for building benchmarks. Benchmarks themselves and Google Benchmark will be built using this build type!")
endif()

####################### Configure and add benchmarks  ########################

# Meta target for benchmarks
add_custom_target(${BENCHMARKS_PROJECT_NAME})

if (X86_HOST)
    set(BENCHMARKS_ISPC_TARGETS "avx2-i32x8" CACHE STRING "Comma separated list of ISPC targets to build benchmarks")
else()
    set(BENCHMARKS_ISPC_TARGETS "neon-i32x4" CACHE STRING "Comma separated list of ISPC targets to build benchmarks")
endif()
set(BENCHMARKS_ISPC_FLAGS "-O3 --woff" CACHE STRING "Flags to pass to ISPC compiler to build benchmarks")
message(STATUS "Using BENCHMARKS_ISPC_TARGETS: ${BENCHMARKS_ISPC_TARGETS}")
message(STATUS "Using BENCHMARKS_ISPC_FLAGS: ${BENCHMARKS_ISPC_FLAGS}")

include(cmake/AddBenchmark.cmake)

add_subdirectory(01_trivial)
add_subdirectory(02_medium)
add_subdirectory(03_complex)
