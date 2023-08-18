#
#  Copyright (c) 2019-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.12)

project(XeExamples)

message(STATUS "Configuring Xe examples")

option(ISPC_INCLUDE_DPCPP_EXAMPLES "Generate build targets for the ISPC/DPCPP interoperability examples" OFF)

if(CMAKE_BUILD_TYPE)
    # Validate build type
    set(CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo")

    string(FIND "${CONFIGURATION_TYPES}" "${CMAKE_BUILD_TYPE}" MATCHED_CONFIG)
    if (${MATCHED_CONFIG} EQUAL -1)
         message(FATAL_ERROR "CMAKE_BUILD_TYPE (${CMAKE_BUILD_TYPE}) allows only the following values: ${CONFIGURATION_TYPES}")
    endif()
else(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Build type not specified: Use Release by default.")
endif(CMAKE_BUILD_TYPE)

if (NOT DEFINED ISPC_EXECUTABLE)
    find_program (ISPC_EXECUTABLE ispc)
endif()

if (ISPC_BUILD)
    include(${CMAKE_CURRENT_LIST_DIR}/../../ispcrt/cmake/ispc.cmake)
    set(ISPCRT_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../ispcrt)
    find_library(LEVEL_ZERO_LIB_LOADER ze_loader HINTS ${LEVEL_ZERO_ROOT}/lib)
    set(LEVEL_ZERO_INCLUDE_DIR ${LEVEL_ZERO_ROOT}/include)
    set(GBENCH_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../benchmarks/vendor/google/benchmark/include)
    include_directories_ispc(${ISPCRT_INCLUDE_DIR})
else()
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_LIST_DIR}/../../lib64/cmake)
    find_package(ispcrt REQUIRED)

    # L0 is required, e.g., for sgemm
    list(APPEND CMAKE_MODULE_PATH ${ISPCRT_DIR})
    find_package(level_zero REQUIRED)
    message(STATUS "LEVEL_ZERO_INCLUDE_DIR is ${LEVEL_ZERO_INCLUDE_DIR}")
    message(STATUS "LEVEL_ZERO_LIB_LOADER is ${LEVEL_ZERO_LIB_LOADER}")
endif()
if (NOT ISPC_EXECUTABLE)
    message(FATAL_ERROR "Failed to find ispc")
endif()
message(STATUS "Found ISPC: ${ISPC_EXECUTABLE}")

# Used for testing
find_package(Python3 REQUIRED)
if (NOT Python3_Interpreter_FOUND)
    message(FATAL_ERROR "Failed to find python3 interpretor")
endif()

set(COMMON_PATH "${CMAKE_CURRENT_SOURCE_DIR}/common" "${CMAKE_CURRENT_SOURCE_DIR}/../common")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AddPerfExample.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestDrivers.cmake)

add_subdirectory(noise)
add_subdirectory(sgemm)
add_subdirectory(aobench)
add_subdirectory(mandelbrot)
add_subdirectory(simple)
add_subdirectory(simple-usm)
add_subdirectory(simple-fence)
add_subdirectory(usm-mem)

# DPC++ related examples should not run as part of ISPC_BUILD
# They require complete ISPC installation and ISPC_INCLUDE_DPCPP_EXAMPLES turned ON
if (NOT ISPC_BUILD AND ISPC_INCLUDE_DPCPP_EXAMPLES)
    include(${ISPCRT_DIR}/interop.cmake)
    if (NOT "${CMAKE_CXX_COMPILER}" STREQUAL "${DPCPP_COMPILER}")
        message(FATAL_ERROR "CMAKE_CXX_COMPILER should be set to " ${DPCPP_COMPILER} " when ISPC_INCLUDE_DPCPP_EXAMPLES=ON" )
    endif()
    add_subdirectory(simple-dpcpp)
    add_subdirectory(simple-dpcpp-l0)
    add_subdirectory(pipeline-dpcpp)
    add_subdirectory(simple-esimd)
    add_subdirectory(vadd-esimd)
    add_subdirectory(callback-esimd)
    add_subdirectory(invoke-sycl-aobench)
    add_subdirectory(invoke-simd-vadd)
endif()

include(TestLists.txt)
