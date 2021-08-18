#
#  Copyright (c) 2019-2021, Intel Corporation
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
#   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

# CMC is needed for ISPC build so it should have been already found but it is possible to overwrite this with CM_INSTALL_PATH.
if (CM_INSTALL_PATH)
    find_program (CMC_EXECUTABLE cmc PATHS ${CM_INSTALL_PATH}/bin NO_DEFAULT_PATH)
endif()

if (NOT CMC_EXECUTABLE)
    message(STATUS "CM compiler was not found so CM tests will not run. If you want to run them set path to CM install root with -DCM_INSTALL_PATH")
else()
    message(STATUS "Found CMC to use in examples: ${CMC_EXECUTABLE}")
endif()

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
    find_package(ispcrt REQUIRED)
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
include(FindPythonInterp)

set(COMMON_PATH "${CMAKE_CURRENT_SOURCE_DIR}/common" "${CMAKE_CURRENT_SOURCE_DIR}/../common")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AddPerfExample.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestDrivers.cmake)

add_subdirectory(noise)
add_subdirectory(sgemm)
add_subdirectory(aobench)
add_subdirectory(mandelbrot)
add_subdirectory(simple)
add_subdirectory(simple-usm)

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
endif()

include(TestLists.txt)
