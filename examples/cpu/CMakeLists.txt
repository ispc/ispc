#
#  Copyright (c) 2018-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples CMakeLists.txt
#
cmake_minimum_required(VERSION 3.12)

set (PROJECT_NAME ispc_examples)
project(${PROJECT_NAME} C CXX)

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

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AddISPCExample.cmake)

if(WIN32)
    add_compile_definitions(WIN32)
    if(MSVC)
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/EHsc>)
    endif()
endif()

if (NOT DEFINED ISPC_EXECUTABLE)
    find_program (ISPC_EXECUTABLE ispc)
    if (NOT ISPC_EXECUTABLE)
        message(FATAL_ERROR "Failed to find ispc" )
    endif()
endif()

set (EXAMPLES_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_subdirectory(aobench)
add_subdirectory(aobench_instrumented)
add_subdirectory(deferred)
add_subdirectory(gmres)
add_subdirectory(mandelbrot)
add_subdirectory(mandelbrot_tasks)
add_subdirectory(noise)
add_subdirectory(options)
add_subdirectory(perfbench)
add_subdirectory(rt)
add_subdirectory(sgemm)
add_subdirectory(simple)
add_subdirectory(sort)
add_subdirectory(stencil)
add_subdirectory(volume_rendering)
