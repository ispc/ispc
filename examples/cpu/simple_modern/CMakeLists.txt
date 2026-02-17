#
#  Copyright (c) 2026, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

# Modern CMake ISPC Example
# This example demonstrates using ISPC as a CMake language with proper Ninja support

cmake_minimum_required(VERSION 3.19)

# Enable ISPC as a language
project(simple_modern LANGUAGES CXX ISPC)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable ISPC depfile support for Ninja generator
# This prevents unnecessary rebuilds when using Ninja
if(CMAKE_GENERATOR MATCHES "Ninja")
    # Try to find the module in the ISPC installation
    find_file(ISPC_DEPFILE_SUPPORT
        NAMES ISPCCompilerDepfileSupport.cmake
        PATHS ${CMAKE_MODULE_PATH}
              ${ISPC_ROOT}/lib/cmake/ispc
              /usr/local/lib/cmake/ispc
              /usr/lib/cmake/ispc
        NO_DEFAULT_PATH
    )

    if(ISPC_DEPFILE_SUPPORT)
        include(${ISPC_DEPFILE_SUPPORT})
    else()
        message(WARNING "ISPCCompilerDepfileSupport.cmake not found. Ninja builds may rebuild ISPC files unnecessarily.")
    endif()
endif()

# Create executable with both C++ and ISPC sources
add_executable(simple_modern
    simple.cpp
    simple.ispc
)

# Set ISPC compilation properties
set_target_properties(simple_modern PROPERTIES
    # Specify target instruction sets
    ISPC_INSTRUCTION_SETS "avx2-i32x8;sse4-i32x4"
    # Specify header suffix (default is _ispc.h)
    ISPC_HEADER_SUFFIX "_ispc.h"
)

# ISPC headers are generated in the build directory
target_include_directories(simple_modern PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)
