#
#  Copyright (c) 2025, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.16)
project(simple_jit VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ISPC using our package configuration
find_package(ispc REQUIRED)

# Create the executable
add_executable(simple_jit simple.cpp)

# Link against ISPC library
target_link_libraries(simple_jit ispc::lib)

# Copy simple.ispc to build directory so the example can find it
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/simple.ispc ${CMAKE_CURRENT_BINARY_DIR}/simple.ispc COPYONLY)