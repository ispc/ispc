#
#  Copyright (c) 2021-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: simple-dpcpp
#

cmake_minimum_required(VERSION 3.13)
project(vadd-esimd)

set(TARGET_NAME "vadd-esimd")

add_executable(${TARGET_NAME} vadd_usm.cpp)
set(DPCPP_CUSTOM_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR})
add_dpcpp_library(${TARGET_NAME}_esimd vadd_usm.cpp)

# Use LLVM bitcode linking for interop with ESIMD
set(ISPC_XE_FORMAT "bc")
add_ispc_library(${TARGET_NAME}_ispc2esimd vadd_func.ispc)

# Link for GPU
ispc_target_link_dpcpp_esimd_libraries(${TARGET_NAME}_ispc2esimd ${TARGET_NAME}_esimd)

target_compile_options(${TARGET_NAME} PRIVATE "-fsycl" "-fPIE")
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(${TARGET_NAME} PRIVATE sycl ${TARGET_NAME}_ispc2esimd)
set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
