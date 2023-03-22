# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.13)
project(invoke_simd_vadd)

include(${ISPCRT_DIR}/interop.cmake)

set(TARGET_NAME "invoke_simd_vadd")

add_executable(${TARGET_NAME} invoke_simd_ispc.cpp)

set(ISPC_TARGET_XE "xehpg-x16")
# The linkage for SYCL->ISPC invoke_simd cases is based on LLVM IR link
# Set ISPC_XE_FORMAT to "bc" explicitly (default is spv)
set(ISPC_XE_FORMAT "bc")
add_ispc_library(${TARGET_NAME}_ispc2dpcpp vadd.ispc)
add_dpcpp_library(${TARGET_NAME}_dpcpp invoke_simd_ispc.cpp)

# Link for GPU
ispc_target_link_dpcpp_libraries(${TARGET_NAME}_ispc2dpcpp ${TARGET_NAME}_dpcpp)

target_compile_options(${TARGET_NAME} PRIVATE "-fsycl" "-fPIE" "-Xclang" "-fsycl-allow-func-ptr")
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(${TARGET_NAME} PRIVATE sycl ${TARGET_NAME}_ispc2dpcpp)
set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
