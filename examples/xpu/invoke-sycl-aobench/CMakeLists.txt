# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.13)
project(invoke_sycl_aobench LANGUAGES CXX)

include(${ISPCRT_DIR}/interop.cmake)

set(ISPC_TARGET_XE "xehpg-x16")

# ISPC-only version
add_executable(invoke_sycl_aobench_ispc invoke_sycl_aobench.cpp)
target_link_libraries(invoke_sycl_aobench_ispc PRIVATE ispcrt::ispcrt ze_loader)
target_compile_definitions(invoke_sycl_aobench_ispc PRIVATE ISPC_SIMD_WIDTH=16)

add_ispc_library(ao_ispc ao.ispc)
target_link_libraries(ao_ispc PRIVATE ispcrt::ispcrt)

# ISPC-SYCL version with binary linking
add_executable(invoke_sycl_aobench_ispc_sycl_bin invoke_sycl_aobench.cpp)
target_compile_definitions(invoke_sycl_aobench_ispc_sycl_bin PRIVATE ISPC_SIMD_WIDTH=16 AOBENCH_SYCL)
target_link_libraries(invoke_sycl_aobench_ispc_sycl_bin PRIVATE ispcrt::ispcrt ze_loader)

# ISPC-SYCL version with vISA linking
add_executable(invoke_sycl_aobench_ispc_sycl_visa invoke_sycl_aobench.cpp)
target_compile_definitions(invoke_sycl_aobench_ispc_sycl_visa PRIVATE ISPC_SIMD_WIDTH=16 AOBENCH_SYCL VISA_LINKING)
target_link_libraries(invoke_sycl_aobench_ispc_sycl_visa PRIVATE ispcrt::ispcrt ze_loader)

# Compile ISPC kernel
add_ispc_library(ao_ispc_sycl ao_sycl.ispc)
# Link it with SYCL library
add_dpcpp_library(ao_sycl SPV ao_sycl_lib.cpp)
# Explicitly specify dependency between ISPC and SYCL targets
add_dependencies(ao_ispc_sycl ao_sycl)

target_link_libraries(ao_ispc_sycl PRIVATE ispcrt::ispcrt)