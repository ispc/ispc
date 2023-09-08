# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.13)
project(invoke_sycl_aobench LANGUAGES CXX)

include(${ISPCRT_DIR}/interop.cmake)

set(ISPC_TARGET_XE "xehpg-x16")

# ISPC-only version
add_executable(invoke_sycl_aobench_ispc invoke_sycl_aobench.cpp)
target_link_libraries(invoke_sycl_aobench_ispc PRIVATE ispcrt::ispcrt ${LEVEL_ZERO_LIB_LOADER})
target_compile_definitions(invoke_sycl_aobench_ispc PRIVATE ISPC_SIMD_WIDTH=16)

add_ispc_library(ao_ispc ao.ispc)
target_link_libraries(ao_ispc PRIVATE ispcrt::ispcrt)

# ISPC-SYCL version with binary linking
add_executable(invoke_sycl_aobench_ispc_sycl_bin invoke_sycl_aobench.cpp)
target_compile_definitions(invoke_sycl_aobench_ispc_sycl_bin PRIVATE ISPC_SIMD_WIDTH=16 AOBENCH_SYCL_LINKING)
target_link_libraries(invoke_sycl_aobench_ispc_sycl_bin PRIVATE ispcrt::ispcrt ${LEVEL_ZERO_LIB_LOADER})

# ISPC-SYCL version with vISA linking
add_executable(invoke_sycl_aobench_ispc_sycl_visa invoke_sycl_aobench.cpp)
target_compile_definitions(invoke_sycl_aobench_ispc_sycl_visa PRIVATE ISPC_SIMD_WIDTH=16 AOBENCH_SYCL_LINKING VISA_LINKING)
target_link_libraries(invoke_sycl_aobench_ispc_sycl_visa PRIVATE ispcrt::ispcrt ${LEVEL_ZERO_LIB_LOADER})

# Compile ISPC kernel
add_ispc_library(ao_ispc_sycl ao_sycl.ispc)
# Link it with SYCL library
add_dpcpp_library(ao_sycl SPV ao_sycl_lib.cpp)
# Explicitly specify dependency between ISPC and SYCL targets
add_dependencies(ao_ispc_sycl ao_sycl)

target_link_libraries(ao_ispc_sycl PRIVATE ispcrt::ispcrt)

# Pure SYCL version that compiles the ISPC code as SYCL using the "hybrid code" approach
add_executable(aobench_sycl_bin invoke_sycl_aobench.cpp)
target_compile_definitions(aobench_sycl_bin PRIVATE
  ISPC_SIMD_WIDTH=1
  AOBENCH_TARGET_SYCL
  uniform=
  varying=
  unmasked=
  export=
  programCount=1
  programIndex=0
)
target_compile_options(aobench_sycl_bin PUBLIC
  -fsycl
  -fsycl-unnamed-lambda
  -fsycl-targets=spir64
  -Wdouble-promotion
)
list(APPEND OCL_OPTIONS -cl-intel-greater-than-4GB-buffer-required)
list(APPEND OCL_OTHER_OPTIONS
  -cl-intel-force-global-mem-allocation
  -cl-intel-no-local-to-generic)

target_link_options(aobench_sycl_bin PUBLIC
  -fsycl
  -fsycl-targets=spir64
  -Xsycl-target-backend=spir64 "${OCL_OPTIONS_STR} -options \"${OCL_OTHER_OPTIONS_STR}\""
)

target_link_libraries(aobench_sycl_bin PRIVATE ispcrt::ispcrt)

