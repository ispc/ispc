#
#  Copyright (c) 2021-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: callback-esimd
#

cmake_minimum_required(VERSION 3.13)
project(callback-esimd)

set(TARGET_NAME "callback-esimd")

set(DPCPP_CUSTOM_FLAGS "-Xclang" "-fsycl-allow-func-ptr")

add_executable(${TARGET_NAME} main.cpp)
add_dpcpp_library(${TARGET_NAME}_esimd callback-esimd.cpp)

# Use LLVM bitcode linking for interop with ESIMD
set(ISPC_XE_FORMAT "bc")
add_ispc_library(${TARGET_NAME}_ispc2esimd callback-esimd.ispc)
# Link for GPU
ispc_target_link_dpcpp_esimd_libraries(${TARGET_NAME}_ispc2esimd ${TARGET_NAME}_esimd)
# Link for CPU
target_link_libraries(${TARGET_NAME}_ispc2esimd ${TARGET_NAME}_esimd)

target_compile_definitions(${TARGET_NAME} PRIVATE ISPCRT)
target_link_libraries(${TARGET_NAME} PRIVATE ispcrt::ispcrt ${LEVEL_ZERO_LIB_LOADER})
target_include_directories(${TARGET_NAME} PRIVATE "${COMMON_PATH}" "${LEVEL_ZERO_INCLUDE_DIR}" "${ISPC_INCLUDE_DIR}")

set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

