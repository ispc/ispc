#
#  Copyright (c) 2021-2023, Intel Corporation
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

#
# ispc examples: callback-esimd
#

cmake_minimum_required(VERSION 3.13)
project(callback-esimd)

set(TARGET_NAME "callback-esimd")

set(DPCPP_CUSTOM_FLAGS "-Xclang" "-fsycl-allow-func-ptr")

add_executable(${TARGET_NAME} main.cpp)
add_dpcpp_library(${TARGET_NAME}_esimd callback-esimd.cpp)

add_ispc_library(${TARGET_NAME}_ispc2esimd callback-esimd.ispc)
# Link for GPU
ispc_target_link_dpcpp_esimd_libraries(${TARGET_NAME}_ispc2esimd ${TARGET_NAME}_esimd)
# Link for CPU
target_link_libraries(${TARGET_NAME}_ispc2esimd ${TARGET_NAME}_esimd)

target_compile_definitions(${TARGET_NAME} PRIVATE ISPCRT)
target_link_libraries(${TARGET_NAME} PRIVATE ispcrt::ispcrt ${LEVEL_ZERO_LIB_LOADER})
target_include_directories(${TARGET_NAME} PRIVATE "${COMMON_PATH}" "${LEVEL_ZERO_INCLUDE_DIR}" "${ISPC_INCLUDE_DIR}")

set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

