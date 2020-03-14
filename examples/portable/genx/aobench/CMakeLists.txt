#
#  Copyright (c) 2020, Intel Corporation
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
# ispc examples: aobench
#

cmake_minimum_required(VERSION 3.10)

set(ISPC_SRC_NAME "aobench")
set(TEST_NAME "genx_aobench")

#Linux test only
if(UNIX)
    set(HOST_NAME "l0host")
    set(ISPC_OBJ_NAME "${CMAKE_CURRENT_BINARY_DIR}/aobench_ispc.spv")
    set(ISPC_FLAGS "--emit-spirv" "--addressing=64" "--arch=genx64")

    set(HOST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ao.cpp ${CMAKE_CURRENT_SOURCE_DIR}/ao_serial.cpp)

    add_perf_example(
        NAME "aobench"
        ISPC_SRC_NAME ${ISPC_SRC_NAME}
        ISPC_OBJ_NAME ${ISPC_OBJ_NAME}
        HOST_NAME ${HOST_NAME}
        TEST_NAME ${TEST_NAME}
        ISPC_FLAGS ${ISPC_FLAGS}
        HOST_SOURCES ${HOST_SOURCES}
    )
endif()