#
#  Copyright (c) 2019-2021, Intel Corporation
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
# ispc examples: sgemm
#

cmake_minimum_required(VERSION 3.13)

set(TEST_NAME "sgemm")
set(ISPC_SRC_NAME "naive_sgemm.ispc")
set(ISPC_TARGET_XE "gen9-x16")
set(HOST_SOURCES sgemm.cpp main.cpp)

set(CM_SRC_NAME "NaiveCMSgemm_xe")
set(CM_OBJ_NAME "${CMAKE_CURRENT_BINARY_DIR}/naive_sgemm_cm_mt.spv")
set(CM_TEST_NAME "naive_sgemm_cm")
set(CM_HOST_SOURCES ${HOST_SOURCES})

add_perf_example(
    ISPC_SRC_NAME ${ISPC_SRC_NAME}
    TEST_NAME ${TEST_NAME}
    ISPC_TARGET_XE ${ISPC_TARGET_XE}
    HOST_SOURCES ${HOST_SOURCES}
    GBENCH
    GBENCH_TEST_NAME bench-sgemm
    GBENCH_SRC_NAME bench.cpp sgemm.cpp
    CM_TEST
    CM_SRC_NAME ${CM_SRC_NAME}
    CM_TEST_NAME ${CM_TEST_NAME}
    CM_HOST_SOURCES ${CM_HOST_SOURCES}
    CM_OBJ_NAME ${CM_OBJ_NAME}
)
