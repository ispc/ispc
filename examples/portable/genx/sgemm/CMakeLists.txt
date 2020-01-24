#
#  Copyright (c) 2019, Intel Corporation
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

message(STATUS "Processing sgemm/CMakeLists.txt")
cmake_minimum_required(VERSION 3.12)

set(ISPC_SRC_NAME "naive_sgemm")

if(WIN32)
    set(HOST_NAME "host")
    set(TEST_NAME "naive_sgemm")
    set(ISPC_OBJ_NAME "${CMAKE_CURRENT_BINARY_DIR}/naive_sgemm_mt.isa")
    set(HOST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_NAME}.cpp)

    set(CM_OBJ_NAME "${CMAKE_CURRENT_BINARY_DIR}/naive_sgemm_mt_cm.isa")
    set(CM_SRC_NAME "NaiveCMSgemm_genx")

    set(CM_HOST_NAME "SgemmSVMMultithread")
    #set(CM_HOST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${CM_HOST_NAME}.cpp)
else()
    set(HOST_NAME "l0host")
    set(TEST_NAME "naive_sgemm")
    set(ISPC_OBJ_NAME "${CMAKE_CURRENT_BINARY_DIR}/naive_sgemm_mt.spv")
    set(ISPC_FLAGS "--emit-spirv" "--addressing=32" "-DISPC_GENX_ENABLED")

    set(CM_OBJ_NAME "${CMAKE_CURRENT_BINARY_DIR}/naive_sgemm_cm_mt.spv")
    set(CM_SRC_NAME "NaiveCMSgemm_genx")

    set(CM_TEST_NAME "naive_sgemm_cm")
    set(HOST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_NAME}.cpp)
endif()




add_perf_example(
    NAME "sgemm"
    CM_TEST
    ISPC_SRC_NAME ${ISPC_SRC_NAME} 
    CM_SRC_NAME ${CM_SRC_NAME}
    ISPC_OBJ_NAME ${ISPC_OBJ_NAME} 
    HOST_NAME ${HOST_NAME} 
    CM_OBJ_NAME ${CM_OBJ_NAME}
    TEST_NAME ${TEST_NAME} 
    CM_TEST_NAME ${CM_TEST_NAME}
    ISPC_FLAGS ${ISPC_FLAGS}    
    CM_HOST_NAME ${CM_HOST_NAME}
    HOST_SOURCES ${HOST_SOURCES} 
    CM_HOST_SOURCES ${HOST_SOURCES}
    CM_HOST_FLAGS "/DCMHOST"
)


