#
#  Copyright (c) 2019-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

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
