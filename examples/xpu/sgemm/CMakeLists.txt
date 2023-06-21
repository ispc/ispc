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

add_perf_example(
    ISPC_SRC_NAME ${ISPC_SRC_NAME}
    TEST_NAME ${TEST_NAME}
    ISPC_TARGET_XE ${ISPC_TARGET_XE}
    HOST_SOURCES ${HOST_SOURCES}
    GBENCH
    GBENCH_TEST_NAME bench-sgemm
    GBENCH_SRC_NAME bench.cpp sgemm.cpp
    LINK_L0
)
