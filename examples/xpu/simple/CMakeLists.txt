#
#  Copyright (c) 2020-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: simple
#

cmake_minimum_required(VERSION 3.13)

set(TEST_NAME "simple")
set(ISPC_SRC_NAME "simple.ispc")
set(ISPC_TARGET_XE "gen9-x8")
set(HOST_SOURCES simple.cpp)

add_perf_example(
    ISPC_SRC_NAME ${ISPC_SRC_NAME}
    TEST_NAME ${TEST_NAME}
    ISPC_TARGET_XE ${ISPC_TARGET_XE}
    HOST_SOURCES ${HOST_SOURCES}
)

