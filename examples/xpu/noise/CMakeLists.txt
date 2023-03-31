#
#  Copyright (c) 2019-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: noise
#

cmake_minimum_required(VERSION 3.13)

set(TEST_NAME "noise")
set(ISPC_SRC_NAME "noise.ispc")
set(ISPC_TARGET_XE "gen9-x8")
set(HOST_SOURCES noise.cpp noise_serial.cpp)

add_perf_example(
    ISPC_SRC_NAME ${ISPC_SRC_NAME}
    TEST_NAME ${TEST_NAME}
    ISPC_TARGET_XE ${ISPC_TARGET_XE}
    HOST_SOURCES ${HOST_SOURCES}
)

