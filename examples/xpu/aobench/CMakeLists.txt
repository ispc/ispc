#
#  Copyright (c) 2020-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: aobench
#

cmake_minimum_required(VERSION 3.13)

set(TEST_NAME "aobench")
set(ISPC_SRC_NAME "aobench.ispc")
set(ISPC_TARGET_XE "gen9-x8")
set(ISPC_TARGET_CPU "sse2-i32x8;sse4-i32x8;avx1-i32x8;avx2-i32x8;avx512skx-x8")
set(HOST_SOURCES ao.cpp ao_serial.cpp)

add_perf_example(
    ISPC_SRC_NAME ${ISPC_SRC_NAME}
    TEST_NAME ${TEST_NAME}
    ISPC_TARGET_XE ${ISPC_TARGET_XE}
    HOST_SOURCES ${HOST_SOURCES}
)
