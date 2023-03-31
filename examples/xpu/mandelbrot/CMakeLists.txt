#
#  Copyright (c) 2020-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: mandelbrot
#

cmake_minimum_required(VERSION 3.13)

set(TEST_NAME "mandelbrot")
set(ISPC_SRC_NAME "mandelbrot.ispc")
set(ISPC_TARGET_XE "gen9-x16")
set(ISPC_TARGET_CPU "sse4-i8x16;avx1-i32x16;avx2-i32x16;avx512knl-x16;avx512skx-x16")
set(HOST_SOURCES mandelbrot.cpp mandelbrot_serial.cpp)

add_perf_example(
    ISPC_SRC_NAME ${ISPC_SRC_NAME}
    TEST_NAME ${TEST_NAME}
    ISPC_TARGET_XE ${ISPC_TARGET_XE}
    HOST_SOURCES ${HOST_SOURCES}
)
