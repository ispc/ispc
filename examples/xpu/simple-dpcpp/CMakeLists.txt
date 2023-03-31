#
#  Copyright (c) 2021-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: simple-dpcpp
#

cmake_minimum_required(VERSION 3.13)

set(TEST_NAME "simple-dpcpp")
set(ISPC_SRC_NAME "simple-dpcpp.ispc")
set(ISPC_TARGET_XE "gen9-x8")
set(DPCPP_HOST_SOURCES simple-dpcpp.cpp)

if(UNIX)
    add_perf_example(
        ISPC_SRC_NAME ${ISPC_SRC_NAME}
        TEST_NAME ${TEST_NAME}
        ISPC_TARGET_XE ${ISPC_TARGET_XE}
        DPCPP_HOST_SOURCES ${DPCPP_HOST_SOURCES}
    )
endif()
