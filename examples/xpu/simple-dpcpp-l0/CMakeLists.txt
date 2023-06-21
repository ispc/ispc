#
#  Copyright (c) 2020-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: simple-dpcpp-l0
#

cmake_minimum_required(VERSION 3.13)

set(TEST_NAME "simple-dpcpp-l0")
set(ISPC_SRC_NAME "simple-dpcpp-l0.ispc")
set(ISPC_TARGET_XE "gen9-x8")
set(DPCPP_HOST_SOURCES simple-dpcpp-l0.cpp)

if(UNIX)
    add_perf_example(
        ISPC_SRC_NAME ${ISPC_SRC_NAME}
        TEST_NAME ${TEST_NAME}
        ISPC_TARGET_XE ${ISPC_TARGET_XE}
        DPCPP_HOST_SOURCES ${DPCPP_HOST_SOURCES}
        LINK_L0
    )
endif()
