#
#  Copyright (c) 2020-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: pipeline-dpcpp
#

cmake_minimum_required(VERSION 3.13)

set(TEST_NAME "pipeline-dpcpp")
set(ISPC_SRC_NAME "pipeline-dpcpp.ispc")
set(ISPC_TARGET_XE "gen9-x8")
set(DPCPP_HOST_SOURCES pipeline-dpcpp.cpp)

if(UNIX)
    add_perf_example(
        ISPC_SRC_NAME ${ISPC_SRC_NAME}
        TEST_NAME ${TEST_NAME}
        ISPC_TARGET_XE ${ISPC_TARGET_XE}
        DPCPP_HOST_SOURCES ${DPCPP_HOST_SOURCES}
        LINK_L0
    )
endif()
