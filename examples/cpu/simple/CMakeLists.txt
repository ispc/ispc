#
#  Copyright (c) 2018-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: simple
#
set (ISPC_SRC_NAME "simple")
set (TARGET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/simple.cpp)
set (ISPC_FLAGS -O2)
set (ISPC_IA_TARGETS "sse2")
set (ISPC_ARM_TARGETS "neon")
add_ispc_example(NAME "simple" ISPC_SRC_NAME ${ISPC_SRC_NAME}
              ISPC_IA_TARGETS ${ISPC_IA_TARGETS}
              ISPC_ARM_TARGETS ${ISPC_ARM_TARGETS}
              ISPC_FLAGS ${ISPC_FLAGS}
              TARGET_SOURCES ${TARGET_SOURCES})
