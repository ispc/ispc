#
#  Copyright (c) 2018-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: gmres
#
set (ISPC_SRC_NAME "matrix")
set (TARGET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/algorithm.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/algorithm.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/debug.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/matrix.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/matrix.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/mmio.c
                    ${CMAKE_CURRENT_SOURCE_DIR}/mmio.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/util.h)
set (ISPC_IA_TARGETS "sse2-i32x4,sse4-i32x8,avx1-i32x16,avx2-i32x16,avx512knl-x16,avx512skx-x16" CACHE STRING "ISPC IA targets")
set (ISPC_ARM_TARGETS "neon" CACHE STRING "ISPC ARM targets")
add_ispc_example(NAME "gmres"
              ISPC_IA_TARGETS ${ISPC_IA_TARGETS}
              ISPC_ARM_TARGETS ${ISPC_ARM_TARGETS}
              ISPC_SRC_NAME ${ISPC_SRC_NAME}
              TARGET_SOURCES ${TARGET_SOURCES}
              USE_COMMON_SETTINGS
              DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)
