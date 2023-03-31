#
#  Copyright (c) 2018-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: sgemm
#
set (ISPC_SRC_NAME "SGEMM_kernels")
set (TARGET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/SGEMM_main.cpp)
set (ISPC_IA_TARGETS "sse2-i32x4,sse4-i32x8,avx1-i32x8,avx2-i32x8,avx512knl-x16,avx512skx-x16" CACHE STRING "ISPC IA targets")
set (ISPC_ARM_TARGETS "neon" CACHE STRING "ISPC ARM targets")
add_ispc_example(NAME "sgemm"
              ISPC_IA_TARGETS ${ISPC_IA_TARGETS}
              ISPC_ARM_TARGETS ${ISPC_ARM_TARGETS}
              ISPC_SRC_NAME ${ISPC_SRC_NAME}
              TARGET_SOURCES ${TARGET_SOURCES}
              USE_COMMON_SETTINGS)
