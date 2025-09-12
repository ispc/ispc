#
#  Copyright (c) 2018-2025, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: deferred_shading
#
set (ISPC_SRC_NAME "kernels")
set (TARGET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/deferred.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_c.cpp)
set (ISPC_FLAGS "--opt=fast-math")
set (ISPC_IA_TARGETS "sse2-i32x4,sse4-i32x8,avx1-i32x16,avx2-i32x16,avx512skx-x16")
set (ISPC_ARM_TARGETS "neon")
add_ispc_example(NAME "deferred_shading"
              ISPC_IA_TARGETS ${ISPC_IA_TARGETS}
              ISPC_ARM_TARGETS ${ISPC_ARM_TARGETS}
              ISPC_FLAGS ${ISPC_FLAGS}
              ISPC_SRC_NAME ${ISPC_SRC_NAME}
              TARGET_SOURCES ${TARGET_SOURCES}
              USE_COMMON_SETTINGS
              DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data
              RUN_ARGS "${CMAKE_CURRENT_SOURCE_DIR}/data/pp1280x720.bin")
