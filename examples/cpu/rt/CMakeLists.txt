#
#  Copyright (c) 2018-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: rt
#
set (ISPC_SRC_NAME "rt")
set (TARGET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/rt.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/rt_serial.cpp)
set (ISPC_IA_TARGETS "sse2-i32x4,sse4-i32x8,avx1-i32x8,avx2-i32x8,avx512knl-x16,avx512skx-x16" CACHE STRING "ISPC IA targets")
set (ISPC_ARM_TARGETS "neon" CACHE STRING "ISPC ARM targets")
set (DATA_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cornell.bvh
                ${CMAKE_CURRENT_SOURCE_DIR}/cornell.camera
                ${CMAKE_CURRENT_SOURCE_DIR}/sponza.bvh
                ${CMAKE_CURRENT_SOURCE_DIR}/sponza.camera
                ${CMAKE_CURRENT_SOURCE_DIR}/teapot.bvh
                ${CMAKE_CURRENT_SOURCE_DIR}/teapot.camera)
add_ispc_example(NAME "rt"
              ISPC_IA_TARGETS ${ISPC_IA_TARGETS}
              ISPC_ARM_TARGETS ${ISPC_ARM_TARGETS}
              ISPC_SRC_NAME ${ISPC_SRC_NAME}
              TARGET_SOURCES ${TARGET_SOURCES}
              DATA_FILES ${DATA_FILES}
              USE_COMMON_SETTINGS)
