#
#  Copyright (c) 2018-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: volume_rendering
#
set (ISPC_SRC_NAME "volume")
set (TARGET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/volume.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/volume_serial.cpp)
set (ISPC_IA_TARGETS "sse2-i32x4,sse4-i32x8,avx1-i32x8,avx2-i32x8,avx512knl-x16,avx512skx-x16" CACHE STRING "ISPC IA targets")
set (ISPC_ARM_TARGETS "neon" CACHE STRING "ISPC ARM targets")
set (DATA_FILES ${CMAKE_CURRENT_SOURCE_DIR}/camera.dat
                ${CMAKE_CURRENT_SOURCE_DIR}/density_highres.vol
                ${CMAKE_CURRENT_SOURCE_DIR}/density_lowres.vol)
add_ispc_example(NAME "volume_rendering"
              ISPC_IA_TARGETS ${ISPC_IA_TARGETS}
              ISPC_ARM_TARGETS ${ISPC_ARM_TARGETS}
              ISPC_SRC_NAME ${ISPC_SRC_NAME}
              TARGET_SOURCES ${TARGET_SOURCES}
              USE_COMMON_SETTINGS
              DATA_FILES ${DATA_FILES}
              )
