#
#  Copyright (c) 2018-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: aobench_instrumented
#
set (ISPC_SRC_NAME "ao_instrumented")
set (TARGET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ao.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/instrument.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/instrument.h
                    ${EXAMPLES_ROOT}/common/timing.h
                    ${EXAMPLES_ROOT}/common/tasksys.cpp)
set (ISPC_FLAGS -O2 --instrument)
set (ISPC_IA_TARGETS "sse2")
set (ISPC_ARM_TARGETS "neon")
if (WIN32)
    set (LIBRARIES)
else()
    set (LIBRARIES pthread m)
endif()
add_ispc_example(NAME "aobench_instrumented"
              ISPC_SRC_NAME ${ISPC_SRC_NAME}
              ISPC_IA_TARGETS ${ISPC_IA_TARGETS}
              ISPC_ARM_TARGETS ${ISPC_ARM_TARGETS}
              ISPC_FLAGS ${ISPC_FLAGS}
              TARGET_SOURCES ${TARGET_SOURCES}
              LIBRARIES ${LIBRARIES})
