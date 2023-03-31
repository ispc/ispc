#
#  Copyright (c) 2018-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc Git.cmake
#
find_program(GIT_BINARY NAMES git)
    message(STATUS "GIT_BINARY: ${GIT_BINARY}")
    if (GIT_BINARY)
        execute_process(
            COMMAND ${GIT_BINARY} rev-parse --short=16 HEAD
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_COMMIT 
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    string(CONCAT GIT_COMMIT_HASH "commit " ${GIT_COMMIT})
    endif()
