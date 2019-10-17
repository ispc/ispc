#
#  Copyright (c) 2019, Intel Corporation
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
#   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.12)

project(GenxExamples)

if (NOT MDF)
    message(FATAL_ERROR "Provide path to MDF root")
endif()

if (CMC_INSTALL_PATH)
    find_program (CM_EXECUTABLE cmc)
else()
    if (WIN32)
        set(CM_EXECUTABLE ${MDF}/compiler/bin/cmc)
    endif()
endif()

if (NOT CM_EXECUTABLE)
    message(FATAL_ERROR "Failed to find cmc")
endif()
message(STATUS "Found CMC: ${CM_EXECUTABLE}")

if (ISPC_BUILD)
    set(ISPC_EXECUTABLE $<TARGET_FILE:ispc>)
else()
    find_program(ISPC_EXECUTABLE ispc)
endif()
if (NOT ISPC_EXECUTABLE)
    message(FATAL_ERROR "Failed to find ispc")
endif()
message(STATUS "Found ISPC: ${ISPC_EXECUTABLE}")


# Used for testing
find_package(Python3 REQUIRED)
if (NOT Python3_Interpreter_FOUND)
    message(FATAL_ERROR "Failed to find python3 interpretor")
endif()
include(FindPythonInterp)

set(COMMON_PATH "${CMAKE_CURRENT_SOURCE_DIR}/common")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AddPerfExample.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestDrivers.cmake)

add_subdirectory(noise)
add_subdirectory(sgemm)

include(TestLists.txt)

