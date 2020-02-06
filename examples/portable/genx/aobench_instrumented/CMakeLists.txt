#
#  Copyright (c) 2018, Intel Corporation
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

#
# ispc examples: aobench_instrumented
#

cmake_minimum_required(VERSION 3.10)
option(L0_LOCAL_BUILD "Use local build of L0" ON)
if (L0_LOCAL_BUILD AND NOT NEO_INSTALL_PATH)
    message (FATAL_ERROR "Set NEO_INSTALL_PATH variable")
endif()
list(APPEND ISPC_BUILD_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ao_instrumented_ispc.h ${CMAKE_CURRENT_BINARY_DIR}/ao_instrumented_ispc.spv)

add_custom_command(OUTPUT ${ISPC_BUILD_OUTPUT}
    COMMAND ispc ${PROJECT_SOURCE_DIR}/ao_instrumented.ispc ${ISPC_FLAGS} --pic -O3 --target=genx --arch=genx64 --addressing=64 --emit-spirv
    -h ${CMAKE_CURRENT_BINARY_DIR}/ao_instrumented_ispc.h -o ${CMAKE_CURRENT_BINARY_DIR}/ao_instrumented_ispc.spv
    VERBATIM
    DEPENDS "ao_instrumented.ispc")


add_executable(ao ${ISPC_BUILD_OUTPUT})

target_sources(ao PRIVATE ao.cpp ${EXAMPLES_ROOT}/tasksys.cpp)
target_compile_options(ao PRIVATE -fPIE -fPIC -std=c++11)
#target_compile_definitions(ao PRIVATE)
target_include_directories(ao PRIVATE
                           ${NEO_INSTALL_PATH}/neo/usr/local/include
                           ${EXAMPLES_ROOT}
                           ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ao pthread m level_zero -L${NEO_INSTALL_PATH}/neo/usr/local/lib)

