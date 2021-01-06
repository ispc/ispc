#
#  Copyright (c) 2018-2020, Intel Corporation
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
# ispc ADDISPCTest.cmake
#
function(add_ispc_example)
    set(options USE_COMMON_SETTINGS)
    set(oneValueArgs NAME ISPC_SRC_NAME DATA_DIR)
    set(multiValueArgs ISPC_IA_TARGETS ISPC_ARM_TARGETS ISPC_FLAGS TARGET_SOURCES LIBRARIES DATA_FILES)
    cmake_parse_arguments("example" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    if ("${ISPC_ARCH}" MATCHES "x86")
        string(REPLACE "," ";" ISPC_TARGETS ${example_ISPC_IA_TARGETS})
    elseif ("${ISPC_ARCH}" STREQUAL "arm" OR "${ISPC_ARCH}" STREQUAL "aarch64")
        string(REPLACE "," ";" ISPC_TARGETS ${example_ISPC_ARM_TARGETS})
    else()
        message(FATAL_ERROR "Unknown architecture ${ISPC_ARCH}")
    endif()

    add_executable(${example_NAME})
    target_sources(${example_NAME}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/${ISPC_SRC_NAME}.ispc"
            ${example_TARGET_SOURCES}
        )

    # Set C++ standard to C++11.
    set_target_properties(${example_NAME} PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES)

    set_property(TARGET ${example_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET ${example_NAME} PROPERTY ISPC_INSTRUCTION_SETS "${ISPC_TARGETS}")
    target_compile_options(${example_NAME} PRIVATE $<$<COMPILE_LANGUAGE:ISPC>:${example_ISPC_FLAGS}>)
    target_compile_options(${example_NAME} PRIVATE $<$<COMPILE_LANGUAGE:ISPC>:--arch=${ISPC_ARCH}>)

    if (UNIX)
        set(arch_flag "-m${ISPC_ARCH_BIT}")
        target_compile_options(${example_NAME} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${arch_flag}>)
    else()
        target_compile_options(${example_NAME} PRIVATE  $<$<COMPILE_LANGUAGE:C,CXX>:"/fp:fast;/Oi">)
    endif()

    if (example_USE_COMMON_SETTINGS)
        find_package(Threads)
        target_sources(${example_NAME} PRIVATE ${EXAMPLES_ROOT}/common/tasksys.cpp)
        target_sources(${example_NAME} PRIVATE ${EXAMPLES_ROOT}/common/timing.h)
        target_link_libraries(${example_NAME} PRIVATE Threads::Threads)
    endif()

    # Link libraries
    if (example_LIBRARIES)
        target_link_libraries(${example_NAME} ${example_LIBRARIES})
    endif()

    set_target_properties(${example_NAME} PROPERTIES FOLDER "Examples")
    if(MSVC)
        # Group ISPC files inside Visual Studio
        source_group("ISPC" FILES "${CMAKE_CURRENT_SOURCE_DIR}/${ISPC_SRC_NAME}.ispc")
    endif()

endfunction()
