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

    set(ISPC_KNOWN_TARGETS "sse2" "sse4" "avx1-" "avx2" "avx512knl" "avx512skx" "neon")
    set(ISPC_HEADER_NAME "${CMAKE_CURRENT_BINARY_DIR}/${ISPC_SRC_NAME}_ispc.h")
    set(ISPC_OBJ_NAME "${CMAKE_CURRENT_BINARY_DIR}/${ISPC_SRC_NAME}.ispc${CMAKE_CXX_OUTPUT_EXTENSION}")
    set(ISPC_FLAGS ${example_ISPC_FLAGS})
    if (UNIX)
      list(APPEND ISPC_FLAGS --pic)
    endif()

    # Collect list of expected outputs
    list(APPEND ISPC_BUILD_OUTPUT ${ISPC_HEADER_NAME} ${ISPC_OBJ_NAME})
    if (example_USE_COMMON_SETTINGS)
        if ("${ISPC_ARCH}" MATCHES "x86")
            set(ISPC_TARGETS ${example_ISPC_IA_TARGETS})
            string(FIND ${example_ISPC_IA_TARGETS} "," MULTI_TARGET)
            if (${MULTI_TARGET} GREATER -1)
                foreach (ispc_target ${ISPC_KNOWN_TARGETS})
                    string(FIND ${example_ISPC_IA_TARGETS} ${ispc_target} FOUND_TARGET)
                    if (${FOUND_TARGET} GREATER -1)
                        set(OUTPUT_TARGET ${ispc_target})
                        if (${ispc_target} STREQUAL "avx1-")
                            set(OUTPUT_TARGET "avx")
                        endif()
                        list(APPEND ISPC_BUILD_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${ISPC_SRC_NAME}_ispc_${OUTPUT_TARGET}.h"
                                    "${CMAKE_CURRENT_BINARY_DIR}/${ISPC_SRC_NAME}.ispc_${OUTPUT_TARGET}${CMAKE_CXX_OUTPUT_EXTENSION}")
                    endif()
                endforeach()
            endif()
        elseif ("${ISPC_ARCH}" STREQUAL "arm" OR "${ISPC_ARCH}" STREQUAL "aarch64")
            set(ISPC_TARGETS ${example_ISPC_ARM_TARGETS})
        else()
            message(FATAL_ERROR "Unknown architecture ${ISPC_ARCH}")
        endif()
    else()
        if ("${ISPC_ARCH}" MATCHES "x86")
            set(ISPC_TARGETS ${example_ISPC_IA_TARGETS})
        elseif ("${ISPC_ARCH}" STREQUAL "arm" OR "${ISPC_ARCH}" STREQUAL "aarch64")
            set(ISPC_TARGETS ${example_ISPC_ARM_TARGETS})
        else()
            message(FATAL_ERROR "Unknown architecture ${ISPC_ARCH}")
        endif()

    endif()
    # ISPC command
    add_custom_command(OUTPUT ${ISPC_BUILD_OUTPUT}
        COMMAND ${ISPC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${ISPC_SRC_NAME}.ispc ${ISPC_FLAGS} --target=${ISPC_TARGETS} --arch=${ISPC_ARCH}
                                   -h ${ISPC_HEADER_NAME} -o ${ISPC_OBJ_NAME}
        VERBATIM
        DEPENDS ${ISPC_EXECUTABLE}
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${ISPC_SRC_NAME}.ispc")

    # To show ispc source in VS solution:
    if (WIN32)
        set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/${ISPC_SRC_NAME}.ispc" PROPERTIES HEADER_FILE_ONLY TRUE)
    endif()

    add_executable(${example_NAME} ${ISPC_BUILD_OUTPUT} "${CMAKE_CURRENT_SOURCE_DIR}/${ISPC_SRC_NAME}.ispc")
    target_sources(${example_NAME} PRIVATE ${example_TARGET_SOURCES})
    target_include_directories(${example_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

     # Set C++ standard to C++11.
     set_target_properties(${example_NAME} PROPERTIES
         CXX_STANDARD 11
         CXX_STANDARD_REQUIRED YES)

    # Compile options
    set_property(TARGET ${example_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
    if (UNIX)
        if (${ISPC_ARCH_BIT} EQUAL 32)
            target_compile_options(${example_NAME} PRIVATE -m32)
        else()
            target_compile_options(${example_NAME} PRIVATE -m64)
        endif()
    else()
        target_compile_options(${example_NAME} PRIVATE /fp:fast /Oi)
    endif()

    # Common settings
    if (example_USE_COMMON_SETTINGS)
        target_sources(${example_NAME} PRIVATE ${EXAMPLES_ROOT}/common/tasksys.cpp)
        target_sources(${example_NAME} PRIVATE ${EXAMPLES_ROOT}/common/timing.h)
        if (UNIX)
            target_compile_options(${example_NAME} PRIVATE -O2)
            target_link_libraries(${example_NAME} pthread m stdc++)
        endif()
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

    # Install example
    # We do not need to include examples binaries to the package
    if (NOT ISPC_PREPARE_PACKAGE)
        install(TARGETS ${example_NAME} RUNTIME DESTINATION examples/${example_NAME})
        if (example_DATA_FILES)
            install(FILES ${example_DATA_FILES}
                    DESTINATION examples/${example_NAME})
        endif()

        if (example_DATA_DIR)
            install(DIRECTORY ${example_DATA_DIR}
                    DESTINATION examples/${example_NAME})
        endif()
    endif()

endfunction()
