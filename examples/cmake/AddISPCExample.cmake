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
# ispc ADDISPCTest.cmake
#
function(add_ispc_example)
    set(options USE_COMMON_SETTINGS)
    set(oneValueArgs NAME ISPC_SRC_NAME DATA_DIR)
    set(multiValueArgs ISPC_IA_TARGETS ISPC_ARM_TARGETS ISPC_FLAGS TARGET_SOURCES LIBRARIES DATA_FILES)
    cmake_parse_arguments("example" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    set(ISPC_KNOWN_TARGETS "sse2" "sse4" "avx1-" "avx1.1" "avx2" "avx512knl" "avx512skx")
    set(ISPC_HEADER_NAME "${CMAKE_CURRENT_BINARY_DIR}/${ISPC_SRC_NAME}_ispc.h")
    set(ISPC_OBJ_NAME "${CMAKE_CURRENT_BINARY_DIR}/${ISPC_SRC_NAME}_ispc${CMAKE_CXX_OUTPUT_EXTENSION}")

    if (UNIX)
        execute_process( COMMAND bash "-c" "uname -m | sed -e s/x86_64/x86/ -e s/i686/x86/ -e s/arm.*/arm/ -e s/sa110/arm/" OUTPUT_VARIABLE ARCH)
        string(STRIP ${ARCH} ARCH)
        execute_process( COMMAND getconf LONG_BIT OUTPUT_VARIABLE ARCH_BIT)
        string(STRIP ${ARCH_BIT} ARCH_BIT)
        if (${ARCH_BIT} EQUAL 32)
            set(ISPC_ARCH "x86")
        else()
            set(ISPC_ARCH "x86-64")
        endif()
    else()
        set(ARCH "x86")
        if (CMAKE_SIZEOF_VOID_P EQUAL 8 )
            set(ISPC_ARCH "x86-64")
        else()
            set(ISPC_ARCH "x86")
        endif()
    endif()

    # Collect list of expected outputs
    list(APPEND ISPC_BUILD_OUTPUT ${ISPC_HEADER_NAME} ${ISPC_OBJ_NAME})
    if (example_USE_COMMON_SETTINGS)
        if ("${ARCH}" STREQUAL "x86")
            set(ISPC_TARGETS ${example_ISPC_IA_TARGETS})
            string(FIND ${example_ISPC_IA_TARGETS} "," MULTI_TARGET)
            if (${MULTI_TARGET} GREATER -1)
                foreach (ispc_target ${ISPC_KNOWN_TARGETS})
                    string(FIND ${example_ISPC_IA_TARGETS} ${ispc_target} FOUND_TARGET)
                    if (${FOUND_TARGET} GREATER -1)
                        set(OUTPUT_TARGET ${ispc_target})
                        if (${ispc_target} STREQUAL "avx1-")
                            set(OUTPUT_TARGET "avx")
                        elseif (${ispc_target} STREQUAL "avx1.1")
                            set(OUTPUT_TARGET "avx11")
                        endif()
                        list(APPEND ISPC_BUILD_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${ISPC_SRC_NAME}_ispc_${OUTPUT_TARGET}.h"
                                    "${CMAKE_CURRENT_BINARY_DIR}/${ISPC_SRC_NAME}_ispc_${OUTPUT_TARGET}${CMAKE_CXX_OUTPUT_EXTENSION}")
                    endif()
                endforeach()
            endif()
        elseif ("${ARCH}" STREQUAL "arm")
            set(ISPC_TARGETS ${example_ISPC_ARM_TARGETS})
        else()
            message(FATAL_ERROR "Unknown architecture ${ARCH}")
        endif()
    else()
        set(ISPC_TARGETS ${example_ISPC_IA_TARGETS})
    endif()
    # ISPC command
    add_custom_command(OUTPUT ${ISPC_BUILD_OUTPUT}
        COMMAND ${ISPC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${ISPC_SRC_NAME}.ispc ${example_ISPC_FLAGS} --target=${ISPC_TARGETS} --arch=${ISPC_ARCH}
                                    -h ${ISPC_HEADER_NAME} -o ${ISPC_OBJ_NAME}
        VERBATIM
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${ISPC_SRC_NAME}.ispc")

    # To show ispc source in VS solution:
    if (WIN32)
        set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/${ISPC_SRC_NAME}.ispc" PROPERTIES HEADER_FILE_ONLY TRUE)
    endif()

    add_executable(${example_NAME} ${ISPC_BUILD_OUTPUT} "${CMAKE_CURRENT_SOURCE_DIR}/${ISPC_SRC_NAME}.ispc")
    target_sources(${example_NAME} PRIVATE ${example_TARGET_SOURCES})
    target_include_directories(${example_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    # Compile options
    if (UNIX)
        if (${ARCH_BIT} EQUAL 32)
            target_compile_options(${example_NAME} PRIVATE -m32)
        else()
            target_compile_options(${example_NAME} PRIVATE -m64)
        endif()
    else()
        target_compile_options(${example_NAME} PRIVATE /fp:fast /Oi)
    endif()

    # Common settings
    if (example_USE_COMMON_SETTINGS)
        target_sources(${example_NAME} PRIVATE ${EXAMPLES_ROOT}/tasksys.cpp)
        target_sources(${example_NAME} PRIVATE ${EXAMPLES_ROOT}/timing.h)
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
