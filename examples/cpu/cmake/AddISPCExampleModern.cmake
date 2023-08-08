#
#  Copyright (c) 2018-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

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
    elseif (WIN32 AND MSVC)
        target_compile_options(${example_NAME} PRIVATE  $<$<COMPILE_LANGUAGE:C,CXX>:/fp:fast /Oi>)
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
