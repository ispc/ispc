#
#  Copyright (c) 2019-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

function(add_perf_example)
    set(options GBENCH LINK_L0)
    set(oneValueArgs ISPC_SRC_NAME ISPC_TARGET_XE TEST_NAME GBENCH_TEST_NAME)
    set(multiValueArgs ISPC_XE_ADDITIONAL_ARGS HOST_SOURCES DPCPP_HOST_SOURCES GBENCH_SRC_NAME)
    cmake_parse_arguments("parsed" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # Compile host code
    if (NOT HOST_SOURCES AND NOT DPCPP_HOST_SOURCES)
        return()
    endif()

    if (HOST_SOURCES AND DPCPP_HOST_SOURCES)
        message(FATAL_ERROR "Cannot build regular and DPCPP example at the same time")
    endif()

    if (DPCPP_HOST_SOURCES AND NOT ISPC_INCLUDE_DPCPP_EXAMPLES)
        message(STATUS "Skipping ${parsed_TEST_NAME} example. Use ISPC_INCLUDE_DPCPP_EXAMPLES option to enable DPCPP examples")
        return()
    endif()

    set(HOST_EXECUTABLE "host_${parsed_TEST_NAME}")

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    if (HOST_SOURCES)
        add_executable(${HOST_EXECUTABLE} ${parsed_HOST_SOURCES})
    endif()
    if (DPCPP_HOST_SOURCES)
        add_executable(${HOST_EXECUTABLE} ${parsed_DPCPP_HOST_SOURCES})
    endif()

    # Compile device code
    set(ISPC_TARGET_XE ${parsed_ISPC_TARGET_XE})

    if (WIN32)
        set(ISPC_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
    else()
        set(ISPC_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR})
    endif()
    set(ISPC_XE_ADDITIONAL_ARGS ${parsed_ISPC_XE_ADDITIONAL_ARGS})

    # Add "ispcrt" suffix here to avoid CMake target conflicts with CPU examples
    add_ispc_library(xe_${parsed_TEST_NAME} ${parsed_ISPC_SRC_NAME})

    # Show ispc source in VS solution:
    if (WIN32)
        set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/${parsed_ISPC_SRC_NAME}" PROPERTIES HEADER_FILE_ONLY TRUE)
    endif()

    if (DPCPP_HOST_SOURCES)
        target_compile_options(${HOST_EXECUTABLE} PRIVATE "-fsycl")
        target_link_options(${HOST_EXECUTABLE} PRIVATE "-fsycl")
    endif()

    if (parsed_LINK_L0)
        target_include_directories(${HOST_EXECUTABLE} PUBLIC ${LEVEL_ZERO_INCLUDE_DIR})
        target_link_libraries(${HOST_EXECUTABLE} PRIVATE ${LEVEL_ZERO_LIB_LOADER})
    endif()

    target_compile_definitions(${HOST_EXECUTABLE} PRIVATE ISPCRT)
    if (ISPC_BUILD)
        target_link_libraries(${HOST_EXECUTABLE} PRIVATE ${ISPCRT_LIB})
    else()
        target_link_libraries(${HOST_EXECUTABLE} PRIVATE ispcrt::ispcrt)
    endif()

    target_include_directories(${HOST_EXECUTABLE} PRIVATE "${COMMON_PATH}" "${LEVEL_ZERO_INCLUDE_DIR}" "${ISPC_INCLUDE_DIR}" )

    target_link_libraries(${HOST_EXECUTABLE} PRIVATE ${LEVEL_ZERO_LIB_LOADER})
    set_target_properties(${HOST_EXECUTABLE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    # Compile Google Benchmark executable if requested
    if (parsed_GBENCH AND ISPC_INCLUDE_BENCHMARKS)
        if (WIN32)
            message(WARNING "Benchmarks are not supported on Windows")
        else()
            set(GBENCH_TEST_NAME "${parsed_GBENCH_TEST_NAME}")
            set(GBENCH_SRC_NAME "${parsed_GBENCH_SRC_NAME}")
            add_executable(${GBENCH_TEST_NAME} ${parsed_GBENCH_SRC_NAME})
            target_compile_definitions(${GBENCH_TEST_NAME} PRIVATE ISPCRT)
            target_include_directories(${GBENCH_TEST_NAME} PRIVATE ${COMMON_PATH} ${LEVEL_ZERO_INCLUDE_DIR} ${GBENCH_INCLUDE_DIR})
            link_directories(${CMAKE_BINARY_DIR})
            if (ISPC_BUILD)
                target_link_libraries(${GBENCH_TEST_NAME} PRIVATE ${ISPCRT_LIB} ${BENCHMARK_LIB} ${LEVEL_ZERO_LIB_LOADER} pthread)
            else()
                target_link_libraries(${GBENCH_TEST_NAME} PRIVATE ispcrt::ispcrt benchmark::benchmark)
            endif()
            set_target_properties(${GBENCH_TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        endif()
    endif()
endfunction()
