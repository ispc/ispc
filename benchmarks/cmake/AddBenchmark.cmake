#
#  Copyright (c) 2020-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

if (ISPC_BUILD)
    set (ISPC_EXECUTABLE $<TARGET_FILE:ispc>)
else()
    find_program (ISPC_EXECUTABLE ispc)
    if (NOT ISPC_EXECUTABLE)
        message(FATAL_ERROR "Failed to find ispc" )
    endif()
endif()

# Identify host arch
if(UNIX)
    execute_process(COMMAND sh "-c" "uname -m | sed -e s/x86_64/x86/ -e s/i686/x86/ -e s/arm.*/arm/ -e s/sa110/arm/" OUTPUT_VARIABLE ARCH)
    string(STRIP ${ARCH} ARCH)
    execute_process(COMMAND getconf LONG_BIT OUTPUT_VARIABLE ARCH_BIT)
    string(STRIP ${ARCH_BIT} ARCH_BIT)
    if("${ARCH}" STREQUAL "x86")
        if(${ARCH_BIT} EQUAL 32)
            set(ISPC_ARCH "x86")
        else()
            set(ISPC_ARCH "x86-64")
        endif()
    elseif("${ARCH}" STREQUAL "arm")
        if(${ARCH_BIT} EQUAL 32)
            set(ISPC_ARCH "arm")
        else()
            set(ISPC_ARCH "aarch64")
        endif()
    else()
        message(FATAL_ERROR "Cannot detect host architecture for benchamrks: ${ARCH}.")
    endif()
else()
    set(ARCH "x86")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8 )
        set(ISPC_ARCH "x86-64")
    else()
        set(ISPC_ARCH "x86")
    endif()
endif()

# Suffixes for multi-target compilation (x86 only)
set(ISPC_KNOWN_TARGETS "sse2" "sse4" "avx1" "avx2" "avx512knl" "avx512skx")

#######################
#  add_ispc_to_target
#######################
#
#  Adds a ISPC compilation custom command associated with an existing
#  target and sets a dependancy on that new command.
#
#  TARGET : Name of the target to add ISPC to.
#  CPP_MAIN_FILE : Main cpp file which includes ispc headers
#  SOURCES : List of ISPC source files.
#
function(add_ispc_to_target)
    set(options)
    set(one_value_args
        TARGET
        CPP_MAIN_FILE
    )
    set(multi_value_args
        SOURCES
    )
    cmake_parse_arguments("ADD_ISPC"
        "${options}"
        "${one_value_args}"
        "${multi_value_args}"
        ${ARGN}
    )

    set(ISPC_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/CMakeFiles/ispc/")
    file(TO_NATIVE_PATH "${ISPC_DST_DIR}" ISPC_DST_DIR)
    file(MAKE_DIRECTORY ${ISPC_DST_DIR})

    string(FIND ${BENCHMARKS_ISPC_TARGETS} "," MULTI_TARGET)

    foreach(ISPC_SRC_FILE ${ADD_ISPC_SOURCES})
        set(ISPC_TARGET_HEADERS "")
        set(ISPC_TARGET_OBJS "")

        # Full path to source file
        get_filename_component(SRC_LOCATION "${ISPC_SRC_FILE}" ABSOLUTE "${CMAKE_CURRENT_SOURCE_DIR}")

        # Construct names for header and object files
        string(REPLACE ".ispc" "${CMAKE_CXX_OUTPUT_EXTENSION}" ISPC_OBJ "${ISPC_SRC_FILE}")
        set(ISPC_OBJ "${ISPC_DST_DIR}${ISPC_OBJ}")
        string(REPLACE ".ispc" "_ispc.h" ISPC_HEADER "${ISPC_SRC_FILE}")
        set(ISPC_HEADER "${ISPC_DST_DIR}${ISPC_HEADER}")
        list(APPEND ISPC_TARGET_OBJS "${ISPC_OBJ}")
        list(APPEND ISPC_TARGET_HEADERS "${ISPC_HEADER}")

        # Collect list of expected outputs in case of multiple targets
        if(${MULTI_TARGET} GREATER -1)
            foreach (ISPC_TARGET ${ISPC_KNOWN_TARGETS})
                string(FIND ${BENCHMARKS_ISPC_TARGETS} ${ISPC_TARGET} FOUND_TARGET)
                if(${FOUND_TARGET} GREATER -1)
                    set(OUTPUT_TARGET ${ISPC_TARGET})
                    if (${ISPC_TARGET} STREQUAL "avx1")
                        set(OUTPUT_TARGET "avx")
                    endif()
                    string(REPLACE ".ispc" "_${OUTPUT_TARGET}${CMAKE_CXX_OUTPUT_EXTENSION}" ISPC_TARGET_OBJ ${ISPC_SRC_FILE})
                    set(ISPC_TARGET_OBJ ${ISPC_DST_DIR}${ISPC_TARGET_OBJ})
                    list(APPEND ISPC_TARGET_OBJS ${ISPC_TARGET_OBJ})
                    string(REPLACE ".ispc" "_ispc_${OUTPUT_TARGET}.h" ISPC_TARGET_HEADER ${ISPC_SRC_FILE})
                    set(ISPC_TARGET_HEADER ${ISPC_DST_DIR}${ISPC_TARGET_HEADER})
                    list(APPEND ISPC_TARGET_HEADERS ${ISPC_TARGET_HEADER})
                endif()
            endforeach()
        endif()

        if(UNIX)
            set(ISPC_PIC "--pic")
        endif()

        # Passing space separate string yields escaped spaces.
        # So convert to a list and then use generator expression, i.e. "$<JOIN:${FLAGS},;>"
        separate_arguments(FLAGS NATIVE_COMMAND ${BENCHMARKS_ISPC_FLAGS})

        add_custom_command(
            OUTPUT ${ISPC_TARGET_OBJS} ${ISPC_TARGET_HEADERS}
            COMMENT "Compiling ${ISPC_SRC_FILE} for ${BENCHMARKS_ISPC_TARGETS} target(s)"
            COMMAND           ${ISPC_EXECUTABLE} ${SRC_LOCATION} -o ${ISPC_OBJ} -h ${ISPC_HEADER} --arch=${ISPC_ARCH} --target=${BENCHMARKS_ISPC_TARGETS} ${ISPC_PIC} "$<JOIN:${FLAGS},;>"
            DEPENDS ${ISPC_EXECUTABLE}
            DEPENDS ${ISPC_SRC_FILE}
            COMMAND_EXPAND_LISTS
        )
        if(MSVC)
            # Add .ispc file to VS solution.
            target_sources(${ADD_ISPC_TARGET} PUBLIC ${SRC_LOCATION})
            # Group .ispc files inside Visual Studio
            source_group("ISPC" FILES ${SRC_LOCATION})
            # Group benchmarks in "Benchmarks" folder
            set_target_properties(${ADD_ISPC_TARGET} PROPERTIES FOLDER "Benchmarks")
        endif()

        set_source_files_properties(${ISPC_TARGET_OBJS} PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
        set_source_files_properties(${ISPC_TARGET_HEADERS} PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
        list(APPEND ISPC_HEADERS_LIST ${SRC_LOCATION} ${ISPC_TARGET_HEADERS})
        list(APPEND ISPC_OBJS_LIST ${ISPC_TARGET_OBJS})
    endforeach()

    set_property (SOURCE ${ADD_ISPC_CPP_MAIN_FILE} PROPERTY OBJECT_DEPENDS ${ISPC_HEADERS_LIST})
    target_include_directories(${ADD_ISPC_TARGET} PRIVATE ${ISPC_DST_DIR})
    target_compile_definitions(${ADD_ISPC_TARGET} PRIVATE ISPC_ENABLED)
    target_link_libraries(${ADD_ISPC_TARGET} PRIVATE ${ISPC_OBJS_LIST})
endfunction()

# A macro to add a benchmark
macro(compile_benchmark_test name)
    add_executable(${name} "")

    # aligned_alloc() requires C++17
    set_target_properties(${name} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES)

    add_ispc_to_target(
        TARGET ${name}
        CPP_MAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp
        SOURCES ${name}.ispc)

    target_sources(
        ${name}
        PRIVATE ${name}.cpp)

    target_compile_definitions(
        ${name}
        PRIVATE BENCHMARKS_ISPC_TARGETS=\"${BENCHMARKS_ISPC_TARGETS}\"
                BENCHMARKS_ISPC_FLAGS=\"${BENCHMARKS_ISPC_FLAGS}\")

    # Turn on AVX2 support in the C++ compiler to be able to use AVX2 intrinsics.
    if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
        add_compile_options(-mavx2)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        add_compile_options(/QxAVX2)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        add_compile_options(/arch:AVX2)
    endif()

    # To enable google benchmarks:
    target_link_libraries(${name} PRIVATE benchmark)

    get_filename_component(INSTALL_SUBFOLDER "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

    install(
        TARGETS ${name}
        RUNTIME DESTINATION "benchmarks/${INSTALL_SUBFOLDER}")

    add_test(NAME ${name}_test COMMAND ${name} --benchmark_min_time=0.01)
    add_dependencies(${BENCHMARKS_PROJECT_NAME} ${name})
endmacro(compile_benchmark_test)
