#
#  Copyright (c) 2020-2026, Intel Corporation
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
    elseif("${ARCH}" STREQUAL "aarch64")
        set(ISPC_ARCH "aarch64")
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


include(CheckCXXCompilerFlag)

#######################
#  ispc_isa_suffix
#######################
#
#  Returns the suffix that ISPC appends to object and header file names for
#  the given target in a multi-target build, e.g. avx2-i32x8 -> avx2.
#
function(ispc_isa_suffix ISPC_TARGET OUT)
    string(REGEX REPLACE "-.*" "" ISA "${ISPC_TARGET}")
    if (ISA MATCHES "^sse4")
        set(ISA "sse4")
    elseif (ISA MATCHES "^avx1?$")
        set(ISA "avx")
    endif()
    string(REPLACE "." "_" ISA "${ISA}")
    set(${OUT} ${ISA} PARENT_SCOPE)
endfunction()

#######################
#  benchmark_cxx_isa_flags
#######################
#
#  Returns C++ compiler flags that enable the same ISA as the given ISPC
#  target, so the C++ reference implementations are built for the ISA under
#  test. GNU-like compilers get the same CPU that ISPC uses for the target.
#
function(benchmark_cxx_isa_flags ISPC_TARGET OUT)
    set(FLAGS "")
    if (ISPC_ARCH MATCHES "x86" AND (MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
        if (ISPC_TARGET MATCHES "^avx1?(-|$)")
            set(FLAGS "/arch:AVX")
        elseif (ISPC_TARGET MATCHES "^avx2")
            set(FLAGS "/arch:AVX2")
        elseif (ISPC_TARGET MATCHES "^avx512|^avx10")
            set(FLAGS "/arch:AVX512")
        endif()
    elseif (ISPC_ARCH MATCHES "x86")
        # Ordered pairs of target name pattern (covering aliases) and CPU.
        set(TARGET_CPUS
            "^host$" "native"
            "^sse2" "x86-64"
            "^sse4\\.1" "penryn"
            "^sse4" "nehalem"
            "^avx1?(-|$)" "sandybridge"
            "^avx2vnni" "alderlake"
            "^avx2" "haswell"
            "^avx512skx" "skylake-avx512"
            "^avx512icl" "icelake-client"
            "^avx512spr" "sapphirerapids"
            "^avx512gnr" "graniterapids"
            "^avx10\\.2dmr" "diamondrapids"
            "^avx10\\.2nvl" "novalake")
        set(CPU "")
        while (TARGET_CPUS AND NOT CPU)
            list(POP_FRONT TARGET_CPUS PATTERN TARGET_CPU)
            if (ISPC_TARGET MATCHES "${PATTERN}")
                set(CPU ${TARGET_CPU})
            endif()
        endwhile()
        if (NOT CPU)
            # Generic targets don't need a specific ISA.
            if (NOT ISPC_TARGET MATCHES "^generic-")
                message(FATAL_ERROR "Unknown x86 ISPC target for benchmarks: ${ISPC_TARGET}")
            endif()
            set(${OUT} "" PARENT_SCOPE)
            return()
        endif()
        string(MAKE_C_IDENTIFIER "CXX_HAS_MARCH_${CPU}" HAS_FLAG)
        check_cxx_compiler_flag("-march=${CPU}" ${HAS_FLAG})
        if (${HAS_FLAG})
            set(FLAGS "-march=${CPU}")
        else()
            message(WARNING "C++ compiler doesn't support -march=${CPU}, C++ parts of ${ISPC_TARGET} benchmarks "
                            "are built for the default ISA.")
        endif()
    endif()
    set(${OUT} ${FLAGS} PARENT_SCOPE)
endfunction()

# Every benchmark is built once per target in BENCHMARKS_ISPC_TARGETS.
string(REPLACE "," ";" BENCHMARKS_ISPC_TARGET_LIST "${BENCHMARKS_ISPC_TARGETS}")
list(TRANSFORM BENCHMARKS_ISPC_TARGET_LIST STRIP)
list(REMOVE_ITEM BENCHMARKS_ISPC_TARGET_LIST "")
list(REMOVE_DUPLICATES BENCHMARKS_ISPC_TARGET_LIST)
foreach (ISPC_TARGET ${BENCHMARKS_ISPC_TARGET_LIST})
    benchmark_cxx_isa_flags(${ISPC_TARGET} BENCHMARKS_CXX_ISA_FLAGS_${ISPC_TARGET})
endforeach()

#######################
#  add_benchmark_executable
#######################
#
#  Builds one executable from <NAME>.cpp and <NAME>.ispc.
#
#  NAME : Name of the benchmark sources.
#  EXECUTABLE : Name of the executable.
#  ISPC_TARGETS : Comma separated list of ISPC targets. More than one target
#                 produces an auto-dispatch binary.
#  CXX_FLAGS : Extra C++ compiler flags.
#
function(add_benchmark_executable)
    set(options)
    set(one_value_args NAME EXECUTABLE ISPC_TARGETS)
    set(multi_value_args CXX_FLAGS)
    cmake_parse_arguments("BENCH" "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    set(EXE ${BENCH_EXECUTABLE})
    # ISPC outputs don't depend on the build configuration.
    set(ISPC_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ispc/${EXE}/")
    file(MAKE_DIRECTORY ${ISPC_DST_DIR})

    set(ISPC_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${BENCH_NAME}.ispc")
    set(ISPC_OBJ "${ISPC_DST_DIR}${BENCH_NAME}${CMAKE_CXX_OUTPUT_EXTENSION}")
    set(ISPC_HEADER "${ISPC_DST_DIR}${BENCH_NAME}_ispc.h")
    set(ISPC_OBJS ${ISPC_OBJ})
    set(ISPC_HEADERS ${ISPC_HEADER})

    # For multiple targets ISPC also emits an object and a header per ISA.
    string(REPLACE "," ";" TARGET_LIST "${BENCH_ISPC_TARGETS}")
    list(LENGTH TARGET_LIST TARGET_COUNT)
    if (TARGET_COUNT GREATER 1)
        foreach (ISPC_TARGET ${TARGET_LIST})
            ispc_isa_suffix(${ISPC_TARGET} ISA)
            list(APPEND ISPC_OBJS "${ISPC_DST_DIR}${BENCH_NAME}_${ISA}${CMAKE_CXX_OUTPUT_EXTENSION}")
            list(APPEND ISPC_HEADERS "${ISPC_DST_DIR}${BENCH_NAME}_ispc_${ISA}.h")
        endforeach()
    endif()

    if (UNIX)
        set(ISPC_PIC "--pic")
    endif()

    # Passing space separate string yields escaped spaces.
    # So convert to a list and then use generator expression, i.e. "$<JOIN:${FLAGS},;>"
    separate_arguments(FLAGS NATIVE_COMMAND ${BENCHMARKS_ISPC_FLAGS})

    # Pass HAS_FP16 definition to ISPC compiler
    if (HAS_FP16_SUPPORT)
        set(ISPC_FP16_FLAG "-DHAS_FP16")
    else()
        set(ISPC_FP16_FLAG "")
    endif()
    set(INTRINSICS_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/../../stdlib/include)

    add_custom_command(
        OUTPUT ${ISPC_OBJS} ${ISPC_HEADERS}
        COMMENT "Compiling ${BENCH_NAME}.ispc for ${BENCH_ISPC_TARGETS}"
        COMMAND ${ISPC_EXECUTABLE} ${ISPC_SRC} -o ${ISPC_OBJ} -h ${ISPC_HEADER} --arch=${ISPC_ARCH}
                --target=${BENCH_ISPC_TARGETS} ${ISPC_PIC} ${ISPC_FP16_FLAG} -I${INTRINSICS_HEADERS}
                "$<JOIN:${FLAGS},;>"
        DEPENDS ${ISPC_EXECUTABLE} ${ISPC_DEPS} ${ISPC_SRC}
        COMMAND_EXPAND_LISTS
    )
    set_source_files_properties(${ISPC_OBJS} PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
    set_source_files_properties(${ISPC_HEADERS} PROPERTIES GENERATED TRUE)

    # Generated headers are listed as sources, so they are produced before the C++ file is compiled.
    add_executable(${EXE} ${BENCH_NAME}.cpp ${ISPC_OBJS} ${ISPC_HEADERS})

    # Benchmarks are written in C++17
    set_target_properties(${EXE} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES)

    if (MSVC)
        # Add .ispc file to VS solution.
        target_sources(${EXE} PRIVATE ${ISPC_SRC})
        # Group .ispc files inside Visual Studio
        source_group("ISPC" FILES ${ISPC_SRC})
        # Group benchmarks in "Benchmarks" folder
        set_target_properties(${EXE} PROPERTIES FOLDER "Benchmarks")
    endif()

    target_include_directories(${EXE} PRIVATE ${ISPC_DST_DIR})
    if (X86_HOST)
        target_compile_definitions(${EXE} PRIVATE IS_X86_ARCH)
    endif()
    target_compile_definitions(${EXE} PRIVATE ISPC_ENABLED
        BENCHMARKS_ISPC_TARGETS=\"${BENCH_ISPC_TARGETS}\"
        BENCHMARKS_ISPC_FLAGS=\"${BENCHMARKS_ISPC_FLAGS}\")

    # Build the C++ part for the ISA under test. Auto-dispatch binaries use the
    # compiler's default ISA, so they run on any of their targets.
    set(ISA_FLAGS "")
    if (TARGET_COUNT EQUAL 1)
        set(ISA_FLAGS ${BENCHMARKS_CXX_ISA_FLAGS_${BENCH_ISPC_TARGETS}})
    endif()
    target_compile_options(${EXE} PRIVATE ${ISA_FLAGS} ${BENCH_CXX_FLAGS})

    # To enable google benchmarks:
    target_link_libraries(${EXE} PRIVATE benchmark)

    get_filename_component(INSTALL_SUBFOLDER "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
    install(
        TARGETS ${EXE}
        RUNTIME DESTINATION "benchmarks/${INSTALL_SUBFOLDER}")

    add_test(NAME ${EXE}_test COMMAND ${EXE} --benchmark_min_time=0.01)
    add_dependencies(${BENCHMARKS_PROJECT_NAME} ${EXE})
endfunction()

#######################
#  compile_benchmark_test
#######################
#
#  Adds a benchmark built from <NAME>.cpp and <NAME>.ispc. One executable
#  <NAME>.<target> is built for every target in BENCHMARKS_ISPC_TARGETS.
#
#  DISPATCH : Also build <NAME>.dispatch, an auto-dispatch binary with the
#             first target of every ISA in BENCHMARKS_ISPC_TARGETS. It is
#             skipped if there are less than two ISAs.
#  CXX_FLAGS : Extra C++ compiler flags.
#
function(compile_benchmark_test NAME)
    cmake_parse_arguments("BENCH" "DISPATCH" "" "CXX_FLAGS" ${ARGN})
    if (BENCH_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "compile_benchmark_test(${NAME}): unknown arguments ${BENCH_UNPARSED_ARGUMENTS}")
    endif()

    foreach (ISPC_TARGET ${BENCHMARKS_ISPC_TARGET_LIST})
        add_benchmark_executable(
            NAME ${NAME}
            EXECUTABLE ${NAME}.${ISPC_TARGET}
            ISPC_TARGETS ${ISPC_TARGET}
            CXX_FLAGS ${BENCH_CXX_FLAGS})
    endforeach()

    if (BENCH_DISPATCH)
        # ISPC accepts only one target per ISA in a multi-target build. The ISA
        # of "host" (and so its output file names) is unknown here, skip it.
        # Generic targets can't be dispatched to, skip them as well.
        set(DISPATCH_TARGETS "")
        set(DISPATCH_ISAS "")
        foreach (ISPC_TARGET ${BENCHMARKS_ISPC_TARGET_LIST})
            ispc_isa_suffix(${ISPC_TARGET} ISA)
            if (NOT ISA STREQUAL "host" AND NOT ISPC_TARGET MATCHES "^generic-" AND NOT ISA IN_LIST DISPATCH_ISAS)
                list(APPEND DISPATCH_ISAS ${ISA})
                list(APPEND DISPATCH_TARGETS ${ISPC_TARGET})
            endif()
        endforeach()
        list(LENGTH DISPATCH_TARGETS DISPATCH_COUNT)
        if (DISPATCH_COUNT GREATER 1)
            string(REPLACE ";" "," DISPATCH_TARGETS "${DISPATCH_TARGETS}")
            add_benchmark_executable(
                NAME ${NAME}
                EXECUTABLE ${NAME}.dispatch
                ISPC_TARGETS ${DISPATCH_TARGETS}
                CXX_FLAGS ${BENCH_CXX_FLAGS})
        endif()
    endif()
endfunction()
