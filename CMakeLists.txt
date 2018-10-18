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
# ispc CMakeLists.txt
#
cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME ispc)
project(${PROJECT_NAME} CXX)

option(ARM_ENABLED "Enable ARM support OFF")
option(NVPTX_ENABLED "Enable NVPTX support OFF")

set(OUTPUT_DEBUG Debug/bin)
set(OUTPUT_RELEASE Release/bin)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Build type not specified: Use Release by default.")
endif(NOT CMAKE_BUILD_TYPE)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FixWindowsPath.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/LLVMConfig.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Git.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateBuiltins.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Stdlib.cmake)

find_package(PythonInterp 2.7 REQUIRED)
    if (NOT PYTHONINTERP_FOUND)
        message(FATAL_ERROR "Python interpreter is not found")
    endif()
    if (WIN32)
        # For correct VS project the path should be in quotes
        set(PYTHON_EXECUTABLE \"${PYTHON_EXECUTABLE}\")
    endif()

find_package(BISON 2.4 REQUIRED)
    if (BISON_FOUND)
        set(BISON_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/parse.yy)
        set(BISON_CPP_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/parse.cc)
        set(BISON_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/parse.hh
                         ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/parse.output)
        if (WIN32)
            win_path_to_cygwin(${BISON_INPUT} BISON_INPUT)
        endif()
        add_custom_command (
            OUTPUT ${BISON_CPP_OUTPUT} ${BISON_OUTPUT}
            COMMAND ${BISON_EXECUTABLE} -d -t -v
                --output=${BISON_CPP_OUTPUT}
                ${BISON_INPUT}
            COMMENT "Generating parse.cc"
    )
    endif()

find_package(FLEX 2.5 REQUIRED)
    if (FLEX_FOUND)
        set(FLEX_INPUT  ${CMAKE_CURRENT_SOURCE_DIR}/lex.ll)
        set(FLEX_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lex.cpp)
        if (WIN32)
            win_path_to_cygwin(${FLEX_INPUT} FLEX_INPUT)
        endif()
        add_custom_command(
            OUTPUT ${FLEX_OUTPUT}
            COMMAND ${FLEX_EXECUTABLE}
                --outfile=${FLEX_OUTPUT}
                ${FLEX_INPUT}
            COMMENT "Generating lex.cpp"
        )
    endif()

set (ISPC_MASKS 1 8 16 32 64)
set (ISPC_TARGETS avx2-i64x4 avx11-i64x4 avx1-i64x4 avx1 avx1-x2 avx11 avx11-x2 avx2 avx2-x2
    sse2 sse2-x2 sse4-8 sse4-16 sse4 sse4-x2
    generic-4 generic-8 generic-16 generic-32 generic-64 generic-1 knl skx)

set(CLANG_LIBRARY_LIST clangFrontend clangDriver clangSerialization clangParse clangSema clangAnalysis clangAST clangBasic clangEdit clangLex)
set(LLVM_COMPONENTS engine ipo bitreader bitwriter instrumentation linker)
# Component "option" was introduced in 3.3 and starting with 3.4 it is required for the link step.
if (${LLVM_VERSION_NUMBER} VERSION_GREATER "3.3.0")
    set(LLVM_COMPONENTS ${LLVM_COMPONENTS} option)
endif()

if (ARM_ENABLED)
    list(APPEND LLVM_COMPONENTS arm)
    list(APPEND ISPC_TARGETS neon-32 neon-16 neon-8)
endif()
if (NVPTX_ENABLED)
    list(APPEND LLVM_COMPONENTS nvptx)
    list(APPEND ISPC_TARGETS nvptx)
endif()
get_llvm_libfiles(LLVM_LIBRARY_LIST ${LLVM_COMPONENTS})
get_llvm_cppflags(LLVM_CPP_FLAGS)

generate_target_builtins(BUILTIN_FILES ${ISPC_TARGETS})
generate_common_builtins(BUILTIN_CPP_FILES)
generate_stdlib(STDLIB_FILES ${ISPC_MASKS})

add_executable(${PROJECT_NAME} ${BUILTIN_FILES} ${BUILTIN_CPP_FILES}
               ${STDLIB_FILES} ${BISON_CPP_OUTPUT} ${FLEX_OUTPUT})

target_sources(${PROJECT_NAME}
    PRIVATE
        "src/ast.cpp"
        "src/ast.h"
        "src/builtins.cpp"
        "src/builtins.h"
        "src/cbackend.cpp"
        "src/ctx.cpp"
        "src/ctx.h"
        "src/decl.cpp"
        "src/decl.h"
        "src/expr.cpp"
        "src/expr.h"
        "src/func.cpp"
        "src/func.h"
        "src/ispc.cpp"
        "src/ispc.h"
        "src/ispc_version.h"
        "src/llvmutil.cpp"
        "src/llvmutil.h"
        "src/main.cpp"
        "src/module.cpp"
        "src/module.h"
        "src/opt.cpp"
        "src/opt.h"
        "src/stmt.cpp"
        "src/stmt.h"
        "src/sym.cpp"
        "src/sym.h"
        "src/type.cpp"
        "src/type.h"
        "src/util.cpp"
        "src/util.h"
)

# Build definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE ${LLVM_VERSION})
if (UNIX)
    string(TIMESTAMP BUILD_DATE "%Y%m%d")
    target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_DATE=\"${BUILD_DATE}\"
                            BUILD_VERSION=\"${GIT_COMMIT_HASH}\")
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)
    if (NOT CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
        target_compile_definitions(${PROJECT_NAME} PRIVATE NDEBUG)
    endif()
endif()

if (ARM_ENABLED)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ISPC_ARM_ENABLED)
endif()
if (NVPTX_ENABLED)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ISPC_NVPTX_ENABLED)
endif()

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
                           ${LLVM_INCLUDE_DIRS}
                           ${CMAKE_CURRENT_SOURCE_DIR}/src
                           ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
# Compile options
if (UNIX)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wno-sign-compare -Wno-unused-function -Werror ${LLVM_CPP_FLAGS})
    set_source_files_properties("src/cbackend.cpp" PROPERTIES COMPILE_FLAGS "-fno-exceptions")
else()
    target_compile_options(${PROJECT_NAME} PRIVATE /W3 /wd4146 /wd4800 /wd4996 /wd4355 /wd4624 /wd4244 /wd4141 /wd4291 /wd4018)
    set_source_files_properties(${FLEX_OUTPUT} PROPERTIES COMPILE_FLAGS "/wd4005 /wd4003")
    set_source_files_properties(${BISON_OUTPUT} PROPERTIES COMPILE_FLAGS "/wd4005 /wd4065")
endif()

if (${LLVM_VERSION_NUMBER} VERSION_GREATER "3.4.0")
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
    if (UNIX)
        set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)
        target_compile_options(${PROJECT_NAME} PRIVATE -Wno-c99-extensions -Wno-deprecated-register -fno-rtti)
    endif()
endif()

# Link options
if (WIN32)
    if (NOT CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "/OPT:REF /OPT:ICF")
    endif()
endif()

# Link against Clang libraries
foreach(clangLib ${CLANG_LIBRARY_LIST})
    find_library(${clangLib}Path NAMES ${clangLib} HINTS ${LLVM_LIBRARY_DIRS})
    list(APPEND CLANG_LIBRARY_FULL_PATH_LIST ${${clangLib}Path})
endforeach()
target_link_libraries(${PROJECT_NAME} ${CLANG_LIBRARY_FULL_PATH_LIST})

# Link against LLVM libraries
target_link_libraries(${PROJECT_NAME} ${LLVM_LIBRARY_LIST})
if (WIN32)
    target_link_libraries(${PROJECT_NAME} version.lib shlwapi.lib odbc32.lib odbccp32.lib)
else()
    target_link_libraries(${PROJECT_NAME} pthread dl)
    if (${LLVM_VERSION_NUMBER} VERSION_GREATER "3.4.0")
        target_link_libraries(${PROJECT_NAME} z)
    endif()
    if (APPLE)
        target_link_libraries(${PROJECT_NAME} curses)
    else()
        target_link_libraries(${PROJECT_NAME} tinfo)
        if (${LLVM_VERSION_NUMBER} VERSION_GREATER "3.3.0")
            target_link_libraries(${PROJECT_NAME} curses)
        endif()
    endif()
endif()

install (TARGETS ${PROJECT_NAME} DESTINATION bin)