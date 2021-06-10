#
#  Copyright (c) 2018-2021, Intel Corporation
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
cmake_minimum_required(VERSION 3.13)

if (UNIX)
    set(CMAKE_C_COMPILER "clang")
    set(CMAKE_CXX_COMPILER "clang++")
endif()

set(PROJECT_NAME ispc)
set(ISPC_BUILD TRUE)
project(${PROJECT_NAME})

set(X86_HOST FALSE)
if (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "AMD64|86")
    set(X86_HOST TRUE)
endif()

option(X86_ENABLED "Enable x86 support" ${X86_HOST})
option(ARM_ENABLED "Enable ARM support" ON)
option(WASM_ENABLED "Enable experimental Web Assembly support" OFF)
option(GENX_ENABLED "Enable GenX support" OFF)
option(ISPC_INCLUDE_EXAMPLES "Generate build targets for the ISPC examples" ON)
option(ISPC_INCLUDE_DPCPP_EXAMPLES "Generate build targets for the ISPC/DPCPP interoperability examples" OFF)
option(ISPC_INCLUDE_TESTS "Generate build targets for the ISPC tests." ON)
option(ISPC_INCLUDE_BENCHMARKS "Generate build targets for the ISPC tests." OFF)
option(ISPC_INCLUDE_UTILS "Generate build targets for the utils." ON)
option(ISPC_PREPARE_PACKAGE "Generate build targets for ispc package" OFF)
option(ISPC_NO_DUMPS "Turn off functionality, which requires LLVM dump() functions" OFF)

option(ISPC_CROSS "Build ISPC with cross compilation support" OFF)
# Default settings for cross compilation
if (ISPC_CROSS)
    option(ISPC_WINDOWS_TARGET "Build ISPC with windows target support" ON)
    option(ISPC_LINUX_TARGET "Build ISPC with linux target support" ON)
    option(ISPC_FREEBSD_TARGET "Build ISPC with freebsd target support" ON)
    # It's possbile to build macOS target on Windows or Linux, but we don't do that by default.
    option(ISPC_MACOS_TARGET "Build ISPC with macos target support" OFF)
    option(ISPC_IOS_TARGET "Build ISPC with ios target support" ON)
    option(ISPC_ANDROID_TARGET "Build ISPC with android target support" ON)
    option(ISPC_PS4_TARGET "Build ISPC with ps4 target support" ON)
    # We silently disable some of cross targets. We should probably warn user or issue an error,
    # but it seems good enough.
    if (WIN32)
        set(ISPC_IOS_TARGET OFF)
        if ((ISPC_LINUX_TARGET OR ISPC_ANDROID_TARGET OR ISPC_PS4_TARGET) AND NOT ISPC_GNUWIN32_PATH)
            message (FATAL_ERROR "Set ISPC_GNUWIN32_PATH variable for cross compilation to Linux/Android/PS4 e.g. C:/gnuwin32")
        endif()
        if (ISPC_MACOS_TARGET AND NOT ISPC_MACOS_SDK_PATH)
            message (FATAL_ERROR "Set ISPC_MACOS_SDK_PATH variable for cross compilation to MacOS e.g. C:/iusers/MacOSX10.14.sdk")
        endif()
    elseif (APPLE)
        set(ISPC_MACOS_TARGET ON)
        set(ISPC_WINDOWS_TARGET OFF)
        set(ISPC_PS4_TARGET OFF)
        set(ISPC_FREEBSD_TARGET OFF)
        if ((ISPC_LINUX_TARGET OR ISPC_ANDROID_TARGET) AND NOT ISPC_ANDROID_NDK_PATH)
            message (FATAL_ERROR "Set ISPC_ANDROID_NDK_PATH variable for cross compilation to Linux/Android e.g. /Users/Shared/android-ndk-r20")
        endif()
        if (ISPC_IOS_TARGET AND NOT ISPC_IOS_SDK_PATH)
            # Use standard iOS SDK location if this is not specified.
            set(command "xcrun" "--show-sdk-path" "--sdk" "iphoneos")
            execute_process(COMMAND ${command}
                OUTPUT_VARIABLE ISPC_IOS_SDK_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            message(STATUS "Using iOS SDK path ${ISPC_IOS_SDK_PATH}")
        endif()
    else()
        set(ISPC_WINDOWS_TARGET OFF)
        set(ISPC_PS4_TARGET OFF)
        set(ISPC_IOS_TARGET OFF)
        if (ISPC_MACOS_TARGET AND NOT ISPC_MACOS_SDK_PATH)
            message (FATAL_ERROR "Set ISPC_MACOS_SDK_PATH variable for cross compilation to MacOS e.g. /iusers/MacOSX10.14.sdk")
        endif()
    endif()
else()
    if (WIN32)
        set(ISPC_WINDOWS_TARGET ON)
    elseif (APPLE)
        set(ISPC_MACOS_TARGET ON)
    elseif (CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
        set(ISPC_FREEBSD_TARGET ON)
    else()
        set(ISPC_LINUX_TARGET ON)
    endif()
endif()

# Get ispc version
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/ispc_version.h" ispc_ver)
string(REGEX MATCH "ISPC_VERSION \"([0-9]*)\.([0-9]*)\.([0-9]*)([a-z]*)" _ ${ispc_ver})
set(ISPC_VERSION_MAJOR ${CMAKE_MATCH_1})
set(ISPC_VERSION_MINOR ${CMAKE_MATCH_2})
set(ISPC_VERSION_PATCH ${CMAKE_MATCH_3})
set(ISPC_VERSION_SUFFIX ${CMAKE_MATCH_4})
if (${ISPC_VERSION_SUFFIX} MATCHES ".*dev")
    set (REPO_TAG "main")
else()
    set (REPO_TAG "v${ISPC_VERSION_MAJOR}.${ISPC_VERSION_MINOR}.${ISPC_VERSION_PATCH}${ISPC_VERSION_SUFFIX}")
endif()

if (APPLE)
    # Use standard macOS SDK location if this is not specified.
    if (NOT ISPC_MACOS_SDK_PATH)
        set(command "xcrun" "--show-sdk-path")
        execute_process(COMMAND ${command}
            OUTPUT_VARIABLE ISPC_MACOS_SDK_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        message(STATUS "Using macOS SDK path ${ISPC_MACOS_SDK_PATH}")
    endif()
endif()

set(ISPC_MACOS_ARM_TARGET OFF)
if (ISPC_MACOS_TARGET AND ISPC_MACOS_SDK_PATH)
    if (IS_SYMLINK ${ISPC_MACOS_SDK_PATH})
        # Follow symlinks, relative and absolute.
        file(READ_SYMLINK "${ISPC_MACOS_SDK_PATH}" ISPC_MACOS_SDK_PATH_NEW)
        if (NOT IS_ABSOLUTE "${ISPC_MACOS_SDK_PATH_NEW}")
            get_filename_component(ISPC_MACOS_SDK_PATH_DIR "${ISPC_MACOS_SDK_PATH}" DIRECTORY)
            set(ISPC_MACOS_SDK_PATH_NEW "${ISPC_MACOS_SDK_PATH_DIR}/${ISPC_MACOS_SDK_PATH_NEW}")
        endif()
        set(ISPC_MACOS_SDK_PATH "${ISPC_MACOS_SDK_PATH_NEW}")
    endif()

    # Grepping path to figure out the version is not the most reliable way,
    # but it seems to be the most practical.
    string(REGEX MATCH "MacOSX([0-9]*.[0-9]*).sdk" _ ${ISPC_MACOS_SDK_PATH})
    set(SDK_VER "${CMAKE_MATCH_1}")
    message(STATUS "MacOS_SDK version: ${SDK_VER}")

    if ("${SDK_VER}" STREQUAL "")
        message(WARNING "MacOS SDK version was not detected, assuming 11.0, enabling ARM support")
        set(ISPC_MACOS_ARM_TARGET ON)
    elseif("${SDK_VER}" VERSION_GREATER_EQUAL "11.0")
        message(STATUS "MacOS_SDK supports ARM (SDK ver >= 11.0)")
        set(ISPC_MACOS_ARM_TARGET ON)
    else()
        message(STATUS "MacOS_SDK does NOT supports ARM (SDK ver < 11.0)")
    endif()
endif()

# If GenX target is enabled, generate build targets for genx examples and ISPCRT on Linux
if (GENX_ENABLED)
    option(ISPC_INCLUDE_RT "Generate build targets for ISPC runtime." ON)
    option(ISPC_INCLUDE_GENX_EXAMPLES "Generate build targets for the ISPC GenX examples" ON)
endif()

if (UNIX)
    option(ISPC_STATIC_STDCXX_LINK "Link statically with libstdc++ and libgcc" OFF)
    if (ISPC_PREPARE_PACKAGE AND (NOT APPLE))
        option(ISPC_STATIC_LINK "Link statically (all except glibc)" ON)
    else()
        option(ISPC_STATIC_LINK "Link statically (all except glibc)" OFF)
    endif()
    option(ISPC_USE_ASAN "Build ispc with address sanitizer instrumentation using clang compiler" OFF)
endif()

# Use solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(OUTPUT_DEBUG Debug/bin)
set(OUTPUT_RELEASE Release/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin )

if(CMAKE_BUILD_TYPE)
    # Validate build type
    set(CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo")

    string(FIND "${CONFIGURATION_TYPES}" "${CMAKE_BUILD_TYPE}" MATCHED_CONFIG)
    if (${MATCHED_CONFIG} EQUAL -1)
         message(FATAL_ERROR "CMAKE_BUILD_TYPE (${CMAKE_BUILD_TYPE}) allows only the following values: ${CONFIGURATION_TYPES}")
    endif()
else(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Build type not specified: Use Release by default.")
endif(CMAKE_BUILD_TYPE)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ispcrt/cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FixWindowsPath.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindLLVM.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Git.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateBuiltins.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Stdlib.cmake)

find_package(Python3 REQUIRED)
    if (NOT Python3_Interpreter_FOUND)
        message(FATAL_ERROR "Python interpreter is not found")
    endif()

find_package(BISON 3.0 REQUIRED)
    if (BISON_FOUND)
        set(BISON_INPUT src/parse.yy)
        set(BISON_CPP_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/parse.cc)
        set(BISON_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/parse.hh
                         ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/parse.output)
        add_custom_command (
            OUTPUT ${BISON_CPP_OUTPUT} ${BISON_OUTPUT}
            COMMAND ${BISON_EXECUTABLE} -d -t -v
                --output=${BISON_CPP_OUTPUT}
                ${BISON_INPUT}
            COMMENT "Generating parse.cc"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    endif()

find_package(FLEX 2.6 REQUIRED)
    if (FLEX_FOUND)
        set(FLEX_INPUT  src/lex.ll)
        set(FLEX_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lex.cpp)
        add_custom_command(
            OUTPUT ${FLEX_OUTPUT}
            COMMAND ${FLEX_EXECUTABLE}
                --outfile=${FLEX_OUTPUT}
                ${FLEX_INPUT}
            COMMENT "Generating lex.cpp"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()

set (ISPC_MASKS 1 8 16 32 64)

if (NOT X86_ENABLED AND NOT ARM_ENABLED AND NOT WASM_ENABLED AND NOT GENX_ENABLED)
    message( FATAL_ERROR "Either X86, ARM, WASM or GENX targets need to be enabled.")
endif ()

if (X86_ENABLED)
    list(APPEND ISPC_TARGETS
        sse2-i32x4 sse2-i32x8
        sse4-i32x4 sse4-i32x8 sse4-i8x16 sse4-i16x8
        # Note here that avx1-i32x4 is aliased to sse4-i32x4 plus avx attribute.
        avx1-i32x8 avx1-i32x16 avx1-i64x4
        avx2-i8x32 avx2-i16x16
        avx2-i32x4 avx2-i32x8 avx2-i32x16 avx2-i64x4
        avx512knl-i32x16
        avx512skx-i32x16 avx512skx-i32x8
        avx512skx-i8x64 avx512skx-i16x32)
endif()
if (ARM_ENABLED)
    list(APPEND ISPC_TARGETS neon-i8x16 neon-i16x8 neon-i32x4 neon-i32x8)
endif()
if (WASM_ENABLED)
    find_program(EMCC_EXECUTABLE emcc)
    if (NOT EMCC_EXECUTABLE)
        message(FATAL_ERROR "emcc not found!")
    endif()
    message(STATUS "EMCC_EXECUTABLE: ${EMCC_EXECUTABLE}")
    list(APPEND ISPC_TARGETS wasm-i32x4)
endif()

set(CLANG_LIBRARY_LIST clangFrontend clangDriver clangSerialization clangParse clangSema clangAnalysis clangAST clangBasic clangEdit clangLex)
set(LLVM_COMPONENTS engine ipo bitreader bitwriter instrumentation linker option frontendopenmp)

if (X86_ENABLED)
    list(APPEND LLVM_COMPONENTS x86)
endif()
if (ARM_ENABLED)
    list(APPEND LLVM_COMPONENTS arm aarch64)
endif()
if (WASM_ENABLED)
    list(APPEND LLVM_COMPONENTS webassembly)
endif()
if (GENX_ENABLED)
    list(APPEND ISPC_TARGETS genx-x16 genx-x8)
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(TARGET_MODIFIER "32")
        set(PB_PATH_MODIFIER "x86")
    else()
        set(TARGET_MODIFIER "64")
        set(PB_PATH_MODIFIER "x64")
    endif()

    list(APPEND GEN_LIBRARY_LIST LLVMGenXIntrinsics LLVMSPIRVLib)
endif()

get_llvm_libfiles(LLVM_LIBRARY_LIST ${LLVM_COMPONENTS})
get_llvm_cppflags(LLVM_CPP_FLAGS)

generate_target_builtins(BUILTIN_FILES ${ISPC_TARGETS})
generate_common_builtins(BUILTIN_CPP_FILES)
generate_stdlib(STDLIB_FILES ${ISPC_MASKS})

add_executable(${PROJECT_NAME} ${BUILTIN_FILES} ${BUILTIN_CPP_FILES}
               ${STDLIB_FILES} ${BISON_CPP_OUTPUT} ${FLEX_OUTPUT}
               ${CMAKE_CURRENT_SOURCE_DIR}/stdlib.ispc)

target_sources(${PROJECT_NAME}
    PRIVATE
        "src/ast.cpp"
        "src/ast.h"
        "src/bitcode_lib.cpp"
        "src/bitcode_lib.h"
        "src/builtins.cpp"
        "src/builtins.h"
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
        "src/target_enums.cpp"
        "src/target_enums.h"
        "src/target_registry.cpp"
        "src/target_registry.h"
        "src/type.cpp"
        "src/type.h"
        "src/util.cpp"
        "src/util.h"
        "src/parse.yy"
        "src/lex.ll"
)

if (GENX_ENABLED)
    target_sources(${PROJECT_NAME} PRIVATE "src/gen/GlobalsLocalization.cpp")
endif()

# To show stdlib.ispc in VS solution:
if (WIN32)
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/stdlib.ispc" PROPERTIES HEADER_FILE_ONLY TRUE)
    source_group("ISPC" FILES "${CMAKE_CURRENT_SOURCE_DIR}/stdlib.ispc")
endif()

# Build definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE ${LLVM_VERSION})
if (UNIX)
    string(TIMESTAMP BUILD_DATE "%Y%m%d")
    target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_DATE=\"${BUILD_DATE}\"
                            BUILD_VERSION=\"${GIT_COMMIT_HASH}\")
    # Compile-time protection against static sized buffer overflows.
    target_compile_definitions(${PROJECT_NAME} PRIVATE "_FORTIFY_SOURCE=2")
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)
    if (NOT CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
        target_compile_definitions(${PROJECT_NAME} PRIVATE NDEBUG)
    endif()
endif()

if (X86_ENABLED)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ISPC_X86_ENABLED)
endif()

if (ARM_ENABLED)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ISPC_ARM_ENABLED)
endif()
if (GENX_ENABLED)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ISPC_GENX_ENABLED)
endif()

if (WASM_ENABLED)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ISPC_WASM_ENABLED)
endif()

if (ISPC_NO_DUMPS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ISPC_NO_DUMPS)
endif()

# Compile definitions for cross compilation
if (NOT ISPC_WINDOWS_TARGET)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ISPC_WINDOWS_TARGET_OFF)
endif()
if (NOT ISPC_LINUX_TARGET)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ISPC_LINUX_TARGET_OFF)
endif()
if (NOT ISPC_FREEBSD_TARGET)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ISPC_FREEBSD_TARGET_OFF)
endif()
if (NOT ISPC_MACOS_TARGET)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ISPC_MACOS_TARGET_OFF)
endif()
if (NOT ISPC_IOS_TARGET)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ISPC_IOS_TARGET_OFF)
endif()
if (NOT ISPC_ANDROID_TARGET)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ISPC_ANDROID_TARGET_OFF)
endif()
if (NOT ISPC_PS4_TARGET)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ISPC_PS4_TARGET_OFF)
endif()

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
                           ${LLVM_INCLUDE_DIRS}
                           ${GENX_DEPS_DIR}/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/src
                           ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})

# Compile options
if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W3 /wd4018 /wd4065 /wd4141 /wd4146 /wd4244 /wd4267 /wd4291 /wd4355 /wd4624 /wd4800 /wd4996)
    # Security options
    target_compile_options(${PROJECT_NAME} PRIVATE /GS)
    set_source_files_properties(${FLEX_OUTPUT} PROPERTIES COMPILE_FLAGS "/wd4005 /wd4003")
    set_source_files_properties(${BISON_OUTPUT} PROPERTIES COMPILE_FLAGS "/wd4005 /wd4065")
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wno-sign-compare -Wno-unused-function ${LLVM_CPP_FLAGS})
    # Security options
    target_compile_options(${PROJECT_NAME} PRIVATE -fstack-protector-strong
                           -fdata-sections -ffunction-sections -fno-delete-null-pointer-checks
                           -Wformat -Wformat-security -fpie -fwrapv)
endif()

# Set C++ standard to C++14.
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES)

if (NOT MSVC)
    set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-c99-extensions -Wno-deprecated-register -fno-rtti)
    if (ISPC_USE_ASAN)
        target_compile_options(${PROJECT_NAME} PRIVATE -fsanitize=address)
    endif()
endif()

# Link options
if (WIN32)
    target_link_options(${PROJECT_NAME} PRIVATE /DYNAMICBASE)
    if (MSVC AND NOT CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
        target_link_options(${PROJECT_NAME} PUBLIC /OPT:REF /OPT:ICF)
    endif()
elseif (APPLE)
else()
    # Link options for security hardening.
    target_link_options(${PROJECT_NAME}
        PUBLIC "SHELL: -z noexecstack"
               "SHELL: -z relro"
               "SHELL: -z now"
               "SHELL: -Wl,--gc-sections")
endif()

if (ISPC_STATIC_STDCXX_LINK OR ISPC_STATIC_LINK)
    target_link_options(${PROJECT_NAME} PUBLIC -static-libgcc -static-libstdc++)
endif()

if (ISPC_USE_ASAN)
    target_link_options(${PROJECT_NAME} PUBLIC -fsanitize=address)
endif()

if (NOT WIN32 AND NOT APPLE)
    # To resolve circular dependencies between libraries use --start-group/--end-group
    target_link_libraries(${PROJECT_NAME} "-Wl,--start-group")
endif()

# Link against Clang libraries
foreach(clangLib ${CLANG_LIBRARY_LIST})
    find_library(${clangLib}Path NAMES ${clangLib} HINTS ${LLVM_LIBRARY_DIRS})
    list(APPEND CLANG_LIBRARY_FULL_PATH_LIST ${${clangLib}Path})
endforeach()
target_link_libraries(${PROJECT_NAME} ${CLANG_LIBRARY_FULL_PATH_LIST})

# Link against LLVM libraries
target_link_libraries(${PROJECT_NAME} ${LLVM_LIBRARY_LIST} ${CMAKE_DL_LIBS})

if (GENX_ENABLED)
    # Link against GEN libraries
    foreach(genLib ${GEN_LIBRARY_LIST})
        find_library(${genLib}Path NAMES ${genLib} HINTS ${GENX_DEPS_DIR}/lib)
        list(APPEND GEN_LIBRARY_FULL_PATH_LIST ${${genLib}Path})
    endforeach()
    target_link_libraries(${PROJECT_NAME} ${GEN_LIBRARY_FULL_PATH_LIST})
endif()

if (NOT WIN32 AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} "-Wl,--end-group")
endif()

# System libraries, our own and transitive dependencies from LLVM libs.
if (WIN32)
    target_link_libraries(${PROJECT_NAME} version.lib shlwapi.lib odbc32.lib odbccp32.lib)
else()
    if (APPLE)
        target_link_libraries(${PROJECT_NAME} pthread z curses)
    else()
        if (ISPC_STATIC_LINK)
            target_link_libraries(${PROJECT_NAME} pthread z.a tinfo.a curses.a)
        else()
            find_package(Curses REQUIRED)
            find_package(ZLIB REQUIRED)
            find_library(NCURSES_TINFO_LIBRARY tinfo)
            target_link_libraries(${PROJECT_NAME} pthread ${ZLIB_LIBRARIES} ${NCURSES_TINFO_LIBRARY} ${CURSES_LIBRARIES})
        endif()
    endif()
endif()

# Build target for utility checking host ISA
if (ISPC_INCLUDE_UTILS)
    add_executable(check_isa "")
    target_sources(check_isa PRIVATE check_isa.cpp)
    set_target_properties(check_isa PROPERTIES FOLDER "Utils")
    if (NOT ISPC_PREPARE_PACKAGE)
        install (TARGETS check_isa DESTINATION bin)
    endif()
endif()

if (ISPC_INCLUDE_TESTS)
    add_subdirectory(tests)
endif()

if (ISPC_INCLUDE_BENCHMARKS)
    enable_testing()
    add_subdirectory(benchmarks)
endif()

if (ISPC_INCLUDE_RT)
    add_subdirectory(ispcrt)
endif()

if (ISPC_INCLUDE_EXAMPLES AND NOT ISPC_PREPARE_PACKAGE)
    include(ExternalProject)
    ExternalProject_Add(ispc_cpu_examples
        PREFIX cpu_examples
        DEPENDS ispc
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpu"
        CMAKE_CACHE_ARGS
            -DISPC_BUILD:BOOL=TRUE
            -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
            -DCMAKE_ISPC_COMPILER:PATH=$<TARGET_FILE:ispc>
            -DISPC_EXECUTABLE:PATH=$<TARGET_FILE:ispc>
            -DCMAKE_C_COMPILER:PATH=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER:PATH=${CMAKE_CXX_COMPILER}
        INSTALL_COMMAND ""
    )
endif()

if (ISPC_INCLUDE_GENX_EXAMPLES AND ISPC_INCLUDE_RT AND NOT ISPC_PREPARE_PACKAGE)
    include(ExternalProject)
    if (WIN32)
        set(ISPCRT_LIB ${CMAKE_CURRENT_BINARY_DIR}/ispcrt/${CMAKE_CFG_INTDIR}/ispcrt.lib)
    else()
        set(ISPCRT_LIB $<TARGET_FILE:ispcrt>)
    endif()
    if (ISPC_INCLUDE_BENCHMARKS)
        # Duplicated ExternalProject_Add for xpu examples when
        # ISPC_INCLUDE_BENCHMARKS is on and off since all efforts of
        # using cmake generator conditional expressions were unsuccessful.
        ExternalProject_Add(ispc_xpu_examples
            PREFIX xpu_examples
            DEPENDS ispc ispcrt benchmark
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/examples/xpu"
            CMAKE_CACHE_ARGS
                -DISPC_BUILD:BOOL=TRUE
                -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
                -DCMAKE_ISPC_COMPILER:PATH=$<TARGET_FILE:ispc>
                -DISPC_EXECUTABLE:PATH=$<TARGET_FILE:ispc>
                -DISPCRT_LIB:PATH=${ISPCRT_LIB}
                -DLEVEL_ZERO_ROOT:PATH=${LEVEL_ZERO_ROOT}
                -DBENCHMARK_LIB:PATH=$<TARGET_FILE:benchmark>
                -DCMAKE_C_COMPILER:PATH=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER:PATH=${CMAKE_CXX_COMPILER}
                -DISPC_INCLUDE_BENCHMARKS:BOOL=${ISPC_INCLUDE_BENCHMARKS}
            INSTALL_COMMAND ""
        )
    else()
        ExternalProject_Add(ispc_xpu_examples
            PREFIX xpu_examples
            DEPENDS ispc ispcrt
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/examples/xpu"
            CMAKE_CACHE_ARGS
                -DISPC_BUILD:BOOL=TRUE
                -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
                -DCMAKE_ISPC_COMPILER:PATH=$<TARGET_FILE:ispc>
                -DISPC_EXECUTABLE:PATH=$<TARGET_FILE:ispc>
                -DISPCRT_LIB:PATH=${ISPCRT_LIB}
                -DLEVEL_ZERO_ROOT:PATH=${LEVEL_ZERO_ROOT}
                -DCMAKE_C_COMPILER:PATH=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER:PATH=${CMAKE_CXX_COMPILER}
            INSTALL_COMMAND ""
    )
    endif()
endif()

# Install
install (TARGETS ${PROJECT_NAME} DESTINATION bin)
if (ISPC_PREPARE_PACKAGE)
    if (GENX_ENABLED)
        install (DIRECTORY "${PROJECT_SOURCE_DIR}/examples/" DESTINATION examples)
    else()
        install (DIRECTORY "${PROJECT_SOURCE_DIR}/examples/" DESTINATION examples PATTERN "xpu" EXCLUDE)
    endif()
    install (DIRECTORY "${PROJECT_SOURCE_DIR}/contrib/" DESTINATION contrib)
    install (FILES "${PROJECT_SOURCE_DIR}/LICENSE.txt" DESTINATION .)
    install (FILES "${PROJECT_SOURCE_DIR}/third-party-programs.txt" DESTINATION .)
    install (FILES "${PROJECT_SOURCE_DIR}/docs/ReleaseNotes.txt" DESTINATION .)

    # Clone corresponding version of documentation
    include(ExternalProject)
    ExternalProject_Add(ispc_web
      GIT_REPOSITORY    https://github.com/ispc/ispc.github.com.git
      GIT_TAG ${REPO_TAG}
      PREFIX ispc_web
      SOURCE_DIR ispc_web/repo
      CONFIGURE_COMMAND cmake -E echo "Skipping configure step."
      BUILD_COMMAND cmake -E echo "Skipping build step."
      INSTALL_COMMAND cmake -E echo "Skipping install step."
    )
    ExternalProject_Get_Property(ispc_web SOURCE_DIR)
    set(ISPC_WEB_SOURCE_DIR "${SOURCE_DIR}")
    set(ISPC_DOCS  ${ISPC_WEB_SOURCE_DIR}/faq.html
                   ${ISPC_WEB_SOURCE_DIR}/ispc.html
                   ${ISPC_WEB_SOURCE_DIR}/perfguide.html)

    install(FILES ${ISPC_DOCS} DESTINATION .)
    install(DIRECTORY "${ISPC_WEB_SOURCE_DIR}/css" DESTINATION .)

    # CPack configuration
    if (WIN32)
        set(CPACK_GENERATOR "WIX" "ZIP")
        set(ISPC_SYSTEM_NAME "windows")
        # WIX generator expects version in format x.y.z where x, y, z are numbers
        set(CPACK_PACKAGE_VERSION "${ISPC_VERSION_MAJOR}.${ISPC_VERSION_MINOR}.${ISPC_VERSION_PATCH}")
        set(CPACK_WIX_UPGRADE_GUID "EDD858F2-19B0-4E5B-B9B9-2A09B85D451C")
        set(CPACK_WIX_CMAKE_PACKAGE_REGISTRY "ISPC")
    else()
        set(CPACK_GENERATOR "TGZ")
        if (APPLE)
            set(ISPC_SYSTEM_NAME "macOS")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
            set(ISPC_SYSTEM_NAME "freebsd")
        else()
            set(ISPC_SYSTEM_NAME "linux")
        endif()
    endif()
    set(CPACK_PACKAGE_NAME "Intel(R) Implicit SPMD Program Compiler")
    set(CPACK_PACKAGE_DESCRIPTION "Intel(R) Implicit SPMD Program Compiler")
    set(CPACK_PACKAGE_VERSION_MAJOR ${ISPC_VERSION_MAJOR})
    set(CPACK_PACKAGE_VERSION_MINOR ${ISPC_VERSION_MINOR})
    set(CPACK_PACKAGE_VERSION_PATCH ${ISPC_VERSION_PATCH}${ISPC_VERSION_SUFFIX})
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    # Allow use custom package name with -DISPC_PACKAGE_NAME
    if (NOT ISPC_PACKAGE_NAME)
        if (${CPACK_PACKAGE_VERSION_PATCH} MATCHES ".*dev")
            string(CONCAT CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}"
                                "-trunk"
                                "-${ISPC_SYSTEM_NAME}")
        else()
            string(CONCAT CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}"
                                "-v${CPACK_PACKAGE_VERSION_MAJOR}"
                                ".${CPACK_PACKAGE_VERSION_MINOR}"
                                ".${CPACK_PACKAGE_VERSION_PATCH}"
                                "-${ISPC_SYSTEM_NAME}")

        endif()
    else()
        set (CPACK_PACKAGE_FILE_NAME ${ISPC_PACKAGE_NAME})
    endif()

    if (WIN32)
        set(CPACK_PACKAGE_INSTALL_DIRECTORY "ISPC/${CPACK_PACKAGE_FILE_NAME}")
    endif()
    set(CPACK_PACKAGE_VENDOR "Intel Corporation")
    include(CPack)
endif()
