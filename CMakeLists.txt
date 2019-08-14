#
#  Copyright (c) 2018-2019, Intel Corporation
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
cmake_minimum_required(VERSION 3.12)

if (UNIX)
    set(CMAKE_C_COMPILER "clang")
    set(CMAKE_CXX_COMPILER "clang++")
endif()

set(PROJECT_NAME ispc)
set(ISPC_BUILD TRUE)
project(${PROJECT_NAME})

option(ARM_ENABLED "Enable ARM support" ON)
option(NVPTX_ENABLED "Enable NVPTX support" OFF)
option(ISPC_INCLUDE_EXAMPLES "Generate build targets for the ISPC examples" ON)
option(ISPC_INCLUDE_TESTS "Generate build targets for the ISPC tests." ON)
option(ISPC_INCLUDE_UTILS "Generate build targets for the utils." ON)
option(ISPC_PREPARE_PACKAGE "Generate build targets for ispc package" OFF)
option(ISPC_NO_DUMPS "Turn off functionality, which requires LLVM dump() functions" OFF)

option(ISPC_CROSS "Build ISPC with cross compilation support" OFF)
# Default settings for cross compilation
if (ISPC_CROSS)
    option(ISPC_WINDOWS_TARGET "Build ISPC with windows target support" ON)
    option(ISPC_LINUX_TARGET "Build ISPC with linux target support" ON)
    option(ISPC_MACOS_TARGET "Build ISPC with macos target support" ON)
    option(ISPC_IOS_TARGET "Build ISPC with ios target support" ON)
    option(ISPC_ANDROID_TARGET "Build ISPC with android target support" ON)
    option(ISPC_PS4_TARGET "Build ISPC with ps4 target support" ON)
    if (WIN32)
        if ((ISPC_LINUX_TARGET OR ISPC_ANDROID_TARGET OR ISPC_PS4_TARGET) AND NOT ISPC_GNUWIN32_PATH)
            message (FATAL_ERROR "Set ISPC_GNUWIN32_PATH variable for cross compilation to Linux/Android/PS4 e.g. C:/gnuwin32")
        endif()
        if (ISPC_MACOS_TARGET AND NOT ISPC_MACOS_SDK_PATH)
            message (FATAL_ERROR "Set ISPC_MACOS_SDK_PATH variable for cross compilation to MacOS e.g. C:/iusers/MacOSX10.14.sdk")
        endif()
    elseif (APPLE)
        if ((ISPC_LINUX_TARGET OR ISPC_ANDROID_TARGET) AND NOT ISPC_ANDROID_NDK_PATH)
            message (FATAL_ERROR "Set ISPC_ANDROID_NDK_PATH variable for cross compilation to Linux/Android e.g. /Users/Shared/android-ndk-r20")
        elseif (ISPC_IOS_TARGET AND NOT ISPC_IOS_SDK_PATH)
            message (FATAL_ERROR "Set ISPC_IOS_SDK_PATH variable for cross compilation to IOS "
                "e.g. /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk")
        endif()
    else()
        if (ISPC_MACOS_TARGET AND NOT ISPC_MACOS_SDK_PATH)
            message (FATAL_ERROR "Set ISPC_MACOS_SDK_PATH variable for cross compilation to MacOS e.g. /iusers/MacOSX10.14.sdk")
        endif()
    endif()
else()
    if (WIN32)
        set(ISPC_WINDOWS_TARGET ON)
    elseif (APPLE)
        set(ISPC_MACOS_TARGET ON)
    else()
        set(ISPC_LINUX_TARGET ON)
    endif()
endif()

if (UNIX)
    option(ISPC_STATIC_STDCXX_LINK "Link statically with libstdc++ and libgcc" OFF)
    if (ISPC_PREPARE_PACKAGE AND (NOT APPLE))
        option(ISPC_STATIC_LINK "Link statically" ON)
    else()
        option(ISPC_STATIC_LINK "Link statically" OFF)
    endif()
    option(ISPC_USE_ASAN "Build ispc with address sanitizer instrumentation using clang compiler" OFF)
endif()

# Use solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(OUTPUT_DEBUG Debug/bin)
set(OUTPUT_RELEASE Release/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin )

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Build type not specified: Use Release by default.")
endif(NOT CMAKE_BUILD_TYPE)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FixWindowsPath.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/LLVMConfig.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Git.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateBuiltins.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Stdlib.cmake)

find_package(Python3 REQUIRED)
    if (NOT Python3_Interpreter_FOUND)
        message(FATAL_ERROR "Python interpreter is not found")
    endif()

find_package(BISON 2.4 REQUIRED)
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

find_package(FLEX 2.5 REQUIRED)
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
set (ISPC_TARGETS
        sse2-i32x4 sse2-i32x8
        sse4-i32x4 sse4-i32x8 sse4-i8x16 sse4-i16x8
        # Note here that avx1-i32x4 is aliased to sse4-i32x4 plus avx attribute.
        avx1-i32x8 avx1-i32x16 avx1-i64x4
        avx2-i32x4 avx2-i32x8 avx2-i32x16 avx2-i64x4
        avx512knl-i32x16
        avx512skx-i32x8 avx512skx-i32x16
        generic-1 generic-4 generic-8 generic-16 generic-32 generic-64)

set(CLANG_LIBRARY_LIST clangFrontend clangDriver clangSerialization clangParse clangSema clangAnalysis clangAST clangBasic clangEdit clangLex)
set(LLVM_COMPONENTS engine ipo bitreader bitwriter instrumentation linker)
# Component "option" was introduced in 3.3 and starting with 3.4 it is required for the link step.
if (${LLVM_VERSION_NUMBER} VERSION_GREATER "3.3.0")
    set(LLVM_COMPONENTS ${LLVM_COMPONENTS} option)
endif()

list(APPEND LLVM_COMPONENTS x86)
if (ARM_ENABLED)
    list(APPEND LLVM_COMPONENTS arm aarch64)
    list(APPEND ISPC_TARGETS neon-i8x16 neon-i16x8 neon-i32x4 neon-i32x8)
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
               ${STDLIB_FILES} ${BISON_CPP_OUTPUT} ${FLEX_OUTPUT}
               ${CMAKE_CURRENT_SOURCE_DIR}/stdlib.ispc)

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
        "src/parse.yy"
        "src/lex.ll"
)
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
                           ${CMAKE_CURRENT_SOURCE_DIR}/src
                           ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
# Compile options
if (UNIX)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wno-sign-compare -Wno-unused-function -Werror ${LLVM_CPP_FLAGS})
    # Security options
    target_compile_options(${PROJECT_NAME} PRIVATE -fstack-protector -fdata-sections -ffunction-sections
                           -Wformat -Wformat-security -fpie -fwrapv)
    set_source_files_properties("src/cbackend.cpp" PROPERTIES COMPILE_FLAGS "-fno-exceptions")
else()
    target_compile_options(${PROJECT_NAME} PRIVATE /W3 /wd4146 /wd4800 /wd4996 /wd4355 /wd4624 /wd4244 /wd4141 /wd4291 /wd4018)
    # Security options
    target_compile_options(${PROJECT_NAME} PRIVATE /GS /DynamicBase)
    set_source_files_properties(${FLEX_OUTPUT} PROPERTIES COMPILE_FLAGS "/wd4005 /wd4003")
    set_source_files_properties(${BISON_OUTPUT} PROPERTIES COMPILE_FLAGS "/wd4005 /wd4065")
endif()

if (${LLVM_VERSION_NUMBER} VERSION_GREATER "3.4.0")
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
    if (UNIX)
        set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)
        target_compile_options(${PROJECT_NAME} PRIVATE -Wno-c99-extensions -Wno-deprecated-register -fno-rtti)
        if (ISPC_USE_ASAN)
            target_compile_options(${PROJECT_NAME} PRIVATE -fsanitize=address)
        endif()
    endif()
endif()

# Link options
if (WIN32)
    if (NOT CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "/OPT:REF /OPT:ICF")
    endif()
elseif (APPLE)
else()
    # Link options for security hardening.
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-z noexecstack -z relro -z now -Wl,--gc-sections")
endif()

if (ISPC_STATIC_STDCXX_LINK)
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-static-libgcc -static-libstdc++")
endif()

if (ISPC_STATIC_LINK)
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-static")
endif()

if (ISPC_USE_ASAN)
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-fsanitize=address")
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

# Build target for utility checking host ISA
if (ISPC_INCLUDE_UTILS)
    add_executable(check_isa "")
    target_sources(check_isa PRIVATE check_isa.cpp)
    set_target_properties(check_isa PROPERTIES FOLDER "Utils")
    if (NOT ISPC_PREPARE_PACKAGE)
        install (TARGETS check_isa DESTINATION bin)
    endif()
endif()

if (ISPC_INCLUDE_EXAMPLES AND NOT ISPC_PREPARE_PACKAGE)
    add_subdirectory(examples)
endif()

if (ISPC_INCLUDE_TESTS)
    add_subdirectory(tests)
endif()

# Install
install (TARGETS ${PROJECT_NAME} DESTINATION bin)
if (ISPC_PREPARE_PACKAGE)
    install (DIRECTORY "${PROJECT_SOURCE_DIR}/examples/" DESTINATION examples PATTERN "portable" EXCLUDE)
    install (DIRECTORY "${PROJECT_SOURCE_DIR}/contrib/" DESTINATION contrib)
    install (FILES "${PROJECT_SOURCE_DIR}/LICENSE.txt" DESTINATION .)
    install (FILES "${PROJECT_SOURCE_DIR}/docs/ReleaseNotes.txt" DESTINATION .)
    include(ExternalProject)
    ExternalProject_Add(ispc_web
      GIT_REPOSITORY    https://github.com/ispc/ispc.github.com.git
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
    install(DIRECTORY "${ISPC_WEB_SOURCE_DIR}/css" DESTINATION css)

    # CPack configuration
    if (WIN32)
        set(CPACK_GENERATOR "ZIP")
        set(ISPC_SYSTEM_NAME "windows")
    else()
        set(CPACK_GENERATOR "TGZ")
        if (APPLE)
            set(ISPC_SYSTEM_NAME "macOS")
        else()
            set(ISPC_SYSTEM_NAME "linux")
        endif()
    endif()
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
    # Get ispc version
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/ispc_version.h" ispc_ver)
    string(REGEX MATCH "ISPC_VERSION \"([0-9]*)\.([0-9]*)\.([0-9]*[a-z]*)" _ ${ispc_ver})
    set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_MATCH_2})
    set(CPACK_PACKAGE_VERSION_PATCH ${CMAKE_MATCH_3})
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
    include(CPack)
endif()
