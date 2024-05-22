#
#  Copyright (c) 2018-2024, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc CMakeLists.txt
#
cmake_minimum_required(VERSION 3.15)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# Set C++ standard to C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Set C standard to C11
set(CMAKE_C_STANDARD   11)
set(CMAKE_C_STANDARD_REQUIRED   ON)
set(CMAKE_C_EXTENSIONS OFF)


if (UNIX)
    if (NOT CMAKE_C_COMPILER)
        set(CMAKE_C_COMPILER "clang")
    endif()
    if (NOT CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER "clang++")
    endif()
endif()

# Generate compile_commands.json by default - this enables VSCode to do better job interpretting
# C++ files form the project.
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON" CACHE BOOL "Export compile commands")

if (APPLE)
    # Setting ISPC_MACOS_UNIVERSAL_BINARIES to ON is conventional way of building ISPC as Universal Binary.
    # If cross compilation is needed (arm64->x86_64 or x86_64->arm64), specify CMAKE_OSX_ARCHITECTURES
    # on the command line explicitly
    option(ISPC_MACOS_UNIVERSAL_BINARIES "Build Universal Binaries on macOS" OFF)
    if (ISPC_MACOS_UNIVERSAL_BINARIES)
        message(STATUS "Building Universal Binaries (x86_64+arm64)")
        set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "OSX architectures")
    endif()

    # We need to target different minimum macOS versions on x86 and arm by default, but before
    # we execute project() command, the regular CMAKE_* variables with host info are not initialized,
    # so we do this hack to detect the platform. It's not perfect, but good enough.
    execute_process(COMMAND uname -m OUTPUT_VARIABLE UNAME)
    if (UNAME MATCHES "x86_64" OR CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
        # Note: it's ok to have 10.12 for Universal Binaries.
        set(CMAKE_OSX_DEPLOYMENT_TARGET "10.12" CACHE STRING "Minimum macOS version to support")
    else()
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version to support")
    endif()
    message(STATUS "Targeting minimum macOS version: ${CMAKE_OSX_DEPLOYMENT_TARGET}")

endif()

set(PROJECT_NAME ispc)
set(ISPC_BUILD TRUE)
project(${PROJECT_NAME})

## Check PIE support
include(CheckPIESupported)
check_pie_supported(OUTPUT_VARIABLE CHECK_PIE_OUTPUT LANGUAGES CXX)
if(NOT CMAKE_CXX_LINK_PIE_SUPPORTED)
    message(WARNING "PIE is not supported at link time: ${CHECK_PIE_OUTPUT}.\n"
                    "PIE link options will not be passed to linker.")
endif()

set(X86_HOST FALSE)
if (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "amd64|AMD64|86")
    set(X86_HOST TRUE)
endif()

option(X86_ENABLED "Enable x86 support" ${X86_HOST})
option(ARM_ENABLED "Enable ARM support" ON)
option(WASM_ENABLED "Enable experimental Web Assembly support" OFF)
option(XE_ENABLED "Enable Intel Xe support" OFF)
option(ISPC_INCLUDE_EXAMPLES "Generate build targets for the ISPC examples" ON)
option(ISPC_INCLUDE_DPCPP_EXAMPLES "Generate build targets for the ISPC/DPCPP interoperability examples" OFF)
option(ISPC_INCLUDE_TESTS "Generate build targets for the ISPC tests." ON)
option(ISPC_INCLUDE_BENCHMARKS "Generate build targets for the ISPC tests." OFF)
option(ISPC_INCLUDE_RT "Generate build targets for ISPC runtime." ON)
option(ISPC_INCLUDE_UTILS "Generate build targets for the utils." ON)
option(ISPC_INCLUDE_UTILS_ONLY "Generate build targets only for the utils." OFF)
option(ISPC_PREPARE_PACKAGE "Generate build targets for ispc package" OFF)
option(ISPC_PACKAGE_EXAMPLES "Pack examples into the ISPC package" ON)

option(ISPC_OPAQUE_PTR_MODE "Build ISPC with usage of opaque pointers" OFF)

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
    option(ISPC_PS_TARGET "Build ISPC with ps4 and ps5 targets support" ON)
    # We silently disable some of cross targets. We should probably warn user or issue an error,
    # but it seems good enough.
    if (WIN32)
        set(ISPC_IOS_TARGET OFF)
        if ((ISPC_LINUX_TARGET OR ISPC_ANDROID_TARGET OR ISPC_PS_TARGET) AND NOT ISPC_GNUWIN32_PATH)
            message (FATAL_ERROR "Set ISPC_GNUWIN32_PATH variable for cross compilation to Linux/Android/PS4 e.g. C:/gnuwin32")
        endif()
        if (ISPC_MACOS_TARGET AND NOT ISPC_MACOS_SDK_PATH)
            message (FATAL_ERROR "Set ISPC_MACOS_SDK_PATH variable for cross compilation to MacOS e.g. C:/iusers/MacOSX10.14.sdk")
        endif()
    elseif (APPLE)
        set(ISPC_MACOS_TARGET ON)
        set(ISPC_WINDOWS_TARGET OFF)
        set(ISPC_PS_TARGET OFF)
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
        set(ISPC_PS_TARGET OFF)
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

    # Workaround CMake/CPack issue https://gitlab.kitware.com/cmake/cmake/-/issues/24601
    # When llvm toolchain is presented in PATH, llvm-strip is preferred over
    # macOS strip. However, CPack fails to correctly call llvm-strip because it
    # adds flags like `-u` that are macOS strip specific.
    # It may be removed later when CMake fixes ambiguity.
    if (${CMAKE_VERSION} VERSION_LESS "3.27.0")
        set(CMAKE_STRIP strip)
    endif()
endif()

set(ISPC_MACOS_ARM_TARGET OFF)
if (ISPC_MACOS_TARGET AND ISPC_MACOS_SDK_PATH)
    # Get macOS SDK version.
    # xcrun is known not to work for standalone SDK or in the default environment
    # on some systems (without explicitly running "xcode-select -s <path>" first).
    # So we try to get the version from the SDK path first, and then fall back
    # to looking at the SDK folder name.
    set(command "xcrun" "--sdk" ${ISPC_MACOS_SDK_PATH} "--show-sdk-version")
    execute_process(COMMAND ${command}
        OUTPUT_VARIABLE SDK_VER
        RESULT_VARIABLE XCRUN_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (NOT XCRUN_RESULT STREQUAL "0")
        if (IS_SYMLINK ${ISPC_MACOS_SDK_PATH})
            # Follow symlinks, relative and absolute.
            file(READ_SYMLINK "${ISPC_MACOS_SDK_PATH}" ISPC_MACOS_SDK_PATH_NEW)
            if (NOT IS_ABSOLUTE "${ISPC_MACOS_SDK_PATH_NEW}")
                get_filename_component(ISPC_MACOS_SDK_PATH_DIR "${ISPC_MACOS_SDK_PATH}" DIRECTORY)
                set(ISPC_MACOS_SDK_PATH_NEW "${ISPC_MACOS_SDK_PATH_DIR}/${ISPC_MACOS_SDK_PATH_NEW}")
            endif()
            set(ISPC_MACOS_SDK_PATH "${ISPC_MACOS_SDK_PATH_NEW}")
        endif()

        string(REGEX MATCH "MacOSX([0-9]*.[0-9]*).sdk" _ ${ISPC_MACOS_SDK_PATH})
        set(SDK_VER "${CMAKE_MATCH_1}")
    endif()

    message(STATUS "MacOS_SDK version: ${SDK_VER}")

    if ("${SDK_VER}" STREQUAL "")
        message(WARNING "MacOS SDK version was not detected, assuming 11.0 or later, enabling ARM support")
        set(ISPC_MACOS_ARM_TARGET ON)
    elseif("${SDK_VER}" VERSION_GREATER_EQUAL "11.0")
        message(STATUS "MacOS_SDK supports ARM (SDK ver >= 11.0)")
        set(ISPC_MACOS_ARM_TARGET ON)
    else()
        message(STATUS "MacOS_SDK does NOT supports ARM (SDK ver < 11.0)")
    endif()
endif()

# If Xe target is enabled, generate build targets for Xe examples and ISPCRT on Linux
if (XE_ENABLED)
    option(ISPC_INCLUDE_XE_EXAMPLES "Generate build targets for the ISPC Xe examples" ON)
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


# Build target for utility checking host ISA
if (ISPC_INCLUDE_UTILS)
    add_executable(check_isa "")
    target_sources(check_isa PRIVATE check_isa.cpp)
    set_target_properties(check_isa PROPERTIES FOLDER "Utils")
    if (NOT ISPC_PREPARE_PACKAGE)
        install (TARGETS check_isa DESTINATION bin)
    endif()
    if (ISPC_INCLUDE_UTILS_ONLY)
        # Skip the below part of CMake. It is useful when only check_isa compiling is needed.
        return()
    endif()
endif()


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ispcrt/cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FixWindowsPath.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindLLVM.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Git.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateBuiltins.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Stdlib.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Version.cmake)

# Get ispc version
get_ispc_version("${CMAKE_CURRENT_SOURCE_DIR}/common/version.h")

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
            MAIN_DEPENDENCY ${BISON_INPUT}
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
            MAIN_DEPENDENCY ${FLEX_INPUT}
            COMMENT "Generating lex.cpp"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()

set (ISPC_MASKS 1 8 16 32 64)

if (NOT X86_ENABLED AND NOT ARM_ENABLED AND NOT WASM_ENABLED AND NOT XE_ENABLED)
    message( FATAL_ERROR "Either X86, ARM, WASM or XE targets need to be enabled.")
endif ()

if (X86_ENABLED)
    list(APPEND ISPC_TARGETS
        sse2-i32x4 sse2-i32x8
        sse4-i32x4 sse4-i32x8 sse4-i8x16 sse4-i16x8
        # Note here that avx1-i32x4 is aliased to sse4-i32x4 plus avx attribute,
        # sse41* targets are aliased to corresponding sse4 targets.
        avx1-i32x8 avx1-i32x16 avx1-i64x4
        avx2-i8x32 avx2-i16x16
        avx2-i32x4 avx2-i32x8 avx2-i32x16 avx2-i64x4
        avx2vnni-i32x4 avx2vnni-i32x8 avx2vnni-i32x16
        avx512knl-x16
        avx512skx-x4 avx512skx-x8 avx512skx-x16
        avx512skx-x32 avx512skx-x64
        avx512icl-x4 avx512icl-x8 avx512icl-x16
        avx512icl-x32 avx512icl-x64)
    if (${LLVM_VERSION_NUMBER} VERSION_GREATER_EQUAL "14.0.0")
        list(APPEND ISPC_TARGETS
            avx512spr-x4 avx512spr-x8 avx512spr-x16 avx512spr-x32 avx512spr-x64)
    endif()
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

set(CLANG_LIBRARY_LIST clangFrontend clangBasic clangEdit clangLex)
if (${LLVM_VERSION_NUMBER} VERSION_GREATER_EQUAL "15.0.0")
    list(APPEND CLANG_LIBRARY_LIST clangSupport)
endif()
if (${LLVM_VERSION_NUMBER} VERSION_GREATER_EQUAL "16.0.0")
    list(APPEND CLANG_LIBRARY_LIST clangASTMatchers)
endif()

set(LLVM_COMPONENTS engine ipo bitreader bitwriter instrumentation linker option frontendopenmp passes)

if (${LLVM_VERSION_NUMBER} VERSION_GREATER_EQUAL "15.0.0")
    # windowsdriver is a small library introduced in LLVM 15. While it's targeted at Windows only,
    # it's used in the LLVM code without any ifdef and is needed on all platforms.
    # An interesting fact, that on Linux it is not required, because linker is smart enough to optimize out
    # the calls to unresolved symbols. It's not the case on macOS though.
    list(APPEND LLVM_COMPONENTS windowsdriver)
endif()

if (X86_ENABLED)
    list(APPEND LLVM_COMPONENTS x86)
endif()
if (ARM_ENABLED)
    list(APPEND LLVM_COMPONENTS arm aarch64)
endif()
if (WASM_ENABLED)
    list(APPEND LLVM_COMPONENTS webassembly)
endif()
if (XE_ENABLED)
    list(APPEND ISPC_TARGETS gen9-x16 gen9-x8 xelp-x16 xelp-x8 xehpg-x16 xehpg-x8 xehpc-x16 xehpc-x32 xelpg-x16 xelpg-x8)
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(TARGET_MODIFIER "32")
        set(PB_PATH_MODIFIER "x86")
    else()
        set(TARGET_MODIFIER "64")
        set(PB_PATH_MODIFIER "x64")
    endif()

    list(APPEND XE_LIBRARY_LIST LLVMGenXIntrinsics LLVMSPIRVLib)
endif()

# Set the flags responsible for generation of opaque pointers in builtins
set(ISPC_OPAQUE_FLAGS)
set(LLVM_TOOLS_OPAQUE_FLAGS)
# Starting LLVM version 17.0 enable opaque pointers by default for CPU build
if (${LLVM_VERSION_NUMBER} VERSION_GREATER_EQUAL "17.0.0")
    set(ISPC_OPAQUE_PTR_MODE ON)
else()
    # On Xe-enabled build, turn off opaque pointers before 17.0.
    # IGC software stack is not ready yet.
    if (XE_ENABLED)
        set(ISPC_OPAQUE_PTR_MODE OFF)
    endif()
endif()

if (ISPC_OPAQUE_PTR_MODE)
    if (${LLVM_VERSION_NUMBER} VERSION_GREATER_EQUAL "15.0.0")
    # Do nothing, opaque pointers mode is default
    elseif (${LLVM_VERSION_NUMBER} VERSION_EQUAL "14.0.0")
    # Explicitly enable opaque pointers for LLVM 14.0
        set(ISPC_OPAQUE_FLAGS "-Xclang" "-opaque-pointers")
    else()
        message(FATAL_ERROR "ISPC opaque pointers mode is not supported with LLVM " ${LLVM_VERSION_NUMBER})
    endif()
else()
    # Explicitly disable opaque pointers starting LLVM 15.0 till LLVM 17.0
    if (${LLVM_VERSION_NUMBER} VERSION_GREATER_EQUAL "15.0.0" AND ${LLVM_VERSION_NUMBER} VERSION_LESS "17.0.0")
        set(ISPC_OPAQUE_FLAGS "-Xclang" "-no-opaque-pointers")
        set(LLVM_TOOLS_OPAQUE_FLAGS "-opaque-pointers=0")
    endif()
endif()
message(STATUS "ISPC opaque pointers mode is " ${ISPC_OPAQUE_PTR_MODE})

get_llvm_libfiles(LLVM_LIBRARY_LIST ${LLVM_COMPONENTS})
get_llvm_cppflags(LLVM_CPP_FLAGS)

generate_target_builtins(BUILTIN_FILES ${ISPC_TARGETS})
generate_common_builtins(BUILTIN_CPP_FILES)
generate_stdlib(STDLIB_FILES ${ISPC_MASKS})

# Common sources between ispc-opt and ispc
list(APPEND COMMON_SOURCES
    common/version.rc
    common/version.h
    src/bitcode_lib.cpp
    src/bitcode_lib.h
    src/builtins-decl.cpp
    src/builtins-decl.h
    src/ispc.cpp
    src/ispc.h
    src/ispc_version.h
    src/llvmutil.cpp
    src/llvmutil.h
    src/target_enums.cpp
    src/target_enums.h
    src/target_registry.cpp
    src/target_registry.h
    src/util.cpp
    src/util.h
)

# Source files used only in ispc
list(APPEND FRONTEND_SOURCES
    ${STDLIB_FILES}
    ${BISON_CPP_OUTPUT}
    ${FLEX_OUTPUT}
    ${CMAKE_CURRENT_SOURCE_DIR}/stdlib.ispc
    src/ast.cpp
    src/ast.h
    src/builtins.cpp
    src/builtins.h
    src/ctx.cpp
    src/ctx.h
    src/decl.cpp
    src/decl.h
    src/expr.cpp
    src/expr.h
    src/func.cpp
    src/func.h
    src/module.cpp
    src/module.h
    src/stmt.cpp
    src/stmt.h
    src/sym.cpp
    src/sym.h
    src/type.cpp
    src/type.h
    src/parse.yy
    src/lex.ll
)

# ISPC optimization passes shared between ispc and ispc-opt
list(APPEND OPT_SOURCES
    src/opt.cpp
    src/opt.h
    src/opt/CheckIRForXeTarget.cpp
    src/opt/CheckIRForXeTarget.h
    src/opt/GatherCoalescePass.cpp
    src/opt/GatherCoalescePass.h
    src/opt/IsCompileTimeConstant.cpp
    src/opt/IsCompileTimeConstant.h
    src/opt/ImproveMemoryOps.cpp
    src/opt/ImproveMemoryOps.h
    src/opt/InstructionSimplify.cpp
    src/opt/InstructionSimplify.h
    src/opt/IntrinsicsOptPass.cpp
    src/opt/IntrinsicsOptPass.h
    src/opt/MangleOpenCLBuiltins.cpp
    src/opt/MangleOpenCLBuiltins.h
    src/opt/MakeInternalFuncsStatic.cpp
    src/opt/MakeInternalFuncsStatic.h
    src/opt/PeepholePass.cpp
    src/opt/PeepholePass.h
    src/opt/ReplaceMaskedMemOps.cpp
    src/opt/ReplaceMaskedMemOps.h
    src/opt/ReplacePseudoMemoryOps.cpp
    src/opt/ReplacePseudoMemoryOps.h
    src/opt/ReplaceStdlibShiftPass.cpp
    src/opt/ReplaceStdlibShiftPass.h
    src/opt/ScalarizePass.cpp
    src/opt/ScalarizePass.h
    src/opt/XeGatherCoalescePass.cpp
    src/opt/XeGatherCoalescePass.h
    src/opt/XeReplaceLLVMIntrinsics.cpp
    src/opt/XeReplaceLLVMIntrinsics.h
)

# To show stdlib.ispc in VS solution:
if (WIN32)
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/stdlib.ispc" PROPERTIES HEADER_FILE_ONLY TRUE)
    source_group("ISPC" FILES "${CMAKE_CURRENT_SOURCE_DIR}/stdlib.ispc")
endif()

# Build definitions
string(TIMESTAMP BUILD_DATE "%Y%m%d" UTC)
list(APPEND COMPILE_DEFINITIONS
    BUILD_DATE=\"${BUILD_DATE}\"
    BUILD_VERSION=\"${GIT_COMMIT_HASH}\"
)
if (UNIX)
    # Compile-time protection against static sized buffer overflows.
    list(APPEND COMPILE_DEFINITIONS _FORTIFY_SOURCE=2)
else()
    list(APPEND COMPILE_DEFINITIONS NOMINMAX)
    if (NOT CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
        list(APPEND COMPILE_DEFINITIONS NDEBUG)
    endif()
endif()

if (X86_ENABLED)
    list(APPEND COMPILE_DEFINITIONS ISPC_X86_ENABLED)
endif()

if (ARM_ENABLED)
    list(APPEND COMPILE_DEFINITIONS ISPC_ARM_ENABLED)
endif()

if (XE_ENABLED)
    list(APPEND COMPILE_DEFINITIONS ISPC_XE_ENABLED)
endif()

if (WASM_ENABLED)
    list(APPEND COMPILE_DEFINITIONS ISPC_WASM_ENABLED)
endif()

# Compile definitions for cross compilation
if (NOT ISPC_WINDOWS_TARGET)
    list(APPEND COMPILE_DEFINITIONS ISPC_WINDOWS_TARGET_OFF)
endif()
if (NOT ISPC_LINUX_TARGET)
    list(APPEND COMPILE_DEFINITIONS ISPC_LINUX_TARGET_OFF)
endif()
if (NOT ISPC_FREEBSD_TARGET)
    list(APPEND COMPILE_DEFINITIONS ISPC_FREEBSD_TARGET_OFF)
endif()
if (NOT ISPC_MACOS_TARGET)
    list(APPEND COMPILE_DEFINITIONS ISPC_MACOS_TARGET_OFF)
endif()
if (NOT ISPC_IOS_TARGET)
    list(APPEND COMPILE_DEFINITIONS ISPC_IOS_TARGET_OFF)
endif()
if (NOT ISPC_ANDROID_TARGET)
    list(APPEND COMPILE_DEFINITIONS ISPC_ANDROID_TARGET_OFF)
endif()
if (NOT ISPC_PS_TARGET)
    list(APPEND COMPILE_DEFINITIONS ISPC_PS_TARGET_OFF)
endif()

# Compile definition for opaque pointers mode
if (ISPC_OPAQUE_PTR_MODE)
    list(APPEND COMPILE_DEFINITIONS ISPC_OPAQUE_PTR_MODE)
endif()


# Include directories
list(APPEND INCLUDE_DIRECTORIES
    ${LLVM_INCLUDE_DIRS}
    ${XE_DEPS_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
)

list(APPEND COMPILE_OPTIONS)
# Compile options
if (MSVC)
    list(APPEND COMPILE_OPTIONS
        /W3
        /wd4018
        /wd4146
        /wd4244
        /wd4267
        /wd4624
        /wd4996
    )
    # Security options
    list(APPEND COMPILE_OPTIONS
        /GS
        /EHsc
    )
    set_source_files_properties(${FLEX_OUTPUT} PROPERTIES COMPILE_FLAGS /wd4005)
    set_source_files_properties(${BISON_OUTPUT} PROPERTIES COMPILE_FLAGS /wd4005)
else()
    list(APPEND COMPILE_OPTIONS
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-sign-compare
        -Wno-unused-function
        ${LLVM_CPP_FLAGS}
    )
    # Security options for both Debug and Release builds
    list(APPEND COMPILE_OPTIONS
        -fstack-protector-strong
        -fno-delete-null-pointer-checks
        -Wformat
        -Wformat-security
        -fwrapv
    )
    # Additional security options for Release mode
    # Should be combined with link-time optimization (-Wl,--gc-sections during the linking phase)
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        list(APPEND COMPILE_OPTIONS
            -fdata-sections
            -ffunction-sections
        )
    endif()
    # LLVM 14 switches default DWARF version from v4 to v5
    # but gdb < 10.1 can't read DWARF v5.
    # https://discourse.llvm.org/t/gdb-10-1-cant-read-clangs-dwarf-v5/6035
    # Use DWARF v4 for ISPC build to allow debugging of ispc binary with gdb
    # on Ubuntu 20.04
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Release" AND ${LLVM_VERSION_NUMBER} VERSION_GREATER_EQUAL "14.0.0")
        list(APPEND COMPILE_OPTIONS -gdwarf-4)
    endif()
endif()


if (NOT MSVC)
    list(APPEND COMPILE_OPTIONS -fno-rtti)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND COMPILE_OPTIONS
            -Wno-register
            -Wno-error=comment
        )
        # The GCC warning issues false positives when one of the allocation and
        # deallocation functions is inlined into its caller but not the other.
        list(APPEND COMPILE_OPTIONS -Wno-mismatched-new-delete)
    else()
        list(APPEND COMPILE_OPTIONS
            -Wno-c99-extensions
            -Wno-deprecated-register
        )
    endif()
    if (ISPC_USE_ASAN)
        list(APPEND COMPILE_OPTIONS -fsanitize=address)
    endif()
endif()


list(APPEND LINK_OPTIONS_PRIVATE)
list(APPEND LINK_OPTIONS_PUBLIC)
# Link options
if (WIN32)
    list(APPEND LINK_OPTIONS_PRIVATE /DYNAMICBASE)
    # Control flow guard
    list(APPEND LINK_OPTIONS_PRIVATE /GUARD:CF)
    if (MSVC AND NOT CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
        list(APPEND LINK_OPTIONS_PUBLIC /OPT:REF /OPT:ICF)
    endif()
elseif (APPLE)
else()
    # Common security hardening options for both Debug and Release
    list(APPEND LINK_OPTIONS_PUBLIC
        "SHELL: -z noexecstack"
        "SHELL: -z relro"
        "SHELL: -z now"
    )
    # Additional options for Release mode
    if(CMAKE_BUILD_TYPE MATCHES Release)
        list(APPEND LINK_OPTIONS_PUBLIC "SHELL: -Wl,--gc-sections")
    endif()
endif()

if (ISPC_STATIC_STDCXX_LINK OR ISPC_STATIC_LINK)
    list(APPEND LINK_OPTIONS_PUBLIC
        -static-libgcc
        -static-libstdc++
    )
endif()

if (ISPC_USE_ASAN)
    list(APPEND LINK_OPTIONS_PUBLIC -fsanitize=address)
endif()


list(APPEND LINK_LIBRARIES)
if (NOT WIN32 AND NOT APPLE)
    # To resolve circular dependencies between libraries use --start-group/--end-group
    list(APPEND LINK_LIBRARIES "-Wl,--start-group")
endif()

# Link against Clang libraries
foreach(clangLib ${CLANG_LIBRARY_LIST})
    # Preferred way to build ISPC is to link it with individual clang static
    # libraries. However, when they omitted it can be link with shared library
    # libclang-cpp.so. Second approach would be utilized in some controlled
    # environments, like in package managers for particular OSes (issue 2418).
    find_library(${clangLib}Path
                 NAMES
                    "${CMAKE_STATIC_LIBRARY_PREFIX}${clangLib}${CMAKE_STATIC_LIBRARY_SUFFIX}"
                    clang-cpp
                 HINTS ${LLVM_LIBRARY_DIRS})
    if (NOT ${${clangLib}Path} IN_LIST CLANG_LIBRARY_FULL_PATH_LIST)
        list(APPEND CLANG_LIBRARY_FULL_PATH_LIST ${${clangLib}Path})
    endif()
endforeach()
list(APPEND LINK_LIBRARIES ${CLANG_LIBRARY_FULL_PATH_LIST})

# Link against LLVM libraries
list(APPEND LINK_LIBRARIES ${LLVM_LIBRARY_LIST} ${CMAKE_DL_LIBS})

if (XE_ENABLED)
    # Link against Xe libraries
    foreach(xeLib ${XE_LIBRARY_LIST})
        find_library(${xeLib}Path NAMES ${xeLib} HINTS ${XE_DEPS_DIR}/lib)
        list(APPEND XE_LIBRARY_FULL_PATH_LIST ${${xeLib}Path})
    endforeach()
    list(APPEND LINK_LIBRARIES ${XE_LIBRARY_FULL_PATH_LIST})
endif()

if (NOT WIN32 AND NOT APPLE)
    list(APPEND LINK_LIBRARIES "-Wl,--end-group")
endif()

# System libraries, our own and transitive dependencies from LLVM libs.
if (WIN32)
    list(APPEND LINK_LIBRARIES version.lib shlwapi.lib odbc32.lib odbccp32.lib)
else()
    if (ISPC_STATIC_LINK)
        list(APPEND LINK_LIBRARIES pthread tinfo.a curses.a)
    else()
        find_package(Curses REQUIRED)
        if (CURSES_EXTRA_LIBRARY)
            # this contains the tinfo library, if found
            list(APPEND LINK_LIBRARIES ${CURSES_EXTRA_LIBRARY})
        endif()
        # Recommended way to get Curses library is to use CURSES_LIBRARIES variable
        # https://cmake.org/cmake/help/v3.27/module/FindCurses.html
        # but it contains both libform.so and libcurses.so.
        # We need libcurses.so only, so exclude libform.so from the list.
        list(FILTER CURSES_LIBRARIES EXCLUDE REGEX ".*libform.*")
        list(APPEND LINK_LIBRARIES pthread ${CURSES_LIBRARIES})
    endif()
endif()

# This function configures compile definitions, options and include dirs.
# It can be used for any objects or libraries that are built in the project.
function(configure_ispc_obj TARGET)
    target_compile_definitions(${TARGET} PRIVATE ${COMPILE_DEFINITIONS})
    target_include_directories(${TARGET} PRIVATE ${INCLUDE_DIRECTORIES})
    target_compile_options(${TARGET} PRIVATE ${COMPILE_OPTIONS})

    # Set hidden visibility for inline functions. This is needed to be in sync
    # with LLVM libraries, we do include some template code from LLVM that requires
    # this options to be in sync with LLVM. Otherwise it causes link warnings.
    set_target_properties(${TARGET} PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
    if (NOT MSVC)
        set_target_properties(${TARGET} PROPERTIES CXX_EXTENSIONS OFF)
    endif()
endfunction()

# This function configures executable targets.
function(configure_ispc_exe TARGET)
    configure_ispc_obj(${TARGET})
    target_link_options(${TARGET} PRIVATE ${LINK_OPTIONS_PRIVATE})
    target_link_options(${TARGET} PUBLIC ${LINK_OPTIONS_PUBLIC})
endfunction()

# Define libraries that are built in the project. Some of them are shared across multiple executable targets.
add_library(builtin OBJECT ${BUILTIN_FILES} ${BUILTIN_CPP_FILES})
configure_ispc_obj(builtin)

add_library(optimization OBJECT ${OPT_SOURCES})
configure_ispc_obj(optimization)

add_library(common OBJECT ${COMMON_SOURCES})
configure_ispc_obj(common)

add_library(frontend OBJECT ${FRONTEND_SOURCES})
configure_ispc_obj(frontend)

# Define executable targets
add_executable(${PROJECT_NAME} src/main.cpp)
configure_ispc_exe(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARIES} builtin optimization common frontend)

add_executable(ispc-opt ispc-opt.cpp)
configure_ispc_exe(ispc-opt)
# TODO! should we link against all the same LLVM/Clang libraries? Clang libs is definetely is not needed
target_link_libraries(ispc-opt ${LINK_LIBRARIES} builtin optimization common)


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

if (ISPC_INCLUDE_XE_EXAMPLES AND ISPC_INCLUDE_RT AND NOT ISPC_PREPARE_PACKAGE)
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
                -DCMAKE_C_FLAGS:PATH=${CMAKE_C_FLAGS}
                -DCMAKE_CXX_FLAGS:PATH=${CMAKE_CXX_FLAGS}
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
                -DCMAKE_C_FLAGS:PATH=${CMAKE_C_FLAGS}
                -DCMAKE_CXX_FLAGS:PATH=${CMAKE_CXX_FLAGS}
            INSTALL_COMMAND ""
    )
    endif()
endif()

# Install
install (TARGETS ${PROJECT_NAME} DESTINATION bin)
if (ISPC_PREPARE_PACKAGE)
    if (ISPC_PACKAGE_EXAMPLES)
        if (XE_ENABLED)
            install (DIRECTORY "${PROJECT_SOURCE_DIR}/examples/" DESTINATION examples)
        else()
            install (DIRECTORY "${PROJECT_SOURCE_DIR}/examples/" DESTINATION examples PATTERN "xpu" EXCLUDE)
        endif()
    endif()
    install (DIRECTORY "${PROJECT_SOURCE_DIR}/contrib/" DESTINATION contrib)
    install (FILES "${PROJECT_SOURCE_DIR}/LICENSE.txt" DESTINATION .)
    install (FILES "${PROJECT_SOURCE_DIR}/third-party-programs.txt" DESTINATION .)
    install (FILES "${PROJECT_SOURCE_DIR}/docs/ReleaseNotes.txt" DESTINATION .)

    # Clone corresponding version of documentation
    include(ExternalProject)
    ExternalProject_Add(ispc_web
      GIT_REPOSITORY    https://github.com/ispc/ispc.github.com.git
      GIT_TAG ${ISPC_DOC_REPO_TAG}
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
        set(CPACK_GENERATOR "ZIP")
        set(ISPC_SYSTEM_NAME "windows")
        # WIX generator is currently turned off.
        # To turn it on uncomment code below:
        # set(CPACK_GENERATOR "WIX" "ZIP")
        # set(CPACK_WIX_UPGRADE_GUID "EDD858F2-19B0-4E5B-B9B9-2A09B85D451C")
        # set(CPACK_WIX_CMAKE_PACKAGE_REGISTRY "ISPC")
        # WIX generator expects version in format x.y.z where x, y, z are numbers
        set(CPACK_PACKAGE_VERSION "${ISPC_VERSION_MAJOR}.${ISPC_VERSION_MINOR}.${ISPC_VERSION_PATCH}")
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

    # Sign files in the package if ISPC_SIGN_KEY is set
    if (DEFINED ISPC_SIGN_KEY)
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Signing.cmake.in"
                       "${CMAKE_CURRENT_BINARY_DIR}/Signing.cmake" @ONLY)
        set(CPACK_PRE_BUILD_SCRIPTS "${CMAKE_CURRENT_BINARY_DIR}/Signing.cmake")
    endif()
    # Allow use custom package name with -DISPC_PACKAGE_NAME
    if (NOT ISPC_PACKAGE_NAME)
        set(ISPC_ARCH_SUFFIX)
        if (APPLE)
            if (CMAKE_OSX_ARCHITECTURES MATCHES "x86_64" AND CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
                set(ISPC_ARCH_SUFFIX ".universal")
            elseif(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64" OR
                   (CMAKE_OSX_ARCHITECTURES STREQUAL "" AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64"))
                set(ISPC_ARCH_SUFFIX ".x86_64")
            elseif(CMAKE_OSX_ARCHITECTURES MATCHES "arm64" OR
                   (CMAKE_OSX_ARCHITECTURES STREQUAL "" AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64"))
                set(ISPC_ARCH_SUFFIX ".arm64")
            else()
                message (FATAL_ERROR "Was not able to detect package architecture")
            endif()
        elseif (UNIX)
            # Add .aarch64 suffix if the package is built on AARCH64 system.
	    # On Linux we don't build cross packages, so host corresponds to target platform.
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
                set(ISPC_ARCH_SUFFIX ".aarch64")
            endif()
        endif()

        if (${CPACK_PACKAGE_VERSION_PATCH} MATCHES ".*dev")
            string(CONCAT CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}"
                                "-trunk"
                                "-${ISPC_SYSTEM_NAME}"
                                "${ISPC_ARCH_SUFFIX}")
            string(CONCAT EXAMPLES_PACKAGE_FILE_NAME "ispc-examples-trunk")
        else()
            string(CONCAT CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}"
                                "-v${CPACK_PACKAGE_VERSION_MAJOR}"
                                ".${CPACK_PACKAGE_VERSION_MINOR}"
                                ".${CPACK_PACKAGE_VERSION_PATCH}"
                                "-${ISPC_SYSTEM_NAME}"
                                "${ISPC_ARCH_SUFFIX}")
            string(CONCAT EXAMPLES_PACKAGE_FILE_NAME "ispc-examples"
                                "-v${CPACK_PACKAGE_VERSION_MAJOR}"
                                ".${CPACK_PACKAGE_VERSION_MINOR}"
                                ".${CPACK_PACKAGE_VERSION_PATCH}")
        endif()
    else()
        set (CPACK_PACKAGE_FILE_NAME ${ISPC_PACKAGE_NAME})
    endif()

    if (WIN32)
        set(CPACK_PACKAGE_INSTALL_DIRECTORY "ISPC/${CPACK_PACKAGE_FILE_NAME}")
    endif()
    set(CPACK_PACKAGE_VENDOR "Intel Corporation")

    # Strip binaries
    set(CPACK_STRIP_FILES TRUE)

    include(CPack)

    # Custom target to pack only examples
    if (WIN32)
        add_custom_target(package-examples
            COMMENT "Package examples directory to ${EXAMPLES_PACKAGE_FILE_NAME}.zip"
            COMMAND ${CMAKE_COMMAND} -E tar cf "${CMAKE_BINARY_DIR}/${EXAMPLES_PACKAGE_FILE_NAME}.zip" --format=zip examples
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            )
    else()
        add_custom_target(package-examples
            COMMENT "Package examples directory to ${EXAMPLES_PACKAGE_FILE_NAME}.tar.gz"
            COMMAND ${CMAKE_COMMAND} -E tar cfz "${CMAKE_BINARY_DIR}/${EXAMPLES_PACKAGE_FILE_NAME}.tar.gz" examples
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            )
    endif()

endif()
