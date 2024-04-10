#
#  Copyright (c) 2023-2024, Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.23)

project(SUPERBUILD NONE)

include(ExternalProject)
include(FetchContent)

# Checks that VAR_NAME has been declared. HELP_MESSAGE is just information.
# There are two ways of declaring options. In command line with -DVAR_NAME=value,
# or in CMakePresets.json creating new preset or changing a existing one.
# Note: command line -D values should override values from CMakePresets.json.
function(check_cache_var_declaration VAR_NAME HELP_MESSAGE)
    if (NOT ${VAR_NAME})
        message(FATAL_ERROR
            "${VAR_NAME} is not defined. You have to set that with -D${VAR_NAME} or use --preset option.")
    endif()
endfunction()

check_cache_var_declaration(LLVM_VERSION "LLVM version to build with")
check_cache_var_declaration(LLVM_URL "LLVM url")
check_cache_var_declaration(VC_INTRINSICS_URL "vc-intrinsics url")
check_cache_var_declaration(VC_INTRINSICS_SHA "vc-intrinsics hash commit to checkout")
check_cache_var_declaration(SPIRV_TRANSLATOR_URL "SPIRV-LLVM-Translator url")
check_cache_var_declaration(SPIRV_TRANSLATOR_SHA "SPIRV-LLVM-Translator hash commit to checkout")
check_cache_var_declaration(L0_URL "Level Zero url")
check_cache_var_declaration(L0_TAG "Level Zero url")
check_cache_var_declaration(ISPC_CORPUS_URL "ispc-corpus url")

set(GNUWIN32 "c:/gnuwin32" CACHE STRING "Set path to gnuwin32 root dir (Windows only)")
set(CCACHE_VERSION "4.8.2" CACHE STRING "ccache version ")
set(CCACHE_DIR "default" CACHE STRING "ccache cache directory")
set(PREBUILT_STAGE1 "NO" CACHE STRING "Provide pre-built stage1 archive to skip stage1 jobs")
set(PREBUILT_STAGE1_PATH "NO" CACHE STRING "Provide pre-built stage1 toolchain path")
set(PREBUILT_STAGE2 "NO" CACHE STRING "Provide pre-built stage2 archive to skip pre ispc-stage2 jobs")
set(PREBUILT_STAGE2_PATH "NO" CACHE STRING "Provide pre-built stage2 toolchain path")
set(PREBUILT_XE_STAGE2_PATH "NO" CACHE STRING "Provide path to pre-built dependencies for ISPC stage2")
set(BUILD_TYPE_ISPC "Release" CACHE STRING "Set the build type for ISPC build")
set(BUILD_TYPE_LLVM "Release" CACHE STRING "Set the build type for LLVM build")
set(BUILD_TYPE_SPIRV_TRANSLATOR "Release" CACHE STRING "Set the build type for SPIR-V Translator build")
set(BUILD_TYPE_VC_INTRINSICS "Release" CACHE STRING "Set the build type for VC Intrinsics build")
set(BUILD_TYPE_L0 "Release" CACHE STRING "Set the build type for L0 build")
set(ISPC_ANDROID_NDK_PATH "NO" CACHE STRING "Path to Android NDK. It is needed to ISPC_CROSS=ON on macOS.")
set(ISPC_SIGN_KEY "NO" CACHE STRING "Key to sign macOS or Windows build.")
option(ISPC_INCLUDE_BENCHMARKS "Include ISPC benchmarks (ispc_benchmarks target)" OFF)
option(ISPC_CROSS "Enable ISPC build with cross compilation support" ON)
option(ISPC_PACKAGE_EXAMPLES "Package examples into the same archive with ISPC binaries" OFF)
option(LLVM_DISABLE_ASSERTIONS "Enable LLVM build without assertions" ON)
option(LLVM_BUILD_OPENMP "Build OpenMP as part of LLVM" OFF)
option(EXTERNAL_VERBOSE "Enable verbose for external project build and install commands" OFF)
option(CCACHE "Enable ccache to speed up repetitive builds" OFF)
option(EXPLICIT_ENV_PATH "Enable explicit ENV PATH setting for external projects" ON)
option(FULL_STAGE2_LLVM "Enable full llvm stage2 build" OFF)
option(LTO "Enable LTO" OFF)
option(PGO "Enale PGO (implicitly enable LTO)" OFF)
option(BUILD_STAGE1_TOOLCHAIN_ONLY "Enable build of stage1 toolchain only." OFF)
option(BUILD_STAGE2_TOOLCHAIN_ONLY "Enable build of stage2 toolchain only, i.e., skip ispc building and stage3." OFF)
option(BUILD_SPIRV_TRANSLATOR_ONLY "Enable only build of SPIRV-LLVM-Translator" OFF)
option(BUILD_VC_INTRINSICS_ONLY "Enable only build of vc-intrinsics" OFF)
option(BUILD_L0_LOADER_ONLY "Enable only build of level-zero loader" OFF)
option(BUILD_XE_DEPS_ONLY "Enable only build of all XE dependencies (L0 loader, vc-intrinsics, SPIRV-LLVM-Translator" OFF)
option(INSTALL_TOOLCHAIN "Enable installation of toolchain to CMAKE_INSTALL_PREFIX" OFF)
option(INSTALL_ISPC "Enable installation of ISPC to CMAKE_INSTALL_PREFIX" OFF)
option(INSTALL_WITH_XE_DEPS "Enable installation ISPC or stage2 toolchain with XE_DEPS to CMAKE_INSTALL_PREFIX" OFF)
option(XE_DEPS "Enable build of all XE dependencies (L0 loader, vc-intrinsics, SPIRV-LLVM-Translator" ON)
option(MACOS_UNIVERSAL_BIN "Enable build of universal binaries for macOS" ON)

# Force LTO when PGO enabled to the sake of simplicity and less number of
# possible build variants.
if (PGO)
    set(LTO ON)
endif()

set(SKIP_STAGE1 OFF)
if (NOT PREBUILT_STAGE1 STREQUAL "NO" OR NOT PREBUILT_STAGE1_PATH STREQUAL "NO")
    set(SKIP_STAGE1 ON)
    set(FULL_STAGE2_LLVM ON)
endif()

if (NOT PREBUILT_STAGE1 STREQUAL "NO" AND NOT PREBUILT_STAGE1_PATH STREQUAL "NO")
    message(FATAL_ERROR "Provide only one source of stage1 toolchain.")
endif()

set(SKIP_STAGE2 OFF)
if (NOT PREBUILT_STAGE2 STREQUAL "NO" OR NOT PREBUILT_STAGE2_PATH STREQUAL "NO")
    set(SKIP_STAGE2 ON)
endif()

if (NOT PREBUILT_STAGE2 STREQUAL "NO" AND NOT PREBUILT_STAGE2_PATH STREQUAL "NO")
    message(FATAL_ERROR "Provide only one source of stage2 toolchain.")
endif()

set(SKIP_FETCHING_XE_DEPS OFF)
if (NOT PREBUILT_XE_STAGE2_PATH STREQUAL "NO")
    set(SKIP_FETCHING_XE_DEPS ON)
    if (NOT SKIP_STAGE2)
        message(FATAL_ERROR "Provide PREBUILT_XE_STAGE2_PATH only when stage2 is skipped")
    endif()
    if (INSTALL_WITH_XE_DEPS)
        message(FATAL_ERROR "INSTALL_WITH_XE_DEPS requires to rebuild XE deps")
    endif()
endif()

set(STAGE2_PATH "${CMAKE_BINARY_DIR}/stage2")
if (NOT PREBUILT_STAGE1_PATH STREQUAL "NO")
    set(STAGE2_PATH "${PREBUILT_STAGE1_PATH}")
endif()
if (NOT PREBUILT_STAGE2_PATH STREQUAL "NO")
    set(STAGE2_PATH "${PREBUILT_STAGE2_PATH}")
endif()
set(STAGE3_PATH "${CMAKE_BINARY_DIR}/stage3")

if (NOT PREBUILT_STAGE1_PATH STREQUAL "NO" AND NOT PREBUILT_STAGE2_PATH STREQUAL "NO")
    message(FATAL_ERROR "PREBUILT_STAGE1_PATH and PREBUILT_STAGE2_PATH is incompatible together.")
endif()

# It may be slightly confusing but stage1 and stage2 effectively are installed
# into same directory.
set(TOOLCHAIN_STAGE1_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/stage2)
if (BUILD_STAGE1_TOOLCHAIN_ONLY)
    set(TOOLCHAIN_STAGE1_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()
set(TOOLCHAIN_STAGE2_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/stage2)
if (BUILD_STAGE2_TOOLCHAIN_ONLY OR INSTALL_TOOLCHAIN)
    set(STAGE2_PATH "${CMAKE_INSTALL_PREFIX}")
    set(TOOLCHAIN_STAGE1_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
    set(TOOLCHAIN_STAGE2_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()
message(STATUS "TOOLCHAIN_STAGE1_INSTALL_PREFIX is ${TOOLCHAIN_STAGE1_INSTALL_PREFIX}")
message(STATUS "TOOLCHAIN_STAGE2_INSTALL_PREFIX is ${TOOLCHAIN_STAGE2_INSTALL_PREFIX}")

set(ISPC_STAGE2_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/ispc-stage2)
set(ISPC_STAGE3_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/ispc-stage3)
if (INSTALL_WITH_XE_DEPS OR INSTALL_ISPC)
    set(ISPC_STAGE2_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
    set(ISPC_STAGE3_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()
message(STATUS "ISPC_STAGE2_INSTALL_PREFIX is ${ISPC_STAGE2_INSTALL_PREFIX}")
message(STATUS "ISPC_STAGE3_INSTALL_PREFIX is ${ISPC_STAGE3_INSTALL_PREFIX}")

if (BUILD_STAGE1_TOOLCHAIN_ONLY OR
    BUILD_SPIRV_TRANSLATOR_ONLY OR BUILD_VC_INTRINSICS_ONLY OR BUILD_L0_LOADER_ONLY)
    set(XE_DEPS OFF)
endif()

set(BUILD_SPIRV_STANDALONE OFF)
set(BUILD_VC_INTRINSICS_STANDALONE OFF)
set(BUILD_L0_LOADER_STANDALONE OFF)
if (INSTALL_WITH_XE_DEPS OR BUILD_XE_DEPS_ONLY)
    set(BUILD_SPIRV_STANDALONE ON)
    set(BUILD_VC_INTRINSICS_STANDALONE ON)
    set(BUILD_L0_LOADER_STANDALONE ON)
endif()
if (BUILD_SPIRV_TRANSLATOR_ONLY)
    set(BUILD_SPIRV_STANDALONE ON)
endif()
if (BUILD_VC_INTRINSICS_ONLY)
    set(BUILD_VC_INTRINSICS_STANDALONE ON)
endif()
if (BUILD_L0_LOADER_ONLY)
    set(BUILD_L0_LOADER_STANDALONE ON)
endif()

if (APPLE)
    # XE are not supported on macOS targets
    set(XE_DEPS OFF)

    set(FULL_STAGE2_LLVM ON)

    set(MACOS_VERSION_MIN 10.12)
    if (NOT MACOS_UNIVERSAL_BIN AND CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        set(MACOS_VERSION_MIN 11.0)
    endif()

    if (APPLE AND ISPC_ANDROID_NDK_PATH STREQUAL "NO")
        message(FATAL_ERROR "You need provide path to ANDROID NDK on macOS via -DISPC_ANDROID_NDK_PATH=..")
    endif()

    if (MACOS_UNIVERSAL_BIN AND CMAKE_OSX_ARCHITECTURES)
        message(WARNING "CMAKE_OSX_ARCHITECTURES value ${CMAKE_OSX_ARCHITECTURES} is "
                        "discarded due to MACOS_UNIVERSAL_BIN is ON")
    endif()
endif()


string(APPEND LIT_TOOLS_DIR ${GNUWIN32} "/bin")

unset(LLVM_TAG)
unset(LLVM_VERSION_DOTTED)
string(REPLACE "." "_" LLVM_VERSION "${LLVM_VERSION}")
if (${LLVM_VERSION} STREQUAL 18_1)
    set(LLVM_TAG    llvmorg-18.1.2)
elseif (${LLVM_VERSION} STREQUAL 17_0)
    set(LLVM_TAG    llvmorg-17.0.6)
elseif (${LLVM_VERSION} STREQUAL 16_0)
    set(LLVM_TAG    llvmorg-16.0.6)
elseif (${LLVM_VERSION} STREQUAL 15_0)
    set(LLVM_TAG    llvmorg-15.0.7)
elseif (${LLVM_VERSION} STREQUAL 14_0)
    set(LLVM_TAG    llvmorg-14.0.6)
elseif (${LLVM_VERSION} STREQUAL 13_0)
    set(LLVM_TAG    llvmorg-13.0.1)
elseif (${LLVM_VERSION} STREQUAL 12_0)
    set(LLVM_TAG    llvmorg-12.0.1)
elseif (${LLVM_VERSION} STREQUAL 11_1)
    set(LLVM_TAG    llvmorg-11.1.0)
elseif (${LLVM_VERSION} STREQUAL 11_0)
    set(LLVM_TAG    llvmorg-11.0.1)
elseif (${LLVM_VERSION} STREQUAL 10_0)
    set(LLVM_TAG    llvmorg-10.0.1)
elseif (${LLVM_VERSION} STREQUAL 9_0)
    set(LLVM_TAG    llvmorg-9.0.1)
elseif (${LLVM_VERSION} STREQUAL 8_0)
    set(LLVM_TAG    llvmorg-8.0.1)
elseif (${LLVM_VERSION} STREQUAL 7_1)
    set(LLVM_TAG    llvmorg-7.1.0)
elseif (${LLVM_VERSION} STREQUAL 7_0)
    set(LLVM_TAG    llvmorg-7.0.1)
elseif (${LLVM_VERSION} STREQUAL 6_0)
    set(LLVM_TAG    llvmorg-6.0.1)
elseif (${LLVM_VERSION} STREQUAL trunk)
    set(LLVM_TAG main)
else()
    message(FATAL_ERROR "Incorrect LLVM version")
endif()

if (${LLVM_VERSION} STREQUAL trunk)
    set(LLVM_VERSION_DOTTED main)
    set(LLVM_VERSION_MAJOR 0)
    set(LLVM_VERSION_MINOR 0)
    set(LLVM_VERSION_PATCH 0)
else()
    string(SUBSTRING ${LLVM_TAG} 8 -1 LLVM_VERSION_DOTTED)
    string(REGEX MATCH "([0-9]*)\.([0-9]*)\.([0-9]*)" _ ${LLVM_VERSION_DOTTED})
    set(LLVM_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(LLVM_VERSION_MINOR ${CMAKE_MATCH_2})
    set(LLVM_VERSION_PATCH ${CMAKE_MATCH_3})
endif()

set(EXTERNAL_VERBOSE_FLAG "")
if (EXTERNAL_VERBOSE)
    set(EXTERNAL_VERBOSE_FLAG "-v")
endif()

if (CMAKE_BUILD_PARALLEL_LEVEL)
    list(APPEND NINJA_JOBS -j ${CMAKE_BUILD_PARALLEL_LEVEL})
else()
    list(APPEND NINJA_JOBS "")
endif()

if (CMAKE_BUILD_TYPE)
    set(BUILD_TYPE_ISPC ${CMAKE_BUILD_TYPE})
    set(BUILD_TYPE_LLVM ${CMAKE_BUILD_TYPE})
    set(BUILD_TYPE_SPIRV_TRANSLATOR ${CMAKE_BUILD_TYPE})
    set(BUILD_TYPE_VC_INTRINSICS ${CMAKE_BUILD_TYPE})
    set(BUILD_TYPE_L0 ${CMAKE_BUILD_TYPE})
endif()

message(STATUS "LLVM_VERSION is ${LLVM_VERSION}")
message(STATUS "LLVM_URL is ${LLVM_URL}")
message(STATUS "VC_INTRINSICS_URL is ${VC_INTRINSICS_URL}")
message(STATUS "VC_INTRINSICS_SHA is ${VC_INTRINSICS_SHA}")
message(STATUS "SPIRV_TRANSLATOR_URL is ${SPIRV_TRANSLATOR_URL}")
message(STATUS "SPIRV_TRANSLATOR_SHA is ${SPIRV_TRANSLATOR_SHA}")
message(STATUS "L0_URL is ${L0_URL}")
message(STATUS "L0_TAG is ${L0_TAG}")
message(STATUS "GNUWIN32 is ${GNUWIN32}")
message(STATUS "LIT_TOOLS_DIR is ${LIT_TOOLS_DIR}")
message(STATUS "ISPC_CORPUS_URL is ${ISPC_CORPUS_URL}")
message(STATUS "CCACHE_VERSION is ${CCACHE_VERSION}")
message(STATUS "CCACHE_DIR is ${CCACHE_DIR}")
message(STATUS "CCACHE is ${CCACHE}")
message(STATUS "PREBUILT_STAGE1 is ${PREBUILT_STAGE1}")
message(STATUS "PREBUILT_STAGE1_PATH is ${PREBUILT_STAGE1_PATH}")
message(STATUS "PREBUILT_STAGE2 is ${PREBUILT_STAGE2}")
message(STATUS "PREBUILT_STAGE2_PATH is ${PREBUILT_STAGE2_PATH}")
message(STATUS "PREBUILT_XE_STAGE2_PATH is ${PREBUILT_XE_STAGE2_PATH}")
message(STATUS "EXTERNAL_VERBOSE is ${EXTERNAL_VERBOSE}")
message(STATUS "EXPLICIT_ENV_PATH is ${EXPLICIT_ENV_PATH}")
message(STATUS "FULL_STAGE2_LLVM is ${FULL_STAGE2_LLVM}")
message(STATUS "ISPC_ANDROID_NDK_PATH is ${ISPC_ANDROID_NDK_PATH}")
if (ISPC_SIGN_KEY STREQUAL "NO")
    message(STATUS "ISPC_SIGN_KEY is not set.")
else()
    message(STATUS "ISPC_SIGN_KEY is set to some value.")
endif()
message(STATUS "ISPC_INCLUDE_BENCHMARKS is ${ISPC_INCLUDE_BENCHMARKS}")
message(STATUS "ISPC_CROSS is ${ISPC_CROSS}")
message(STATUS "ISPC_PACKAGE_EXAMPLES is ${ISPC_PACKAGE_EXAMPLES}")
message(STATUS "LLVM_DISABLE_ASSERTIONS is ${LLVM_DISABLE_ASSERTIONS}")
message(STATUS "LLVM_BUILD_OPENMP is ${LLVM_BUILD_OPENMP}")
message(STATUS "LTO is ${LTO}")
message(STATUS "PGO is ${PGO}")
message(STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_BUILD_PARALLEL_LEVEL is ${CMAKE_BUILD_PARALLEL_LEVEL}")
message(STATUS "BUILD_TYPE_ISPC is ${BUILD_TYPE_ISPC}")
message(STATUS "BUILD_TYPE_LLVM is ${BUILD_TYPE_LLVM}")
message(STATUS "BUILD_TYPE_SPIRV_TRANSLATOR is ${BUILD_TYPE_SPIRV_TRANSLATOR}")
message(STATUS "BUILD_TYPE_VC_INTRINSICS is ${BUILD_TYPE_VC_INTRINSICS}")
message(STATUS "BUILD_TYPE_L0 is ${BUILD_TYPE_L0}")
message(STATUS "BUILD_STAGE2_TOOLCHAIN_ONLY is ${BUILD_STAGE2_TOOLCHAIN_ONLY}")
message(STATUS "BUILD_SPIRV_TRANSLATOR_ONLY is ${BUILD_SPIRV_TRANSLATOR_ONLY}")
message(STATUS "BUILD_VC_INTRINSICS_ONLY is ${BUILD_VC_INTRINSICS_ONLY}")
message(STATUS "BUILD_L0_LOADER_ONLY is ${BUILD_L0_LOADER_ONLY}")
message(STATUS "BUILD_SPIRV_STANDALONE is ${BUILD_SPIRV_STANDALONE}")
message(STATUS "BUILD_VC_INTRINSICS_STANDALONE is ${BUILD_VC_INTRINSICS_STANDALONE}")
message(STATUS "BUILD_L0_LOADER_STANDALONE is ${BUILD_L0_LOADER_STANDALONE}")
message(STATUS "BUILD_XE_DEPS_ONLY is ${BUILD_XE_DEPS_ONLY}")
message(STATUS "INSTALL_TOOLCHAIN is ${INSTALL_TOOLCHAIN}")
message(STATUS "INSTALL_ISPC is ${INSTALL_ISPC}")
message(STATUS "INSTALL_WITH_XE_DEPS is ${INSTALL_WITH_XE_DEPS}")
message(STATUS "XE_DEPS is ${XE_DEPS}")
message(STATUS "SKIP_FETCHING_XE_DEPS is ${SKIP_FETCHING_XE_DEPS}")
message(STATUS "LLVM_TAG is ${LLVM_TAG}")
message(STATUS "LLVM_VERSION_DOTTED is ${LLVM_VERSION_DOTTED}")
message(STATUS "LLVM_VERSION_MAJOR is ${LLVM_VERSION_MAJOR}")
message(STATUS "LLVM_VERSION_MINOR is ${LLVM_VERSION_MINOR}")
message(STATUS "LLVM_VERSION_PATCH is ${LLVM_VERSION_PATCH}")
message(STATUS "SKIP_STAGE1 is ${SKIP_STAGE1}")
message(STATUS "SKIP_STAGE2 is ${SKIP_STAGE2}")
message(STATUS "STAGE2_PATH is ${STAGE2_PATH}")
message(STATUS "CMAKE_INSTALL_PREFIX is ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_C_FLAGS is ${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS is ${CMAKE_CXX_FLAGS}")
message(STATUS "MACOS_VERSION_MIN is ${MACOS_VERSION_MIN}")
message(STATUS "MACOS_UNIVERSAL_BIN is ${MACOS_UNIVERSAL_BIN}")
message(STATUS "CMAKE_OSX_ARCHITECTURES is ${CMAKE_OSX_ARCHITECTURES}")


if (NOT XE_DEPS)
    # To suppress CMake warnings about unused user-provided variables.
    set(ignoreMe "${VC_INTRINSICS_URL}${VC_INTRINSICS_SHA}")
    set(ignoreMe "${SPIRV_TRANSLATOR_URL}${SPIRV_TRANSLATOR_SHA}${SPIRV_TRANSLATOR_BRANCH}")
    set(ignoreMe "${L0_URL}${L0_TAG}")
endif()


# Ninja is hard-coded for two reasons:
# 1. It looks like that only ninja is able out-of-box properly utilize all CPUs.
# 2. Some LLVM stage2 install targets are ninja specific, e.g., genx, spirv, openmp.
find_program(NINJA_EXE NAMES ninja)
if (NOT NINJA_EXE)
    message(FATAL_ERROR "ninja not found.")
endif()


# Get list of patches needed to apply for the requested LLVM version
cmake_path(SET LLVM_PATCHES_DIR NORMALIZE ${PROJECT_SOURCE_DIR}/../llvm_patches)
file(GLOB_RECURSE LLVM_PATCHES_ALL ${LLVM_PATCHES_DIR}/*.patch)
foreach(patch IN LISTS LLVM_PATCHES_ALL)
    if (${patch} MATCHES ".*${LLVM_VERSION}.*")
        list(APPEND LLVM_PATCHES_VER ${patch})
    endif()
endforeach()

if (NOT APPLE)
    list(APPEND LLVM_PATCHES_VER ${PROJECT_SOURCE_DIR}/ld-symlink-lld.patch)
endif()
message(STATUS "For llvm version ${LLVM_VERSION} found patches ${LLVM_PATCHES_VER}")


# Set up ccache use to speed up build process.
# We always include CCACHE_EXE and CCACHE_LAUNCHER_ACTION into *-toolchain.cmake files.
# To disable CCACHE we define default values.
set(CCACHE_LAUNCHER_ACTION unset)

# Comment out cache_dir from ccache.conf when CCACHE_DIR not overridden by user.
set(CCACHE_COMMENT_CACHE_DIR "")
if (${CCACHE_DIR} STREQUAL default)
    set(CCACHE_COMMENT_CACHE_DIR "# ")
endif()

# Default is to check if compiler if up-to-date by size and mtime.
set(CCACHE_DEFAULT_CONF ${CMAKE_BINARY_DIR}/ccache.conf)
# Stage1 clang is needed to check better to hit cache.
set(CCACHE_CLANG_V_CONF ${CMAKE_BINARY_DIR}/ccache-clang.conf)

if (CCACHE)
    # Not default values with set and actual ccache binary full path.
    set(CCACHE_LAUNCHER_ACTION set)
    # cmake_path(CONVERT "${CMAKE_BINARY_DIR}" TO_NATIVE_PATH_LIST CCACHE_BASE_DIR_NATIVE)

    # Configuration file.
    # Same cache-dir for any job on machine to share cache between different compilations.
    # Sloppiness to avoid some time/locale specific setting. Probably not important much.
    # base_dir needed to share cache between compilation in different directories.
    file(CONFIGURE OUTPUT ccache.conf CONTENT [[
${CCACHE_COMMENT_CACHE_DIR}cache_dir = ${CCACHE_DIR}
sloppiness = locale,time_macros
compiler_check = content
base_dir = ${CMAKE_BINARY_DIR}
hash_dir = false
]])

    # compiler_check -v output is needed to enable caching on freshly built
    # compiler, i.e., during stage2, otherwise it would be skipped due to mtime
    # is newer than in the cached command.
    # Clang has been patched to not contain in -v absolute paths of directories
    # which would result in caches misses when build directory changed.
    file(CONFIGURE OUTPUT ccache-clang.conf CONTENT [[
${CCACHE_COMMENT_CACHE_DIR}cache_dir = ${CCACHE_DIR}
sloppiness = locale,time_macros
# compiler_check = %compiler% -v
base_dir = ${CMAKE_BINARY_DIR}
hash_dir = false
]])

    if (WIN32)
        string(APPEND CCACHE_URL
            https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-windows-x86_64.zip)
    elseif (APPLE)
        string(APPEND CCACHE_URL
            https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-darwin.tar.gz)
    else()
        string(APPEND CCACHE_URL
            https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64.tar.xz)
    endif()

    find_program(CCACHE_EXE NAMES ccache)
    if (NOT CCACHE_EXE)
        # Download and unpack ccache executable when no ccache in system.
        FetchContent_Populate(ccache
            QUIET
            URL ${CCACHE_URL}
            SOURCE_DIR ccache
            )

        set(CCACHE_EXE "${CMAKE_BINARY_DIR}/ccache/ccache${CMAKE_EXECUTABLE_SUFFIX}")
    endif()
    message(STATUS "CCACHE_EXE is ${CCACHE_EXE}")
endif()


if (APPLE)
    message(STATUS "APPLE CMAKE_SYSTEM_VERSION is ${CMAKE_SYSTEM_VERSION}")
    # Starting from MacOS 10.9 Maverics which is Darwin 13.0, C and C++ library
    # headers are part of the SDK, not the OS itself. System root must be
    # specified during the compiler build, so the compiler knows the default
    # location to search for headers. C headers are located at system root
    # location, while C++ headers are part of the toolchain. I.e. specifying
    # system root solved C header problem. For C++ headers we enable libc++
    # build as part of clang build (our own toolchain).
    # Note that on Sierra there's an issue with using C headers from High
    # Sierra SDK, which instantiates as compile error:
    #     error: 'utimensat' is only available on macOS 10.13 or newer
    # This is due to using SDK targeting OS, which is newer than current one.
    if (CMAKE_SYSTEM_VERSION VERSION_GREATER_EQUAL 13)
        if (NOT MAC_SYSTEM_ROOT)
            find_program(XCRUN_EXE NAMES xcrun)
            if (NOT XCRUN_EXE)
                message(FATAL_ERROR "xcrun is not found")
            endif()
            execute_process(COMMAND ${XCRUN_EXE} --show-sdk-path
                OUTPUT_VARIABLE MAC_SYSTEM_ROOT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        endif()
        message(STATUS "MAC_SYSTEM_ROOT (macOS SDK path) is ${MAC_SYSTEM_ROOT}")
    endif()
endif()


# common LLVM cmake configuration flags.
list(APPEND LLVM_COMMON_CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE_LLVM}
    -DLLVM_ENABLE_DUMP=OFF
    -DLLVM_INSTALL_UTILS=ON
    -DLLVM_ENABLE_ZLIB=OFF
    -DLLVM_ENABLE_ZSTD=OFF
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    )

# Disable unneded parts of clang.
list(APPEND LLVM_COMMON_CMAKE_ARGS
    -DCLANG_ENABLE_ARCMT=OFF
    -DCLANG_ENABLE_STATIC_ANALYZER=OFF
    -DCLANG_INCLUDE_TESTS=OFF
    )

# Disable extra component in compiler_rt, we only need profile_rt and sanitizers.
list(APPEND LLVM_COMMON_CMAKE_ARGS
    -DCOMPILER_RT_BUILD_PROFILE=ON
    -DCOMPILER_RT_BUILD_SANITIZERS=ON
    -DCOMPILER_RT_BUILD_CRT=OFF
    -DCOMPILER_RT_CRT_USE_EH_FRAME_REGISTRY=OFF
    -DCOMPILER_RT_BUILD_XRAY=OFF
    -DCOMPILER_RT_BUILD_LIBFUZZER=OFF
    -DCOMPILER_RT_BUILD_MEMPROF=OFF
    -DCOMPILER_RT_BUILD_ORC=OFF
    -DCOMPILER_RT_BUILD_GWP_ASAN=OFF
    )

if (APPLE)
    # On macOS, we need builtins also.
    list(APPEND LLVM_COMMON_CMAKE_ARGS
        -DCOMPILER_RT_BUILD_BUILTINS=ON
        -DCOMPILER_RT_ENABLE_IOS=OFF
        )

    # LLVM has lib/Support/BLAKE3 component that consists of C and ASM
    # (optionally) code. ASM code is x86 specific. They are linked together
    # into static library. Under LTO for macOS universal binary scenario,
    # libtool produces an error that it is unable to mix object code with
    # bitcode. LLVM toolchain can't produce bitcode for asm. So, we disable the
    # usage of assembly files that can cause performance penalty. At the
    # moment, LLVM_DISABLE_ASSEMBLY_FILES is used only in BLAKE3.
    if (LTO)
        list(APPEND LLVM_COMMON_CMAKE_ARGS
            -DLLVM_DISABLE_ASSEMBLY_FILES=ON
            )
    endif()
else()
    list(APPEND LLVM_COMMON_CMAKE_ARGS
        -DCOMPILER_RT_BUILD_BUILTINS=OFF
        )
endif()

# Enable or disable assertions.
if (LLVM_DISABLE_ASSERTIONS)
    list(APPEND LLVM_COMMON_CMAKE_ARGS
        -DLLVM_ENABLE_ASSERTIONS=OFF
        )
else()
    list(APPEND LLVM_COMMON_CMAKE_ARGS
        -DLLVM_ENABLE_ASSERTIONS=ON
        )
endif()

# Targets to build.
# It is possible to build for stage1 only X86 if we build only on x86 platforms.
list(APPEND LLVM_COMMON_CMAKE_ARGS
    -DLLVM_TARGETS_TO_BUILD=X86$<SEMICOLON>AArch64$<SEMICOLON>ARM
    -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=WebAssembly
    )

# Enable LLVM subprojects.
set(LLVM_PROJECTS clang$<SEMICOLON>lld)
list(APPEND LLVM_COMMON_CMAKE_ARGS
    -DLLVM_ENABLE_PROJECTS=${LLVM_PROJECTS}
    )

# Install symlinks, nm -> llvm-nm is required for building ISPC.
# Unfortunately, this options doesn't force ld -> lld symlink creation, so
# patch for LLVM tree resolves that (13_0_14_0_15_0-ld-symlink-lld.patch).
list(APPEND LLVM_COMMON_CMAKE_ARGS
    -DLLVM_INSTALL_BINUTILS_SYMLINKS=ON
    )

if (WIN32)
    list(APPEND LLVM_COMMON_CMAKE_ARGS
        -DLLVM_LIT_TOOLS_DIR=${LIT_TOOLS_DIR}
        )
endif()
if (APPLE)
    list(APPEND LLVM_COMMON_CMAKE_ARGS
        -DLLVM_INSTALL_CCTOOLS_SYMLINKS=ON
        )
endif()

# Enable LLVM runtimes.
set(LLVM_RUNTIMES compiler-rt)
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND LLVM_BUILD_OPENMP)
    # OpenMP may be needed for Xe enabled builds.
    set(LLVM_RUNTIMES ${LLVM_RUNTIMES}$<SEMICOLON>openmp)
endif()

if (APPLE AND CMAKE_SYSTEM_VERSION VERSION_GREATER_EQUAL 13)
    list(APPEND LLVM_COMMON_CMAKE_ARGS
        -DDEFAULT_SYSROOT=${MAC_SYSTEM_ROOT}
        )
    # Starting with MacOS 10.9 Maverics, the system doesn't contain headers for
    # standard C++ library and the default library is libc++, bit libstdc++. The
    # headers are part of XCode now. But we are checking out headers as part of
    # LLVM source tree, so they will be installed in clang location and clang
    # will be able to find them. Though they may not match to the library
    # installed in the system, but seems that this should not happen.  Note, that
    # we can also build a libc++ library, but it must be on system default
    # location or should be passed to the linker explicitly (either through
    # command line or environment variables). So we are not doing it currently to
    # make the build process easier.

    # We either need to explicitly opt-out from using libcxxabi from this repo,
    # or build and use it, otherwise a build error will occure (attempt to use
    # just built libcxxabi, which was not built).  An option to build seems to be
    # a better one.
    set(LLVM_RUNTIMES
        ${LLVM_RUNTIMES}$<SEMICOLON>libcxx$<SEMICOLON>libcxxabi
        )
endif()

# LLVM_ENABLE_RUNTIMES is a proper place for compiler-rt and openmp not LLVM_ENABLE_PROJECTS.
list(APPEND LLVM_COMMON_CMAKE_ARGS
    -DLLVM_ENABLE_RUNTIMES=${LLVM_RUNTIMES}
    )

# stage1 specific LLVM cmake arguments.
list(APPEND LLVM_STAGE1_CMAKE_ARGS
    ${LLVM_COMMON_CMAKE_ARGS}
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_BINARY_DIR}/stage1-toolchain.cmake
    )
message(STATUS "LLVM stage1 cmake arguments ${LLVM_STAGE1_CMAKE_ARGS}")


# LLVM stage2 install targets.
if (FULL_STAGE2_LLVM)
    list(APPEND LLVM_STAGE2_INSTALL_TARGETS install)
else()
    list(APPEND LLVM_STAGE2_INSTALL_TARGETS
        install-clang-headers
        install-clangFrontend
        install-clangBasic
        install-clangEdit
        install-clangLex
        )
    list(APPEND LLVM_STAGE2_INSTALL_TARGETS
        install-llvm-headers
        install-llvm-libraries
        install-llvm-config
        )

    if (XE_DEPS)
        list(APPEND LLVM_STAGE2_INSTALL_TARGETS
            tools/vc-intrinsics/install
            tools/SPIRV-LLVM-Translator/install
            )
    endif()
endif()
message(STATUS "LLVM stage2 install targets ${LLVM_STAGE2_INSTALL_TARGETS}")

# stage2 cmake arguments common for both LTO and not LTO build.
# Build stage2 clang/LLVM libraries with XE deps as external LLVM projects
list(APPEND LLVM_STAGE2_COMMON_CMAKE_ARGS
    ${LLVM_COMMON_CMAKE_ARGS}
    )

if (XE_DEPS)
    List(APPEND LLVM_STAGE2_COMMON_CMAKE_ARGS
        -DLLVM_EXTERNAL_PROJECTS=vc-intrinsics$<SEMICOLON>SPIRV-LLVM-Translator
        -DLLVM_EXTERNAL_VC_INTRINSICS_SOURCE_DIR=${CMAKE_BINARY_DIR}/vc-intrinsics
        -DLLVM_EXTERNAL_SPIRV_LLVM_TRANSLATOR_SOURCE_DIR=${CMAKE_BINARY_DIR}/spirv-translator
        )
endif()

if (LTO)
    list(APPEND LLVM_STAGE2_CMAKE_ARGS
        ${LLVM_STAGE2_COMMON_CMAKE_ARGS}
        -DLLVM_ENABLE_LTO=Thin
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_BINARY_DIR}/stage2-lto-toolchain.cmake
        )
else()
    list(APPEND LLVM_STAGE2_CMAKE_ARGS
        ${LLVM_STAGE2_COMMON_CMAKE_ARGS}
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_BINARY_DIR}/stage2-toolchain.cmake
        )
endif()

if (APPLE)
    list(APPEND LLVM_STAGE2_CMAKE_ARGS
        -DLD64_EXECUTABLE=${STAGE2_PATH}/bin/ld64.lld
        )

    # macOS deployment target sets the minimim OS requirement to run the
    # compiled program. All of the object files and static libraries passed to
    # the linker need to be compiled for the same version of the macOS to
    # enable resulting executable to be able to run on the older macOS. So we
    # build LLVM for old macOS to enable ISPC builds targeting an old macOS.
    list(APPEND LLVM_STAGE2_CMAKE_ARGS
        -DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOS_VERSION_MIN}
        )

    if (MACOS_UNIVERSAL_BIN)
        # macOS Universal Binaries is a fat binary for x86_64 and arm64.
        list(APPEND LLVM_STAGE2_CMAKE_ARGS
            -DCMAKE_OSX_ARCHITECTURES=x86_64$<SEMICOLON>arm64
            )
    else()
        list(APPEND LLVM_STAGE2_CMAKE_ARGS
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            )
    endif()
endif()
message(STATUS "LLVM stage2 cmake arguments ${LLVM_STAGE2_CMAKE_ARGS}")


# ISPC cmake arguments.
list(APPEND ISPC_COMMON_CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE_ISPC}
    -DISPC_PREPARE_PACKAGE=ON
    -DISPC_PACKAGE_EXAMPLES=${ISPC_PACKAGE_EXAMPLES}
    )
if (ISPC_CROSS)
    list(APPEND ISPC_COMMON_CMAKE_ARGS
        -DISPC_CROSS=ON
        )
endif()
if (XE_DEPS)
    list(APPEND ISPC_COMMON_CMAKE_ARGS
        -DXE_ENABLED=ON
        )
endif()
if (ISPC_INCLUDE_BENCHMARKS)
    list(APPEND ISPC_COMMON_CMAKE_ARGS
        -DISPC_INCLUDE_BENCHMARKS=${ISPC_INCLUDE_BENCHMARKS}
        )
endif()
if (WIN32)
    # XPU examples are not buildable on windows, issue: TODO!
    list(APPEND ISPC_COMMON_CMAKE_ARGS
        -DISPC_GNUWIN32_PATH=${GNUWIN32}
        -DISPC_INCLUDE_EXAMPLES=OFF
        -DISPC_INCLUDE_XE_EXAMPLES=OFF
        )
endif()
if (WIN32 OR APPLE)
    if (ISPC_SIGN_KEY)
        list(APPEND ISPC_COMMON_CMAKE_ARGS
            -DISPC_SIGN_KEY=${ISPC_SIGN_KEY}
            )
    endif()
endif()
if (APPLE)
    # It is needed because of ISPC_CROSS=ON
    if (ISPC_CROSS)
        list(APPEND ISPC_COMMON_CMAKE_ARGS
            -DISPC_ANDROID_NDK_PATH=${ISPC_ANDROID_NDK_PATH}
            )
    endif()

    # For APPLE explicitly enabled both arch.
    list(APPEND ISPC_COMMON_CMAKE_ARGS
        -DX86_ENABLED=ON
        -DARM_ENABLED=ON
        )

    list(APPEND ISPC_COMMON_CMAKE_ARGS
        -DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOS_VERSION_MIN}
        )

    if(MACOS_UNIVERSAL_BIN)
        list(APPEND ISPC_COMMON_CMAKE_ARGS
            -DISPC_MACOS_UNIVERSAL_BINARIES=ON
            )
    else()
        list(APPEND ISPC_COMMON_CMAKE_ARGS
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            )
    endif()
endif()

list(APPEND ISPC_STAGE2_CMAKE_ARGS
    ${ISPC_COMMON_CMAKE_ARGS}
    )
if (XE_DEPS)
    if (INSTALL_WITH_XE_DEPS)
        list(APPEND ISPC_STAGE2_CMAKE_ARGS
            -DLEVEL_ZERO_ROOT=${CMAKE_INSTALL_PREFIX}
            )
    else()
        if (NOT PREBUILT_XE_STAGE2_PATH STREQUAL "NO")
            list(APPEND ISPC_STAGE2_CMAKE_ARGS
                -DLEVEL_ZERO_ROOT=${PREBUILT_XE_STAGE2_PATH}
                )
        else()
            list(APPEND ISPC_STAGE2_CMAKE_ARGS
                -DLEVEL_ZERO_ROOT=${STAGE2_PATH}
                )
        endif()
    endif()
endif()
if (XE_DEPS)
    if (INSTALL_WITH_XE_DEPS)
        list(APPEND ISPC_STAGE2_CMAKE_ARGS
            -DXE_DEPS_DIR=${CMAKE_INSTALL_PREFIX}
            )
    else()
        if (NOT PREBUILT_XE_STAGE2_PATH STREQUAL "NO")
            list(APPEND ISPC_STAGE2_CMAKE_ARGS
                -DXE_DEPS_DIR=${PREBUILT_XE_STAGE2_PATH}
                )
        else()
            list(APPEND ISPC_STAGE2_CMAKE_ARGS
                -DXE_DEPS_DIR=${STAGE2_PATH}
                )
        endif()
    endif()
endif()
if (WIN32)
    list(APPEND ISPC_STAGE2_CMAKE_ARGS
        -DLLVM_DIR=${STAGE2_PATH}/lib/cmake/llvm
        )
endif()

# LTO specific compiler flags are specified in corresponding toolchain file.
if (LTO)
    list(APPEND ISPC_STAGE2_CMAKE_ARGS
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_BINARY_DIR}/stage2-lto-toolchain.cmake
        )
else()
    list(APPEND ISPC_STAGE2_CMAKE_ARGS
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_BINARY_DIR}/stage2-toolchain.cmake
        )
endif()

list(APPEND ISPC_STAGE2_CMAKE_TARGETS install package)
if (NOT ISPC_PACKAGE_EXAMPLES)
    list(APPEND ISPC_STAGE2_CMAKE_TARGETS
        package-examples
        )
endif()
message(STATUS "ISPC_STAGE2_CMAKE_ARGS is ${ISPC_STAGE2_CMAKE_ARGS}")
message(STATUS "ISPC_STAGE2_CMAKE_TARGETS is ${ISPC_STAGE2_CMAKE_TARGETS}")

# LLVM stage3 specific cmake arguments, here we imply that PGO always works with LTO.
# PGO specific compiler flags is inside toolchain file.
if (PGO)
    list(APPEND LLVM_STAGE3_CMAKE_ARGS
        ${LLVM_STAGE2_COMMON_CMAKE_ARGS}
        -DLLVM_ENABLE_LTO=Thin
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_BINARY_DIR}/stage3-pgo-toolchain.cmake
        )
    if (APPLE)
        list(APPEND LLVM_STAGE3_CMAKE_ARGS
            -DLD64_EXECUTABLE=${STAGE3_PATH}/bin/ld64.lld
            )
    endif()
    message(STATUS "LLVM stage3 cmake arguments ${LLVM_STAGE3_CMAKE_ARGS}")

    list(APPEND ISPC_STAGE3_CMAKE_ARGS
        ${ISPC_COMMON_CMAKE_ARGS}
        -DXE_DEPS_DIR=${CMAKE_BINARY_DIR}/stage3
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_BINARY_DIR}/stage3-pgo-toolchain.cmake
        )
    if (XE_DEPS)
        list(APPEND ISPC_STAGE3_CMAKE_ARGS
            -DLEVEL_ZERO_ROOT=${CMAKE_BINARY_DIR}/stage3
            )
    endif()
    if (WIN32)
        list(APPEND ISPC_STAGE3_CMAKE_ARGS
            -DLLVM_DIR=${CMAKE_BINARY_DIR}/stage3/lib/cmake/llvm
            )
    endif()
    message(STATUS "ISPC stage3 cmake arguments ${ISPC_STAGE3_CMAKE_ARGS}")
endif()


set(PGO_FLAGS "")
if (PGO)
    set(PGO_FLAGS -fprofile-instr-generate)
endif()

set(CODE_PROFDATA ${CMAKE_BINARY_DIR}/code.profdata)
set(CLANG_RT_VERSION ${LLVM_VERSION_DOTTED})
if (${LLVM_VERSION_MAJOR} VERSION_GREATER_EQUAL 16)
    set(CLANG_RT_VERSION ${LLVM_VERSION_MAJOR})
endif()
set(CLANG_RT_TARGET x86_64-pc-windows-msvc)
if (${LLVM_VERSION_MAJOR} VERSION_GREATER_EQUAL 17)
    set(CLANG_RT_TARGET windows)
endif()
set(CLANG_RT_LIB clang_rt.profile.lib)
if (${LLVM_VERSION_MAJOR} VERSION_GREATER_EQUAL 17)
    set(CLANG_RT_LIB clang_rt.profile-x86_64.lib)
endif()
# MSVC has an outdated clang_rt.profile lib in its SDK.
# We need to enforce explicitly our freshly built library to avoid profile mismatch errors.
cmake_path(CONVERT
           ${STAGE2_PATH}/lib/clang/${CLANG_RT_VERSION}/lib/${CLANG_RT_TARGET}/${CLANG_RT_LIB}
           TO_CMAKE_PATH_LIST
           CLANG_RT_PROFLIB
           NORMALIZE)
if (WIN32)
    message(STATUS "CLANG_RT_PROFLIB is ${CLANG_RT_PROFLIB}")
endif()

# Generate value for ENV{PATH} with the corresponding stage toolchain.
# _ESC_SEM is needed because of list expansion in ExternalProject_Add.
function(gen_env_path OUT_VAR OUT_VAR_ESC_SEM STAGE_PATH)
    list(APPEND ENV_PATH
        ${STAGE_PATH}
        "$ENV{PATH}"
        )
    cmake_path(CONVERT "${ENV_PATH}" TO_NATIVE_PATH_LIST ENV_PATH)
    string(REPLACE "\\" "\\\\" ENV_PATH "${ENV_PATH}")
    string(REPLACE ";" "$<SEMICOLON>" ENV_PATH_ESC_SEM "${ENV_PATH}")
    set(${OUT_VAR} ${ENV_PATH} PARENT_SCOPE)
    set(${OUT_VAR_ESC_SEM} ${ENV_PATH_ESC_SEM} PARENT_SCOPE)
endfunction()

gen_env_path(ENV_PATH_STAGE2 ENV_PATH_STAGE2_ESC_SEM "${STAGE2_PATH}/bin")
gen_env_path(ENV_PATH_STAGE3 ENV_PATH_STAGE3_ESC_SEM "${CMAKE_BINARY_DIR}/stage3/bin")
message(STATUS "ENV_PATH_STAGE2 is ${ENV_PATH_STAGE2}")
message(STATUS "ENV_PATH_STAGE3 is ${ENV_PATH_STAGE3}")


# Generate toolchain files regardless current platform and configuration
# because it is easier to check them.
# The stage1 compiler are built with system toolchain.
file(CONFIGURE OUTPUT stage1-toolchain.cmake CONTENT [[
@CCACHE_LAUNCHER_ACTION@(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_EXE})
@CCACHE_LAUNCHER_ACTION@(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_EXE})
set(CMAKE_C_FLAGS   "@CMAKE_C_FLAGS@")
set(CMAKE_CXX_FLAGS "@CMAKE_CXX_FLAGS@")
if (WIN32)
    set(CMAKE_C_COMPILER   cl)
    set(CMAKE_CXX_COMPILER cl)
    set(CMAKE_ASM_COMPILER ml64)
else()
    set(CMAKE_C_COMPILER   cc)
    set(CMAKE_CXX_COMPILER c++)
endif()
]] @ONLY)

# Base toolchain file for building ISPC and LLVM dependencies.
file(CONFIGURE OUTPUT stage2-toolchain.cmake CONTENT [[
@CCACHE_LAUNCHER_ACTION@(CMAKE_C_COMPILER_LAUNCHER   @CCACHE_EXE@)
@CCACHE_LAUNCHER_ACTION@(CMAKE_CXX_COMPILER_LAUNCHER @CCACHE_EXE@)
set(ENV{PATH}       "@ENV_PATH_STAGE2@")
set(WIN_C_FLAGS     "-fuse-ld=lld-link")
set(APPLE_C_FLAGS   "-fuse-ld=lld")
if (WIN32)
    set(CMAKE_C_COMPILER   clang-cl)
    set(CMAKE_CXX_COMPILER clang-cl)
    set(CMAKE_LINKER       lld-link)
    set(CMAKE_AR           llvm-lib)
    set(CMAKE_RC_COMPILER  llvm-rc)
    set(CMAKE_MT           mt)
    set(CMAKE_C_FLAGS      "@CMAKE_C_FLAGS@ ${WIN_C_FLAGS}")
    set(CMAKE_CXX_FLAGS    "@CMAKE_CXX_FLAGS@ ${WIN_C_FLAGS}")
elseif(APPLE)
    set(CMAKE_C_COMPILER   clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_LINKER       ld64.lld)
    set(CMAKE_LIBTOOL      llvm-libtool-darwin)
    set(CMAKE_RANLIB       llvm-ranlib)
    set(CMAKE_C_FLAGS      "@CMAKE_C_FLAGS@")
    set(CMAKE_CXX_FLAGS    "@CMAKE_CXX_FLAGS@")
    set(CMAKE_MODULE_LINKER_FLAGS "${APPLE_C_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${APPLE_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS    "${APPLE_C_FLAGS}")
else()
    set(CMAKE_C_COMPILER   clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_LINKER       lld)
    set(CMAKE_C_FLAGS      "@CMAKE_C_FLAGS@")
    set(CMAKE_CXX_FLAGS    "@CMAKE_CXX_FLAGS@")
endif()
]] @ONLY)

# Toolchain file for LTO build or PGO stage instrumenting LLVM libraries.
file(CONFIGURE OUTPUT stage2-lto-toolchain.cmake CONTENT [[
@CCACHE_LAUNCHER_ACTION@(CMAKE_C_COMPILER_LAUNCHER   @CCACHE_EXE@)
@CCACHE_LAUNCHER_ACTION@(CMAKE_CXX_COMPILER_LAUNCHER @CCACHE_EXE@)
set(ENV{PATH} "@ENV_PATH_STAGE2@")
set(COMMON_FLAGS    "-flto=thin @PGO_FLAGS@")
set(COMMON_C_FLAGS  "@CMAKE_C_FLAGS@")
# Note: -O3 --lto-O3 --icf=...  that these flags lld specific
set(COMMON_LD_FLAGS "-Wl,-O3 -Wl,--lto-O3 -Wl,--icf=all")
set(WIN_C_FLAGS     "-fuse-ld=lld-link")
set(WIN_LD_FLAGS    "@CLANG_RT_PROFLIB@")
set(APPLE_C_FLAGS   "-fuse-ld=lld")
if (WIN32)
    set(CMAKE_C_COMPILER   clang-cl)
    set(CMAKE_CXX_COMPILER clang-cl)
    set(CMAKE_LINKER       lld-link)
    set(CMAKE_AR           llvm-lib)
    set(CMAKE_RC_COMPILER  llvm-rc)
    set(CMAKE_MT           mt)
    set(CMAKE_C_FLAGS             "${COMMON_FLAGS} ${COMMON_C_FLAGS} ${WIN_C_FLAGS}")
    set(CMAKE_CXX_FLAGS           "@CMAKE_CXX_FLAGS@ ${COMMON_FLAGS} ${COMMON_C_FLAGS} ${WIN_C_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${WIN_LD_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${WIN_LD_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS    "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${WIN_LD_FLAGS}")
elseif(APPLE)
    set(CMAKE_C_COMPILER   clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_LINKER       ld64.lld)
    set(CMAKE_LIBTOOL      llvm-libtool-darwin)
    set(CMAKE_RANLIB       llvm-ranlib)
    set(CMAKE_C_FLAGS             "${COMMON_FLAGS} ${COMMON_C_FLAGS}")
    set(CMAKE_CXX_FLAGS           "@CMAKE_CXX_FLAGS@ ${COMMON_FLAGS} ${COMMON_C_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${APPLE_C_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${APPLE_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS    "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${APPLE_C_FLAGS}")
else()
    set(CMAKE_C_COMPILER   clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_LINKER       lld)
    set(CMAKE_C_FLAGS             "${COMMON_FLAGS} ${COMMON_C_FLAGS}")
    set(CMAKE_CXX_FLAGS           "@CMAKE_CXX_FLAGS@ ${COMMON_FLAGS} ${COMMON_C_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS    "${COMMON_FLAGS} ${COMMON_LD_FLAGS}")
endif()
]] @ONLY)

# Special toolchain file for building level-zero.
# Primarily needed because if warning suppression for windows.
# https://github.com/oneapi-src/level-zero/issues/83
file(CONFIGURE OUTPUT stage2-l0-toolchain.cmake CONTENT [[
@CCACHE_LAUNCHER_ACTION@(CMAKE_C_COMPILER_LAUNCHER   @CCACHE_EXE@)
@CCACHE_LAUNCHER_ACTION@(CMAKE_CXX_COMPILER_LAUNCHER @CCACHE_EXE@)
set(ENV{PATH} "@ENV_PATH_STAGE2@")
set(WIN_C_FLAGS "-Wno-error=unused-command-line-argument -Wno-error=unused-but-set-variable")
set(APPLE_C_FLAGS   "-fuse-ld=lld")
if (WIN32)
    set(CMAKE_C_COMPILER   clang-cl)
    set(CMAKE_CXX_COMPILER clang-cl)
    set(CMAKE_LINKER       lld-link)
    set(CMAKE_AR           llvm-lib)
    set(CMAKE_RC_COMPILER  llvm-rc)
    set(CMAKE_MT           mt)
    set(CMAKE_C_FLAGS      "@CMAKE_C_FLAGS@ ${WIN_C_FLAGS}")
    set(CMAKE_CXX_FLAGS    "@CMAKE_CXX_FLAGS@ ${WIN_C_FLAGS}")
    # Since v1.3 at least this version is requited,
    # see  https://github.com/oneapi-src/level-zero/issues/61
    set(CMAKE_SYSTEM_VERSION 10.0.19041.0)
elseif(APPLE)
    set(CMAKE_C_COMPILER   clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_LINKER       ld64.lld)
    set(CMAKE_LIBTOOL      llvm-libtool-darwin)
    set(CMAKE_RANLIB       llvm-ranlib)
    set(CMAKE_C_FLAGS      "@CMAKE_C_FLAGS@")
    set(CMAKE_CXX_FLAGS    "@CMAKE_CXX_FLAGS@")
    set(CMAKE_MODULE_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${APPLE_C_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${APPLE_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS    "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${APPLE_C_FLAGS}")
else()
    set(CMAKE_C_COMPILER   clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_LINKER       lld)
    set(CMAKE_C_FLAGS      "@CMAKE_C_FLAGS@")
    set(CMAKE_CXX_FLAGS    "@CMAKE_CXX_FLAGS@")
endif()
]] @ONLY)

# Stage3 to optimize with the collected profile.
file(CONFIGURE OUTPUT stage3-pgo-toolchain.cmake CONTENT [[
set(ENV{PATH} "@ENV_PATH_STAGE3@")
set(COMMON_FLAGS    "-flto=thin -fprofile-instr-use=@CODE_PROFDATA@")
set(COMMON_C_FLAGS  "@CMAKE_C_FLAGS@")
# Note: -O3 --lto-O3 --icf=...  that these flags lld specific
set(COMMON_LD_FLAGS "-Wl,-O3 -Wl,--lto-O3 -Wl,--icf=all")
set(WIN_C_FLAGS     "-fuse-ld=lld-link")
set(WIN_LD_FLAGS    "")
set(APPLE_C_FLAGS   "-fuse-ld=lld")
if (WIN32)
    set(CMAKE_C_COMPILER   clang-cl)
    set(CMAKE_CXX_COMPILER clang-cl)
    set(CMAKE_LINKER       lld-link)
    set(CMAKE_AR           llvm-lib)
    set(CMAKE_RC_COMPILER  llvm-rc)
    set(CMAKE_MT           mt)
    set(CMAKE_C_FLAGS             "${COMMON_FLAGS} ${COMMON_C_FLAGS} ${WIN_C_FLAGS}")
    set(CMAKE_CXX_FLAGS           "@CMAKE_CXX_FLAGS@ ${COMMON_FLAGS} ${COMMON_C_FLAGS} ${WIN_C_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${WIN_LD_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${WIN_LD_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS    "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${WIN_LD_FLAGS}")
elseif(APPLE)
    set(CMAKE_C_COMPILER   clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_LINKER       ld64.lld)
    set(CMAKE_LIBTOOL      llvm-libtool-darwin)
    set(CMAKE_RANLIB       llvm-ranlib)
    set(CMAKE_C_FLAGS             "${COMMON_FLAGS} ${COMMON_C_FLAGS}")
    set(CMAKE_CXX_FLAGS           "@CMAKE_CXX_FLAGS@ ${COMMON_FLAGS} ${COMMON_C_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${APPLE_C_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${APPLE_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS    "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${APPLE_C_FLAGS}")
else()
    set(CMAKE_C_COMPILER   clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_LINKER       lld)
    set(CMAKE_C_FLAGS             "${COMMON_FLAGS} ${COMMON_C_FLAGS}")
    set(CMAKE_CXX_FLAGS           "@CMAKE_CXX_FLAGS@ ${COMMON_FLAGS} ${COMMON_C_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS    "${COMMON_FLAGS} ${COMMON_LD_FLAGS}")
endif()
]] @ONLY)


# Fetch source code of dependencies. They fetched during cmake configuration run.
# Patch command applies LLVM version specific patches.
if (PGO OR NOT SKIP_STAGE2)
    if (LLVM_PATCHES_VER)
        FetchContent_Populate(llvm-source
            QUIET
            GIT_REPOSITORY ${LLVM_URL}
            GIT_TAG ${LLVM_TAG}
            GIT_SHALLOW ON
            GIT_PROGRESS ON
            UPDATE_COMMAND git checkout .
            PATCH_COMMAND git apply ${LLVM_PATCHES_VER}
            SOURCE_DIR llvm-source
            )
    else()
        FetchContent_Populate(llvm-source
            QUIET
            GIT_REPOSITORY ${LLVM_URL}
            GIT_TAG ${LLVM_TAG}
            GIT_SHALLOW ON
            GIT_PROGRESS ON
            UPDATE_COMMAND git checkout .
            SOURCE_DIR llvm-source
            )
    endif()

    if (NOT BUILD_STAGE2_TOOLCHAIN_ONLY AND PGO AND NOT ${ISPC_CORPUS_URL} STREQUAL "null")
        # May be profiles from examples or less set of programs just enough?
        FetchContent_Populate(ispc-corpus
            QUIET
            GIT_REPOSITORY ${ISPC_CORPUS_URL}
            GIT_TAG main
            GIT_SHALLOW ON
            GIT_PROGRESS ON
            UPDATE_COMMAND git clean -f -d
            SOURCE_DIR ispc-corpus
            )
    endif()
endif()

if (NOT SKIP_FETCHING_XE_DEPS)
    if (XE_DEPS OR (SKIP_STAGE2 AND BUILD_SPIRV_TRANSLATOR_ONLY))
        # Checkout LLVM-SPIRV-Translator needed for XE
        FetchContent_Populate(spirv-translator
            QUIET
            GIT_REPOSITORY ${SPIRV_TRANSLATOR_URL}
            GIT_TAG ${SPIRV_TRANSLATOR_SHA}
            GIT_SHALLOW OFF
            GIT_PROGRESS ON
            SOURCE_DIR spirv-translator
            )
    endif()

    if (XE_DEPS OR (SKIP_STAGE2 AND BUILD_VC_INTRINSICS_ONLY))
        # Checkout vc-intrinsics needed for XE
        FetchContent_Populate(vc-intrinsics
            QUIET
            GIT_REPOSITORY ${VC_INTRINSICS_URL}
            GIT_TAG ${VC_INTRINSICS_SHA}
            GIT_SHALLOW OFF
            GIT_PROGRESS ON
            SOURCE_DIR vc-intrinsics
            UPDATE_COMMAND git checkout .
            PATCH_COMMAND git apply ${PROJECT_SOURCE_DIR}/vc_intrinsics.patch
            )
    endif()

    if (XE_DEPS OR (SKIP_STAGE2 AND BUILD_L0_LOADER_ONLY))
        # Checkout level-zero needed for XE
        FetchContent_Populate(l0-source
            QUIET
            GIT_REPOSITORY ${L0_URL}
            GIT_TAG ${L0_TAG}
            GIT_SHALLOW ON
            GIT_PROGRESS ON
            SOURCE_DIR l0-source
            )
    endif()
endif()


# Run build and install commands in required environment.
# It is needed to find ccache.conf files and to properly locate all needed tools during build.
list(APPEND NINJA_COMMAND_WITH_ENV_STAGE1
    ${CMAKE_COMMAND}
    -E
    env
    "CCACHE_CONFIGPATH=${CCACHE_DEFAULT_CONF}"
    ${NINJA_EXE}
    ${EXTERNAL_VERBOSE_FLAG}
    ${NINJA_JOBS}
    )
list(APPEND NINJA_COMMAND_WITH_ENV_STAGE2
    ${CMAKE_COMMAND}
    -E
    env
    "CCACHE_CONFIGPATH=${CCACHE_CLANG_V_CONF}"
    )
# Workaround of Windows cmd related issue with cmd with too big PATH effectively doing nothing.
if (EXPLICIT_ENV_PATH)
    list(APPEND NINJA_COMMAND_WITH_ENV_STAGE2
        "PATH=${ENV_PATH_STAGE2_ESC_SEM}"
        )
endif()
list(APPEND NINJA_COMMAND_WITH_ENV_STAGE2
    ${NINJA_EXE}
    ${EXTERNAL_VERBOSE_FLAG}
    ${NINJA_JOBS}
    )
list(APPEND NINJA_COMMAND_WITH_ENV_STAGE3
    ${CMAKE_COMMAND}
    -E
    env
    "CCACHE_CONFIGPATH=${CCACHE_CLANG_V_CONF}"
    )
if (EXPLICIT_ENV_PATH)
    list(APPEND NINJA_COMMAND_WITH_ENV_STAGE3
        "PATH=${ENV_PATH_STAGE3_ESC_SEM}"
        )
endif()
list(APPEND NINJA_COMMAND_WITH_ENV_STAGE3
    ${NINJA_EXE}
    ${EXTERNAL_VERBOSE_FLAG}
    ${NINJA_JOBS}
    )
message(STATUS "NINJA_COMMAND_WITH_ENV_STAGE1 is ${NINJA_COMMAND_WITH_ENV_STAGE1}")
message(STATUS "NINJA_COMMAND_WITH_ENV_STAGE2 is ${NINJA_COMMAND_WITH_ENV_STAGE2}")
message(STATUS "NINJA_COMMAND_WITH_ENV_STAGE3 is ${NINJA_COMMAND_WITH_ENV_STAGE3}")

set(XE_DEPS_TOOLCHAIN_FILE ${CMAKE_BINARY_DIR}/stage2-toolchain.cmake)
if (LTO)
    set(XE_DEPS_TOOLCHAIN_FILE ${CMAKE_BINARY_DIR}/stage2-lto-toolchain.cmake)
endif()
message(STATUS "XE_DEPS_TOOLCHAIN_FILE is ${XE_DEPS_TOOLCHAIN_FILE}")

if (BUILD_SPIRV_STANDALONE)
    ExternalProject_Add(spirv-translator
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        SOURCE_DIR spirv-translator
        PREFIX bspirv
        STAMP_DIR bspirv/s
        BINARY_DIR bspirv/b
        TMP_DIR bspirv/t
        INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS
            -DLLVM_DIR=${PREBUILT_STAGE2_PATH}/lib/cmake/llvm
            -DCMAKE_BUILD_TYPE=${BUILD_TYPE_SPIRV_TRANSLATOR}
            -DCMAKE_TOOLCHAIN_FILE=${XE_DEPS_TOOLCHAIN_FILE}
            -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        BUILD_IN_SOURCE OFF
        BUILD_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2}
        INSTALL_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2} install
        )
    if (BUILD_SPIRV_TRANSLATOR_ONLY)
        message(STATUS "BUILD SPIRV-LLVM-Translator only")
        return()
    else()
        message(STATUS "BUILD SPIRV-LLVM-Translator")
    endif()
endif()

if (BUILD_VC_INTRINSICS_STANDALONE)
    ExternalProject_Add(vc-intrinsics
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        SOURCE_DIR vc-intrinsics
        PREFIX bvcint
        STAMP_DIR bvcint/s
        BINARY_DIR bvcint/b
        TMP_DIR bvcint/t
        INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS
            -DLLVM_DIR=${PREBUILT_STAGE2_PATH}/lib/cmake/llvm
            -DCMAKE_BUILD_TYPE=${BUILD_TYPE_VC_INTRINSICS}
            -DCMAKE_TOOLCHAIN_FILE=${XE_DEPS_TOOLCHAIN_FILE}
            -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        BUILD_IN_SOURCE OFF
        BUILD_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2}
        INSTALL_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2} install
        )
    if (BUILD_VC_INTRINSICS_ONLY)
        message(STATUS "BUILD vc-intrinsics only")
        return()
    else()
        message(STATUS "BUILD vc-intrinsics")
    endif()
endif()

if (BUILD_L0_LOADER_STANDALONE)
    # Level-zero
    ExternalProject_Add(l0
        DOWNLOAD_COMMAND ""
        SOURCE_DIR l0-source
        BUILD_IN_SOURCE OFF
        PREFIX bl0
        STAMP_DIR bl0/s
        BINARY_DIR bl0/b
        TMP_DIR bl0/t
        INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_BINARY_DIR}/stage2-l0-toolchain.cmake
            -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
            -DCMAKE_BUILD_TYPE=${BUILD_TYPE_L0}
        BUILD_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2}
        INSTALL_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2} install
        )
    if (BUILD_L0_LOADER_ONLY)
        message(STATUS "BUILD l0 loader only")
        return()
    else()
        message(STATUS "BUILD l0 loader")
    endif()
endif()

if (BUILD_XE_DEPS_ONLY)
    return()
endif()

# Stage1 job to build bootstrap compiler for building required Clang/LLVM libraries and ISPC itself.
if (PGO OR NOT SKIP_STAGE2)

    # It may be empty when PREBUILT_STAGE1_PATH provided otherwise we need to depend on the stage1 external project
    list(APPEND LLVM_STAGE2_DEPS)

    if (NOT SKIP_STAGE1)
        ExternalProject_Add(stage1
            DOWNLOAD_COMMAND ""
            UPDATE_COMMAND ""
            SOURCE_DIR llvm-source
            SOURCE_SUBDIR llvm
            PREFIX bstage1
            STAMP_DIR bstage1/s
            BINARY_DIR bstage1/b
            TMP_DIR bstage1/t
            INSTALL_DIR ${TOOLCHAIN_STAGE1_INSTALL_PREFIX}
            CMAKE_GENERATOR Ninja
            CMAKE_ARGS ${LLVM_STAGE1_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${TOOLCHAIN_STAGE1_INSTALL_PREFIX}
            BUILD_IN_SOURCE OFF
            BUILD_COMMAND ""
            BUILD_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE1}
            INSTALL_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE1} install
            )
        # Note: We may rebuild instead stage2 clang/lld with stage1 toolchain to
        # generate more performant stage2 compiler.

        list(APPEND LLVM_STAGE2_DEPS stage1)

        if (BUILD_STAGE1_TOOLCHAIN_ONLY)
            # Custom target to package stage1 tooclhain
            add_custom_target(package-stage1
                DEPENDS stage1
                COMMENT "Packaging stage1 archive to llvm-stage1-${LLVM_VERSION_DOTTED}.tgz"
                COMMAND ${CMAKE_COMMAND} -E tar cfvz "llvm-stage1-${LLVM_VERSION_DOTTED}.tgz" stage2
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
            return()
        endif()

    else()

        if (NOT PREBUILT_STAGE1 STREQUAL "NO")
            # Extract pre-built stage1, path should be provided in PREBUILT_STAGE1.
            ExternalProject_Add(stage1
                URL ${PREBUILT_STAGE1}
                DOWNLOAD_EXTRACT_TIMESTAMP OFF
                UPDATE_COMMAND ""
                SOURCE_DIR stage2
                CONFIGURE_COMMAND ""
                BUILD_COMMAND ""
                INSTALL_COMMAND ""
                )

            list(APPEND LLVM_STAGE2_DEPS stage1)
        endif()

    endif()


    list(APPEND L0_DEPS)
    if (XE_DEPS)
        # Level-zero
        list(APPEND STAGE1_DEPS)
        if (NOT SKIP_STAGE1)
            list(APPEND STAGE1_DEPS stage1)
        endif()
        ExternalProject_Add(l0
            DEPENDS ${STAGE1_DEPS}
            DOWNLOAD_COMMAND ""
            CMAKE_GENERATOR Ninja
            CMAKE_ARGS
                -DCMAKE_TOOLCHAIN_FILE=${CMAKE_BINARY_DIR}/stage2-l0-toolchain.cmake
                -DCMAKE_INSTALL_PREFIX=${TOOLCHAIN_STAGE2_INSTALL_PREFIX}
                -DCMAKE_BUILD_TYPE=${BUILD_TYPE_L0}
            SOURCE_DIR l0-source
            BUILD_IN_SOURCE OFF
            PREFIX bl0
            STAMP_DIR bl0/s
            BINARY_DIR bl0/b
            TMP_DIR bl0/t
            INSTALL_DIR ${TOOLCHAIN_STAGE2_INSTALL_PREFIX}
            BUILD_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2}
            INSTALL_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2} install
            )
        list(APPEND L0_DEPS l0)
    endif()

    if (PGO)
        # Copy stage1 compiler to stage3 install dir.
        add_custom_command(OUTPUT stage3-copied.stamp
            DEPENDS stage1 ${L0_DEPS}
            COMMENT "Copy stage2 to stage3"
            COMMAND ${CMAKE_COMMAND} -E copy_directory stage2 stage3
            COMMAND ${CMAKE_COMMAND} -E touch stage3-copied.stamp
            )
        add_custom_target(pre-stage3
            DEPENDS stage3-copied.stamp
            )
    endif()

    # Build Clang/LLVM static libraries required for ISPC.
    # It is rather important to build only them as much as possible, because under
    # LTO/PGO linking extra stuff may take long time.
    # On that step, we build external LLVM projects together with LLVM.
    # When LTO, this stage has barely only -flto extra flag comparing with base build.
    # When PGO, this step produces instrumented Clang/LLVM static libraries.
    ExternalProject_Add(stage2
        DEPENDS ${LLVM_STAGE2_DEPS} ${L0_DEPS}
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        SOURCE_DIR llvm-source
        SOURCE_SUBDIR llvm
        PREFIX bstage2
        STAMP_DIR bstage2/s
        BINARY_DIR bstage2/b
        TMP_DIR bstage2/t
        INSTALL_DIR ${TOOLCHAIN_STAGE2_INSTALL_PREFIX}
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS ${LLVM_STAGE2_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${TOOLCHAIN_STAGE2_INSTALL_PREFIX}
        BUILD_IN_SOURCE OFF
        BUILD_COMMAND ""
        INSTALL_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2} ${LLVM_STAGE2_INSTALL_TARGETS}
        )

    # Custom target to package stage2 tooclhain to use after skipping its repetitive build.
    add_custom_target(package-stage2
        DEPENDS stage2
        COMMENT "Packaging stage2 archive to llvm-stage2-${LLVM_VERSION_DOTTED}.tgz"
        COMMAND ${CMAKE_COMMAND} -E tar cfvz "llvm-stage2-${LLVM_VERSION_DOTTED}.tgz" ${TOOLCHAIN_STAGE2_INSTALL_PREFIX}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

list(APPEND ISPC_STAGE2_DEPS)
if (SKIP_STAGE2)
    if (NOT PREBUILT_STAGE2 STREQUAL "NO")
        # Extract pre-built stage2, path should be provided in PREBUILT_STAGE2.
        ExternalProject_Add(stage2
            URL ${PREBUILT_STAGE2}
            DOWNLOAD_EXTRACT_TIMESTAMP OFF
            UPDATE_COMMAND ""
            SOURCE_DIR stage2
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            )
        list(APPEND ISPC_STAGE2_DEPS stage2)
    endif()
    if (INSTALL_WITH_XE_DEPS)
        list(APPEND ISPC_STAGE2_DEPS spirv-translator vc-intrinsics l0)
    endif()
else()
    list(APPEND ISPC_STAGE2_DEPS ${L0_DEPS} stage2)
endif()

if (NOT BUILD_STAGE2_TOOLCHAIN_ONLY)
    # Build ispc first time with stage2 compiler/libraries.
    # For LTO only build, this ispc is final artifact.
    message(STATUS "ISPC stage2 depends on ${ISPC_STAGE2_DEPS}")
    ExternalProject_Add(ispc-stage2
        DEPENDS ${ISPC_STAGE2_DEPS}
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        PREFIX build-ispc-stage2
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/..
        INSTALL_DIR ${ISPC_STAGE2_INSTALL_PREFIX}
        BUILD_IN_SOURCE OFF
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS ${ISPC_STAGE2_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${ISPC_STAGE2_INSTALL_PREFIX}
        BUILD_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2}
        INSTALL_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2} ${ISPC_STAGE2_CMAKE_TARGETS}
        )

    # Custom target to run stage2 ISPC tests
    add_custom_target(ispc-stage2-check-all
        DEPENDS ispc-stage2
        COMMENT "Run ISPC stage2 check-all target"
        COMMAND ${CMAKE_COMMAND} -E chdir build-ispc-stage2/src/ispc-stage2-build ${NINJA_COMMAND_WITH_ENV_STAGE2} check-all
        )

    # Custom target to run stage2 ISPC check-all, ispc_benchmarks and test
    add_custom_target(ispc-stage2-check
        DEPENDS ispc-stage2
        COMMENT "Run ISPC stage2 check-all target"
        COMMAND ${CMAKE_COMMAND} -E chdir build-ispc-stage2/src/ispc-stage2-build ${NINJA_COMMAND_WITH_ENV_STAGE2} check-all ispc_benchmarks test
        )

    # The following steps applicable only for PGO type build.
    if (PGO)
        find_package(Python3 COMPONENTS Interpreter)

        # Collect and merge profiles.
        add_custom_command(OUTPUT ${CODE_PROFDATA}
            DEPENDS ispc-stage2 stage2 ispc-corpus/profile.py
            COMMAND ${Python3_EXECUTABLE}
                ispc-corpus/profile.py
                    --ispc ${CMAKE_BINARY_DIR}/ispc-stage2/bin/ispc${CMAKE_EXECUTABLE_SUFFIX}
                    --llvm-profdata ${STAGE2_PATH}/bin/llvm-profdata${CMAKE_EXECUTABLE_SUFFIX}
                    --output ${CODE_PROFDATA}
            COMMENT "Collecting profile on ispc-corpus"
            )
        add_custom_target(generate-profdata
            DEPENDS ${CODE_PROFDATA}
            )

        # Build LLVM with PGO optimizations.
        ExternalProject_Add(stage3
            DEPENDS pre-stage3 ${L0_DEPS} generate-profdata
            DOWNLOAD_COMMAND ""
            UPDATE_COMMAND ""
            PREFIX build-llvm-stage3
            SOURCE_DIR llvm-source
            SOURCE_SUBDIR llvm
            INSTALL_DIR stage3
            CMAKE_GENERATOR Ninja
            CMAKE_ARGS ${LLVM_STAGE3_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/stage3
            BUILD_IN_SOURCE OFF
            BUILD_COMMAND ""
            INSTALL_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE3} ${LLVM_STAGE2_INSTALL_TARGETS}
            )

        # Finally, build ISPC with PGO based on the collected profile.
        # For PGO build, this ispc is final artifact.
        ExternalProject_Add(ispc-stage3
            DEPENDS stage3
            DOWNLOAD_COMMAND ""
            UPDATE_COMMAND ""
            PREFIX build-ispc-stage3
            SOURCE_DIR ${PROJECT_SOURCE_DIR}/..
            INSTALL_DIR ${ISPC_STAGE3_INSTALL_PREFIX}
            BUILD_IN_SOURCE OFF
            CMAKE_GENERATOR Ninja
            CMAKE_ARGS ${ISPC_STAGE3_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${ISPC_STAGE3_INSTALL_PREFIX}
            BUILD_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE3}
            INSTALL_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE3} install package
            )
    endif()
endif()
