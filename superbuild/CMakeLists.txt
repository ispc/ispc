#
#  Copyright (c) 2023, Intel Corporation
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
set(CCACHE_VERSION "4.7.4" CACHE STRING "ccache version ")
set(CCACHE_DIR "default" CACHE STRING "ccache cache directory")
set(PREBUILT_STAGE2 "NO" CACHE STRING "Provide pre-built stage2 archive to skip pre ispc-stage2 jobs")
set(PREBUILT_STAGE2_PATH "NO" CACHE STRING "Provide pre-built stage2 toolchain path")
set(STAGE2_TOOLCHAIN_INSTALL_PREFIX "NO" CACHE STRING "Path to install stage2 Clang/LLVM to")
set(BUILD_TYPE_ISPC "Release" CACHE STRING "Set the build type for ISPC build")
set(BUILD_TYPE_LLVM "Release" CACHE STRING "Set the build type for LLVM build")
set(BUILD_TYPE_SPIRV_TRANSLATOR "Release" CACHE STRING "Set the build type for SPIR-V Translator build")
set(BUILD_TYPE_VC_INTRINSICS "Release" CACHE STRING "Set the build type for VC Intrinsics build")
set(BUILD_TYPE_L0 "Release" CACHE STRING "Set the build type for L0 build")
option(LLVM_DISABLE_ASSERTIONS "Enable LLVM build without assertions" ON)
option(EXTERNAL_VERBOSE "Enable verbose for external project build and install commands" OFF)
option(CCACHE "Enable ccache to speed up repetitive builds" OFF)
option(FULL_STAGE2_LLVM "Enable full llvm stage2 build" OFF)
option(LTO "Enable LTO" OFF)
option(PGO "Enale PGO (implicitly enable LTO)" OFF)
option(BUILD_STAGE2_TOOLCHAIN_ONLY "Enable build of stage2 toolchain only, i.e., skip ispc building and stage3." OFF)
option(BUILD_SPIRV_TRANSLATOR_ONLY "Enable only build of SPIRV-LLVM-Translator" OFF)
option(BUILD_VC_INTRINSICS_ONLY "Enable only build of vc-intrinsics" OFF)
option(BUILD_L0_LOADER_ONLY "Enable only build of level-zero loader" OFF)
option(XE_DEPS "Enable build of XE dependencies (L0 loader, vc-intrinsics, SPIRV-LLVM-Translator" ON)

# Force LTO when PGO enabled to the sake of simplicity and less number of
# possible build variants.
if (PGO)
    set(LTO ON)
endif()

if (NOT PREBUILT_STAGE2 STREQUAL "NO" OR NOT PREBUILT_STAGE2_PATH STREQUAL "NO")
    set(SKIP_STAGE2_DEPS ON)
endif()

if (NOT PREBUILT_STAGE2 STREQUAL "NO" AND NOT PREBUILT_STAGE2_PATH STREQUAL "NO")
    message(FATAL_ERROR "Provide only one source of stage2 toolchain.")
endif()

set(STAGE2_PATH "${CMAKE_BINARY_DIR}/stage2")
if (NOT PREBUILT_STAGE2_PATH STREQUAL "NO")
    set(STAGE2_PATH "${PREBUILT_STAGE2_PATH}")
endif()

string(APPEND LIT_TOOLS_DIR ${GNUWIN32} "/bin")

unset(LLVM_TAG)
unset(LLVM_VERSION_DOTTED)
string(REPLACE "." "_" LLVM_VERSION "${LLVM_VERSION}")
if     (${LLVM_VERSION} STREQUAL 16_0)
    set(LLVM_TAG    llvmorg-16.0.6)
    set(LLVM_VERSION_DOTTED 16.0.6)
elseif (${LLVM_VERSION} STREQUAL 15_0)
    set(LLVM_TAG    llvmorg-15.0.7)
    set(LLVM_VERSION_DOTTED 15.0.7)
elseif (${LLVM_VERSION} STREQUAL 14_0)
    set(LLVM_TAG    llvmorg-14.0.6)
    set(LLVM_VERSION_DOTTED 14.0.6)
elseif (${LLVM_VERSION} STREQUAL 13_0)
    set(LLVM_TAG    llvmorg-13.0.1)
    set(LLVM_VERSION_DOTTED 13.0.1)
elseif (${LLVM_VERSION} STREQUAL 12_0)
    set(LLVM_TAG    llvmorg-12.0.1)
    set(LLVM_VERSION_DOTTED 12.0.1)
elseif (${LLVM_VERSION} STREQUAL 11_1)
    set(LLVM_TAG    llvmorg-11.1.0)
    set(LLVM_VERSION_DOTTED 11.1.0)
elseif (${LLVM_VERSION} STREQUAL 11_0)
    set(LLVM_TAG    llvmorg-11.0.1)
    set(LLVM_VERSION_DOTTED 11.0.1)
elseif (${LLVM_VERSION} STREQUAL 10_0)
    set(LLVM_TAG    llvmorg-10.0.1)
    set(LLVM_VERSION_DOTTED 10.0.1)
elseif (${LLVM_VERSION} STREQUAL 9_0)
    set(LLVM_TAG    llvmorg-9.0.1)
    set(LLVM_VERSION_DOTTED 9.0.1)
elseif (${LLVM_VERSION} STREQUAL 8_0)
    set(LLVM_TAG    llvmorg-8.0.1)
    set(LLVM_VERSION_DOTTED 8.0.1)
elseif (${LLVM_VERSION} STREQUAL 7_1)
    set(LLVM_TAG    llvmorg-7.1.0)
    set(LLVM_VERSION_DOTTED 7.1.0)
elseif (${LLVM_VERSION} STREQUAL 7_0)
    set(LLVM_TAG    llvmorg-7.0.1)
    set(LLVM_VERSION_DOTTED 7.0.1)
elseif (${LLVM_VERSION} STREQUAL 6_0)
    set(LLVM_TAG    llvmorg-6.0.1)
    set(LLVM_VERSION_DOTTED 6.0.1)
elseif (${LLVM_VERSION} STREQUAL trunk)
    set(LLVM_TAG main)
    set(LLVM_VERSION_DOTTED main)
else()
    message(FATAL_ERROR "Incorrect LLVM version")
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
message(STATUS "PREBUILT_STAGE2 is ${PREBUILT_STAGE2}")
message(STATUS "PREBUILT_STAGE2_PATH is ${PREBUILT_STAGE2_PATH}")
message(STATUS "EXTERNAL_VERBOSE is ${EXTERNAL_VERBOSE}")
message(STATUS "FULL_STAGE2_LLVM is ${FULL_STAGE2_LLVM}")
message(STATUS "STAGE2_TOOLCHAIN_INSTALL_PREFIX is ${STAGE2_TOOLCHAIN_INSTALL_PREFIX}")
message(STATUS "LLVM_DISABLE_ASSERTIONS is ${LLVM_DISABLE_ASSERTIONS}")
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
message(STATUS "XE_DEPS is ${XE_DEPS}")
message(STATUS "LLVM_TAG is ${LLVM_TAG}")
message(STATUS "LLVM_VERSION_DOTTED is ${LLVM_VERSION_DOTTED}")
message(STATUS "SKIP_STAGE2_DEPS is ${SKIP_STAGE2_DEPS}")
message(STATUS "STAGE2_BIN is ${STAGE2_BIN}")
message(STATUS "CMAKE_INSTALL_PREFIX is ${CMAKE_INSTALL_PREFIX}")


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
file(GLOB_RECURSE SUPERBUILD_LLVM_PATCHES_ALL ${PROJECT_SOURCE_DIR}/*.patch)
foreach(patch IN LISTS SUPERBUILD_LLVM_PATCHES_ALL)
    if (${patch} MATCHES ".*${LLVM_VERSION}.*")
        list(APPEND LLVM_PATCHES_VER ${patch})
    endif()
endforeach()
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
    else()
        string(APPEND CCACHE_URL
            https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64.tar.xz)
    endif()

    find_program(CCACHE_EXE NAMES ccache)
    if (NOT CCACHE_EXE)
        if (APPLE)
            message(FATAL_ERROR "No ccache found.")
        endif()
        # Download and unpack ccache executable when no ccache in system.
        FetchContent_Populate(ccache
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
    -DCOMPILER_RT_BUILD_BUILTINS=OFF
    -DCOMPILER_RT_BUILD_CRT=OFF
    -DCOMPILER_RT_CRT_USE_EH_FRAME_REGISTRY=OFF
    -DCOMPILER_RT_BUILD_XRAY=OFF
    -DCOMPILER_RT_BUILD_LIBFUZZER=OFF
    -DCOMPILER_RT_BUILD_MEMPROF=OFF
    -DCOMPILER_RT_BUILD_ORC=OFF
    -DCOMPILER_RT_BUILD_GWP_ASAN=OFF
    )

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
set(LLVM_PROJECTS clang$<SEMICOLON>lld$<SEMICOLON>compiler-rt)
if (NOT WIN32)
    set(LLVM_PROJECTS ${LLVM_PROJECTS}$<SEMICOLON>openmp)
endif()
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
    list(APPEND LLVM_COMMON_CMAKE_ARGS
        -DLLVM_ENABLE_RUNTIMES=libcxx$<SEMICOLON>libcxxabi
        )
endif()

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

    if (NOT WIN32)
        list(APPEND LLVM_STAGE2_INSTALL_TARGETS
            projects/openmp/runtime/install
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
message(STATUS "LLVM stage2 cmake arguments ${LLVM_STAGE2_CMAKE_ARGS}")


# ISPC cmake arguments.
list(APPEND ISPC_COMMON_CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE_ISPC}
    -DISPC_CROSS=ON
    -DXE_ENABLED=ON
    -DISPC_PREPARE_PACKAGE=ON
    )
if (WIN32)
    # XPU examples are not buildable on windows, issue: TODO!
    list(APPEND ISPC_COMMON_CMAKE_ARGS
        -DISPC_GNUWIN32_PATH=${GNUWIN32}
        -DISPC_INCLUDE_EXAMPLES=OFF
        -DISPC_INCLUDE_XE_EXAMPLES=OFF
        )
endif()

list(APPEND ISPC_STAGE2_CMAKE_ARGS
    ${ISPC_COMMON_CMAKE_ARGS}
    -DXE_DEPS_DIR=${STAGE2_PATH}
    -DLEVEL_ZERO_ROOT=${STAGE2_PATH}
    )
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
message(STATUS ISPC_STAGE2_CMAKE_ARGS "ISPC stage2 cmake arguments ${ISPC_STAGE2_CMAKE_ARGS}")


# LLVM stage3 specific cmake arguments, here we imply that PGO always works with LTO.
# PGO specific compiler flags is inside toolchain file.
if (PGO)
    list(APPEND LLVM_STAGE3_CMAKE_ARGS
        ${LLVM_STAGE2_COMMON_CMAKE_ARGS}
        -DLLVM_ENABLE_LTO=Thin
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_BINARY_DIR}/stage3-pgo-toolchain.cmake
        )
    message(STATUS "LLVM stage3 cmake arguments ${LLVM_STAGE3_CMAKE_ARGS}")

    list(APPEND ISPC_STAGE3_CMAKE_ARGS
        ${ISPC_COMMON_CMAKE_ARGS}
        -DXE_DEPS_DIR=${CMAKE_BINARY_DIR}/stage3
        -DLEVEL_ZERO_ROOT=${CMAKE_BINARY_DIR}/stage3
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_BINARY_DIR}/stage3-pgo-toolchain.cmake
        )
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
# MSVC has an outdated clang_rt.profile lib in its SDK.
# We need to enforce explicitly our freshly built library to avoid profile mismatch errors.
set(CLANG_RT_PROFLIB ${STAGE2_PATH}/lib/clang/${LLVM_VERSION_DOTTED}/lib/windows/clang_rt.profile-x86_64.lib)

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
${CCACHE_LAUNCHER_ACTION}(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_EXE})
${CCACHE_LAUNCHER_ACTION}(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_EXE})
if (WIN32)
    set(CMAKE_C_COMPILER   cl)
    set(CMAKE_CXX_COMPILER cl)
    set(CMAKE_ASM_COMPILER ml64)
else()
    set(CMAKE_C_COMPILER   cc)
    set(CMAKE_CXX_COMPILER c++)
endif()
]])

# Base toolchain file for building ISPC and LLVM dependencies.
file(CONFIGURE OUTPUT stage2-toolchain.cmake CONTENT [[
@CCACHE_LAUNCHER_ACTION@(CMAKE_C_COMPILER_LAUNCHER   @CCACHE_EXE@)
@CCACHE_LAUNCHER_ACTION@(CMAKE_CXX_COMPILER_LAUNCHER @CCACHE_EXE@)
set(ENV{PATH} "@ENV_PATH_STAGE2@")
set(WIN_C_FLAGS     "-fuse-ld=lld-link /EHa")
if (WIN32)
    set(CMAKE_C_COMPILER   clang-cl)
    set(CMAKE_CXX_COMPILER clang-cl)
    set(CMAKE_LINKER       lld-link)
    set(CMAKE_AR           llvm-lib)
    set(CMAKE_RC_COMPILER  llvm-rc)
    set(CMAKE_MT           mt)
    set(CMAKE_C_FLAGS      "${WIN_C_FLAGS}")
    set(CMAKE_CXX_FLAGS    "${WIN_C_FLAGS}")
else()
    set(CMAKE_C_COMPILER   clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_LINKER       lld)
endif()
]] @ONLY)

# Toolchain file for LTO build or PGO stage instrumenting LLVM libraries.
file(CONFIGURE OUTPUT stage2-lto-toolchain.cmake CONTENT [[
@CCACHE_LAUNCHER_ACTION@(CMAKE_C_COMPILER_LAUNCHER   @CCACHE_EXE@)
@CCACHE_LAUNCHER_ACTION@(CMAKE_CXX_COMPILER_LAUNCHER @CCACHE_EXE@)
set(ENV{PATH} "@ENV_PATH_STAGE2@")
set(COMMON_FLAGS    "-flto=thin @PGO_FLAGS@")
set(COMMON_C_FLAGS  "")
# Note: -O3 --lto-O3 --icf=...  that these flags lld specific
set(COMMON_LD_FLAGS "-Wl,-O3 -Wl,--lto-O3 -Wl,--icf=all")
set(WIN_C_FLAGS     "-fuse-ld=lld-link /EHa")
set(WIN_LD_FLAGS    "@CLANG_RT_PROFLIB@")
if (WIN32)
    set(CMAKE_C_COMPILER   clang-cl)
    set(CMAKE_CXX_COMPILER clang-cl)
    set(CMAKE_LINKER       lld-link)
    set(CMAKE_AR           llvm-lib)
    set(CMAKE_RC_COMPILER  llvm-rc)
    set(CMAKE_MT           mt)
    set(CMAKE_C_FLAGS             "${COMMON_FLAGS} ${COMMON_C_FLAGS} ${WIN_C_FLAGS}")
    set(CMAKE_CXX_FLAGS           "${COMMON_FLAGS} ${COMMON_C_FLAGS} ${WIN_C_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${WIN_LD_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${WIN_LD_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS    "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${WIN_LD_FLAGS}")
else()
    set(CMAKE_C_COMPILER   clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_LINKER       lld)
    set(CMAKE_C_FLAGS             "${COMMON_FLAGS} ${COMMON_C_FLAGS}")
    set(CMAKE_CXX_FLAGS           "${COMMON_FLAGS} ${COMMON_C_FLAGS}")
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
if (WIN32)
    set(CMAKE_C_COMPILER   clang-cl)
    set(CMAKE_CXX_COMPILER clang-cl)
    set(CMAKE_LINKER       lld-link)
    set(CMAKE_AR           llvm-lib)
    set(CMAKE_RC_COMPILER  llvm-rc)
    set(CMAKE_MT           mt)
    set(CMAKE_C_FLAGS      "${WIN_C_FLAGS}")
    set(CMAKE_CXX_FLAGS    "${WIN_C_FLAGS}")
else()
    set(CMAKE_C_COMPILER   clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_LINKER       lld)
endif()
]] @ONLY)

# Stage3 to optimize with the collected profile.
file(CONFIGURE OUTPUT stage3-pgo-toolchain.cmake CONTENT [[
set(ENV{PATH} "@ENV_PATH_STAGE3@")
set(COMMON_FLAGS    "-flto=thin -fprofile-instr-use=@CODE_PROFDATA@")
set(COMMON_C_FLAGS  "")
# Note: -O3 --lto-O3 --icf=...  that these flags lld specific
set(COMMON_LD_FLAGS "-Wl,-O3 -Wl,--lto-O3 -Wl,--icf=all")
set(WIN_C_FLAGS     "-fuse-ld=lld-link /EHa")
set(WIN_LD_FLAGS    "")
if (WIN32)
    set(CMAKE_C_COMPILER   clang-cl)
    set(CMAKE_CXX_COMPILER clang-cl)
    set(CMAKE_LINKER       lld-link)
    set(CMAKE_AR           llvm-lib)
    set(CMAKE_RC_COMPILER  llvm-rc)
    set(CMAKE_MT           mt)
    set(CMAKE_C_FLAGS             "${COMMON_FLAGS} ${COMMON_C_FLAGS} ${WIN_C_FLAGS}")
    set(CMAKE_CXX_FLAGS           "${COMMON_FLAGS} ${COMMON_C_FLAGS} ${WIN_C_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${WIN_LD_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${WIN_LD_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS    "${COMMON_FLAGS} ${COMMON_LD_FLAGS} ${WIN_LD_FLAGS}")
else()
    set(CMAKE_C_COMPILER   clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_LINKER       lld)
    set(CMAKE_C_FLAGS             "${COMMON_FLAGS} ${COMMON_C_FLAGS}")
    set(CMAKE_CXX_FLAGS           "${COMMON_FLAGS} ${COMMON_C_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${COMMON_FLAGS} ${COMMON_LD_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS    "${COMMON_FLAGS} ${COMMON_LD_FLAGS}")
endif()
]] @ONLY)


# Fetch source code of dependencies. They fetched during cmake configuration run.
# Patch command applies LLVM version specific patches.
if (PGO OR NOT SKIP_STAGE2_DEPS)
    FetchContent_Populate(llvm-source
        GIT_REPOSITORY ${LLVM_URL}
        GIT_TAG ${LLVM_TAG}
        GIT_SHALLOW ON
        GIT_PROGRESS ON
        UPDATE_COMMAND git checkout .
        PATCH_COMMAND git apply ${LLVM_PATCHES_VER}
        SOURCE_DIR llvm-source
        )

    if (XE_DEPS)
        # Checkout level-zero needed for XE
        FetchContent_Populate(l0-source
            GIT_REPOSITORY ${L0_URL}
            GIT_TAG ${L0_TAG}
            GIT_SHALLOW ON
            GIT_PROGRESS ON
            SOURCE_DIR l0-source
            )

        # Checkout LLVM-SPIRV-Translator needed for XE
        FetchContent_Populate(spirv-translator
            GIT_REPOSITORY ${SPIRV_TRANSLATOR_URL}
            GIT_TAG ${SPIRV_TRANSLATOR_SHA}
            GIT_SHALLOW OFF
            GIT_PROGRESS ON
            SOURCE_DIR spirv-translator
            )

        # Checkout vc-intrinsics needed for XE
        FetchContent_Populate(vc-intrinsics
            GIT_REPOSITORY ${VC_INTRINSICS_URL}
            GIT_TAG ${VC_INTRINSICS_SHA}
            GIT_SHALLOW OFF
            GIT_PROGRESS ON
            SOURCE_DIR vc-intrinsics
            UPDATE_COMMAND git checkout .
            PATCH_COMMAND git apply ${PROJECT_SOURCE_DIR}/vc_intrinsics.patch
            )
    endif()

    if (NOT BUILD_STAGE2_TOOLCHAIN_ONLY)
        # May be profiles from examples or less set of programs just enough?
        FetchContent_Populate(ispc-corpus
            GIT_REPOSITORY ${ISPC_CORPUS_URL}
            GIT_TAG main
            GIT_SHALLOW ON
            GIT_PROGRESS ON
            UPDATE_COMMAND git clean -f -d
            SOURCE_DIR ispc-corpus
            )
    endif()
endif()

if (SKIP_STAGE2_DEPS AND BUILD_SPIRV_TRANSLATOR_ONLY)
    # Checkout LLVM-SPIRV-Translator needed for XE
    FetchContent_Populate(spirv-translator
        GIT_REPOSITORY ${SPIRV_TRANSLATOR_URL}
        GIT_TAG ${SPIRV_TRANSLATOR_SHA}
        GIT_SHALLOW OFF
        GIT_PROGRESS ON
        SOURCE_DIR spirv-translator
        )
endif()

if (SKIP_STAGE2_DEPS AND BUILD_VC_INTRINSICS_ONLY)
    # Checkout vc-intrinsics needed for XE
    FetchContent_Populate(vc-intrinsics
        GIT_REPOSITORY ${VC_INTRINSICS_URL}
        GIT_TAG ${VC_INTRINSICS_SHA}
        GIT_SHALLOW OFF
        GIT_PROGRESS ON
        SOURCE_DIR vc-intrinsics
        UPDATE_COMMAND git checkout .
        PATCH_COMMAND git apply ${PROJECT_SOURCE_DIR}/vc_intrinsics.patch
        )
endif()

if (SKIP_STAGE2_DEPS AND BUILD_L0_LOADER_ONLY)
    # Checkout level-zero needed for XE
    FetchContent_Populate(l0-source
        GIT_REPOSITORY ${L0_URL}
        GIT_TAG ${L0_TAG}
        GIT_SHALLOW ON
        GIT_PROGRESS ON
        SOURCE_DIR l0-source
        )
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
    "PATH=${ENV_PATH_STAGE2_ESC_SEM}"
    ${NINJA_EXE}
    ${EXTERNAL_VERBOSE_FLAG}
    ${NINJA_JOBS}
    )
list(APPEND NINJA_COMMAND_WITH_ENV_STAGE3
    ${CMAKE_COMMAND}
    -E
    env
    "CCACHE_CONFIGPATH=${CCACHE_CLANG_V_CONF}"
    "PATH=${ENV_PATH_STAGE3_ESC_SEM}"
    ${NINJA_EXE}
    ${EXTERNAL_VERBOSE_FLAG}
    ${NINJA_JOBS}
    )
message(STATUS "NINJA_COMMAND_WITH_ENV_STAGE1 is ${NINJA_COMMAND_WITH_ENV_STAGE1}")
message(STATUS "NINJA_COMMAND_WITH_ENV_STAGE2 is ${NINJA_COMMAND_WITH_ENV_STAGE2}")
message(STATUS "NINJA_COMMAND_WITH_ENV_STAGE3 is ${NINJA_COMMAND_WITH_ENV_STAGE3}")

if (BUILD_SPIRV_TRANSLATOR_ONLY)
    message(STATUS "BUILD SPIRV-LLVM-Translator only")
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
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_BINARY_DIR}/stage2-toolchain.cmake
            -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        BUILD_IN_SOURCE OFF
        BUILD_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2}
        INSTALL_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2} install
        )
    return()
endif()

if (BUILD_VC_INTRINSICS_ONLY)
    message(STATUS "BUILD vc-intrinsics only")
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
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_BINARY_DIR}/stage2-toolchain.cmake
            -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        BUILD_IN_SOURCE OFF
        BUILD_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2}
        INSTALL_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2} install
        )
    return()
endif()

if (BUILD_L0_LOADER_ONLY)
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
    return()
endif()


# Stage1 job to build bootstrap compiler for building required Clang/LLVM libraries and ISPC itself.
if (PGO OR NOT SKIP_STAGE2_DEPS)
    ExternalProject_Add(stage1
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        SOURCE_DIR llvm-source
        SOURCE_SUBDIR llvm
        PREFIX bstage1
        STAMP_DIR bstage1/s
        BINARY_DIR bstage1/b
        TMP_DIR bstage1/t
        INSTALL_DIR stage2
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS ${LLVM_STAGE1_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/stage2
        BUILD_IN_SOURCE OFF
        BUILD_COMMAND ""
        BUILD_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE1}
        INSTALL_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE1} install
        )
    # Note: We may rebuild instead stage2 clang/lld with stage1 toolchain to
    # generate more performant stage2 compiler.


    list(APPEND L0_DEPS)
    if (XE_DEPS)
        # Level-zero
        ExternalProject_Add(l0
            DEPENDS stage1
            DOWNLOAD_COMMAND ""
            CMAKE_GENERATOR Ninja
            CMAKE_ARGS
                -DCMAKE_TOOLCHAIN_FILE=${CMAKE_BINARY_DIR}/stage2-l0-toolchain.cmake
                -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/stage2
                -DCMAKE_BUILD_TYPE=${BUILD_TYPE_L0}
            SOURCE_DIR l0-source
            BUILD_IN_SOURCE OFF
            PREFIX bl0
            STAMP_DIR bl0/s
            BINARY_DIR bl0/b
            TMP_DIR bl0/t
            INSTALL_DIR stage2
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
        DEPENDS stage1 ${L0_DEPS}
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        SOURCE_DIR llvm-source
        SOURCE_SUBDIR llvm
        PREFIX bstage2
        STAMP_DIR bstage2/s
        BINARY_DIR bstage2/b
        TMP_DIR bstage2/t
        INSTALL_DIR stage2
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS ${LLVM_STAGE2_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/stage2
        BUILD_IN_SOURCE OFF
        BUILD_COMMAND ""
        INSTALL_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2} ${LLVM_STAGE2_INSTALL_TARGETS}
        )


    # Custom target to install stage2 toolchain with libraries in the user-provided directory.
    if (NOT STAGE2_TOOLCHAIN_INSTALL_PREFIX STREQUAL "NO")
        add_custom_target(install-stage2-toolchain
            COMMENT "Install stage2 to ${STAGE2_TOOLCHAIN_INSTALL_PREFIX}"
            COMMAND ${CMAKE_COMMAND} -E copy_directory stage2 ${STAGE2_TOOLCHAIN_INSTALL_PREFIX}
            )
    endif()

    # Custom target to package stage2 tooclhain to use after skipping its repetitive build.
    add_custom_target(package-stage2
        DEPENDS stage2
        COMMENT "Packaging stage2 archive to llvm-stage2-${LLVM_VERSION_DOTTED}.tgz"
        COMMAND ${CMAKE_COMMAND} -E tar cfvz "llvm-stage2-${LLVM_VERSION_DOTTED}.tgz" stage2
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

list(APPEND ISPC_STAGE2_DEPS)
if (SKIP_STAGE2_DEPS)
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
else()
    list(APPEND ISPC_STAGE2_DEPS ${L0_DEPS} stage2)
endif()

if (NOT BUILD_STAGE2_TOOLCHAIN_ONLY)
    # Build ispc first time with stage2 compiler/libraries.
    # For LTO only build, this ispc is final artifact.
    ExternalProject_Add(ispc-stage2
        DEPENDS ${ISPC_STAGE2_DEPS}
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        PREFIX build-ispc-stage2
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/..
        INSTALL_DIR ispc-stage2
        BUILD_IN_SOURCE OFF
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS ${ISPC_STAGE2_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/ispc-stage2
        BUILD_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2}
        INSTALL_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE2} install package
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
            INSTALL_DIR ispc-stage3
            BUILD_IN_SOURCE OFF
            CMAKE_GENERATOR Ninja
            CMAKE_ARGS ${ISPC_STAGE3_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/ispc-stage3
            BUILD_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE3}
            INSTALL_COMMAND ${NINJA_COMMAND_WITH_ENV_STAGE3} install package
            )
    endif()
endif()
