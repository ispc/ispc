## Copyright 2021-2023, Intel Corporation
## SPDX-License-Identifier: BSD-3-Clause

# First try to find the DPCPP compiler in the path
find_program(DPCPP_COMPILER NAMES icpx
    PATHS ${DPCPP_DIR}
    PATH_SUFFIXES bin
)

set(FAILURE_REASON "")

if (DPCPP_COMPILER)
    # If dpcpp was in the path we can assume we're looking at a oneAPI
    # release package, since the nightly builds still name the executable clang++
    get_filename_component(DPCPP_DIR ${DPCPP_COMPILER} DIRECTORY)
    get_filename_component(DPCPP_DIR "${DPCPP_DIR}/../" REALPATH)

    find_path(DPCPP_INCLUDE_DIR NAMES sycl.hpp
        PATHS
        ${DPCPP_DIR}/include/sycl/CL/
    )
    if (NOT DPCPP_INCLUDE_DIR)
        set(FAILURE_REASON "Failed to find sycl.hpp under ${DPCPP_DIR}/include/sycl/CL/")
    endif()

    # oneAPI distributions ship the llvm utility binaries under bin-llvm
    # or under bin/compiler (since oneAPI release 2023.3?)
    set(DPCPP_LLVM_BIN_HINT "bin-llvm" "bin/compiler")
else()
    if(DEFINED ENV{SYCL_BUNDLE_ROOT})
        set(DPCPP_DIR "$ENV{SYCL_BUNDLE_ROOT}")
    endif()

    # If dpcpp wasn't in the path and we have SYCL_BUNDLE_ROOT defined,
    # then we can assume we're looking for a nightly build
    find_program(DPCPP_COMPILER NAMES clang++
        PATHS ${DPCPP_DIR}
        PATH_SUFFIXES bin
        NO_DEFAULT_PATH
    )
    if (NOT DPCPP_COMPILER)
        set(FAILURE_REASON "Failed to find clang++ (dpcpp nightly) under ${DPCPP_DIR}/bin/")
    endif()

    find_path(DPCPP_INCLUDE_DIR sycl/sycl.hpp
        PATHS
        ${DPCPP_DIR}/include
    )
    if (NOT DPCPP_INCLUDE_DIR)
        set(FAILURE_REASON "${FAILURE_REASON};Failed to find sycl.hpp under ${DPCPP_DIR}/include/sycl/")
    endif()

    set(DPCPP_LLVM_BIN_HINT "bin")
endif()

# Under windows oneAPI releases sycl, library has version number in name,
# so add extra names with explicit versions.
# TODO! it is needed to be updated to support newer releases.
find_library(DPCPP_LIB
    NAMES sycl sycl7
    HINTS ${DPCPP_DIR}/lib)
if (NOT DPCPP_LIB)
    set(FAILURE_REASON "${FAILURE_REASON};Failed to find sycl library under ${DPCPP_DIR}/lib/")
endif()

# We should use bin if dpcpp is not in path, bin-llvm if it is.
find_program(DPCPP_CLANG_BUNDLER NAMES clang-offload-bundler
    PATHS ${DPCPP_DIR}
    PATH_SUFFIXES ${DPCPP_LLVM_BIN_HINT}
    NO_DEFAULT_PATH
)
if (NOT DPCPP_CLANG_BUNDLER)
    set(FAILURE_REASON "${FAILURE_REASON};Failed to find clang-offload-bundler under ${DPCPP_DIR}/${DPCPP_LLVM_BIN_HINT}")
endif()

# sycl-post-link is always under bin, including in the oneAPI package
find_program(DPCPP_SYCL_POST_LINK NAMES sycl-post-link
    PATHS ${DPCPP_DIR}
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH
)
if (NOT DPCPP_SYCL_POST_LINK)
    set(FAILURE_REASON "${FAILURE_REASON};Failed to find sycl-post-link under ${DPCPP_DIR}/bin")
endif()

find_program(DPCPP_LLVM_LINK NAMES llvm-link
    PATHS ${DPCPP_DIR}
    PATH_SUFFIXES ${DPCPP_LLVM_BIN_HINT}
    NO_DEFAULT_PATH
)
if (NOT DPCPP_LLVM_LINK)
    set(FAILURE_REASON "${FAILURE_REASON};Failed to find llvm-link under ${DPCPP_DIR}/${DPCPP_LLVM_BIN_HINT}")
endif()

find_program(DPCPP_LLVM_SPIRV NAMES llvm-spirv
    PATHS ${DPCPP_DIR}
    PATH_SUFFIXES ${DPCPP_LLVM_BIN_HINT}
    NO_DEFAULT_PATH
)
if (NOT DPCPP_LLVM_SPIRV)
    set(FAILURE_REASON "${FAILURE_REASON};Failed to find llvm-spirv under ${DPCPP_DIR}/${DPCPP_LLVM_BIN_HINT}")
endif()

include(FindPackageHandleStandardArgs)

string(REPLACE ";" "\n    " FAILURE_REASON "${FAILURE_REASON}")

set(DPCPP_ERROR_MSG
    "
    Could NOT find dpcpp_compiler!
    Ensure dpcpp is in your path or use DPCPP_DIR to point to your oneAPI DPC++ compiler
    installation root or nightly build. For nightly builds you can also source startup.sh
    to set SYCL_BUNDLE_ROOT which will be used to find the compiler.
    Failure reason(s):
    ${FAILURE_REASON}
    "
)

find_package_handle_standard_args(dpcpp_compiler
    ${DPCPP_ERROR_MSG}
    DPCPP_COMPILER
    DPCPP_INCLUDE_DIR
    DPCPP_CLANG_BUNDLER
    DPCPP_SYCL_POST_LINK
    DPCPP_LLVM_LINK
    DPCPP_LLVM_SPIRV
)

if (dpcpp_compiler_FOUND)
    set(DPCPP_LIBRARIES ${DPCPP_LIB})
    set(DPCPP_INCLUDE_DIRS ${DPCPP_INCLUDE_DIR})
endif()

mark_as_advanced(
    DPCPP_LIB
    DPCPP_INCLUDE_DIR
    DPCPP_CLANG_BUNDLER
    DPCPP_SYCL_POST_LINK
    DPCPP_LLVM_LINK
    DPCPP_LLVM_SPIRV)
