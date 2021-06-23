## Copyright 2021 Intel Corporation
## SPDX-License-Identifier: BSD-3-Clause

if(DEFINED ENV{SYCL_BUNDLE_ROOT})
    set(DPCPP_DIR "$ENV{SYCL_BUNDLE_ROOT}")
endif()
if ("${DPCPP_DIR}" STREQUAL "")
    message(FATAL_ERROR "You must source dpcpp_compiler/startup.sh before running CMake to setup DPC++ nightly compiler")
endif()

find_path(DPCPP_DIR include/sycl/sycl.hpp
  DOC "Root of DPC++ nightly compiler installation"
  HINTS ${DPCPP_DIR} $ENV{DPCPP_DIR}
)

find_path(DPCPP_INCLUDE_DIR sycl/sycl.hpp
  PATHS
    ${DPCPP_DIR}/include
)

find_library(DPCPP_LIB sycl HINTS ${DPCPP_DIR}/lib)

find_program(DPCPP_COMPILER NAMES clang++
    PATHS ${DPCPP_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
    if (NOT DPCPP_COMPILER)
        message(FATAL_ERROR "Failed to find " ${DPCPP_DIR}/bin/clang++)
    endif()

find_program(DPCPP_CLANG_BUNDLER NAMES clang-offload-bundler
    PATHS ${DPCPP_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
    if (NOT DPCPP_CLANG_BUNDLER)
        message(FATAL_ERROR "Failed to find " ${DPCPP_DIR}/bin/clang-offload-bundler)
    endif()

find_program(DPCPP_SYCL_POST_LINK NAMES sycl-post-link
    PATHS ${DPCPP_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
    if (NOT DPCPP_SYCL_POST_LINK)
        message(FATAL_ERROR "Failed to find " ${DPCPP_DIR}/bin/sycl-post-link)
    endif()

find_program(DPCPP_LLVM_LINK NAMES llvm-link
    PATHS ${DPCPP_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
    if (NOT DPCPP_LLVM_LINK)
        message(FATAL_ERROR "Failed to find " ${DPCPP_DIR}/bin/llvm-link)
    endif()

find_program(DPCPP_LLVM_SPIRV NAMES llvm-spirv
    PATHS ${DPCPP_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
    if (NOT DPCPP_LLVM_SPIRV)
        message(FATAL_ERROR "Failed to find " ${DPCPP_DIR}/bin/llvm-spirv)
    endif()


include(FindPackageHandleStandardArgs)

set(DPCPP_ERROR_MSG
    "
    Could not find DPC++ nightly compiler!
    Use DPCPP_DIR to point to your oneAPI DPC++ compiler installation
    "
)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(dpcpp_compiler
  ${DPCPP_ERROR_MSG} DPCPP_COMPILER)

if (dpcpp_compiler_FOUND)
    set(DPCPP_LIBRARIES ${DPCPP_LIB})
    set(DPCPP_INCLUDE_DIRS ${DPCPP_INCLUDE_DIR})
endif()

mark_as_advanced(DPCPP_LIB DPCPP_INCLUDE_DIR)