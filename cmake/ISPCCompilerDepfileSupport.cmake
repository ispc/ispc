#
#  Copyright (c) 2026, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

# ISPCCompilerDepfileSupport.cmake
#
# This module enables dependency file (depfile) support for ISPC when using
# the Ninja generator. CMake's built-in Intel-ISPC compiler module only enables
# depfile support for Makefiles generators, which causes Ninja builds to always
# rebuild ISPC files even when they haven't changed.
#
# This fix should be included after project() in CMakeLists.txt files that use
# ISPC as a language with Ninja:
#
#   cmake_minimum_required(VERSION 3.19)
#   project(myproject LANGUAGES CXX ISPC)
#   include(ISPCCompilerDepfileSupport)
#
# This module can be removed once CMake's upstream Intel-ISPC.cmake is fixed
# to support Ninja generators.

if(CMAKE_ISPC_COMPILER_ID STREQUAL "Intel")
  # Enable depfile support for Ninja generator
  # This is the fix for https://github.com/ispc/ispc/issues/XXXX
  if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER)
      AND CMAKE_GENERATOR MATCHES "Ninja")
    # ISPC supports GCC-style dependency files via -M -MT -MF flags
    # CMAKE_DEPFILE_FLAGS_ISPC is already set by CMake's Intel-ISPC.cmake
    # We just need to enable the format and compiler use flags
    set(CMAKE_ISPC_DEPFILE_FORMAT "gcc")
    set(CMAKE_ISPC_DEPENDS_USE_COMPILER TRUE)

    message(STATUS "Enabled ISPC depfile support for Ninja generator")
  endif()
endif()
