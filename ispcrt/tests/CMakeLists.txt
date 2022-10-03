## Copyright 2020-2022 Intel Corporation
## SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.13)

set(ISPCRT_TESTS_PROJECT_NAME ispcrt_tests)
project(${ISPCRT_TESTS_PROJECT_NAME}
        DESCRIPTION "ISPCRT tests"
        LANGUAGES CXX)

#################### Add Google Test to the project #####################

# Tests depend on Google Test submodule
if (NOT GIT_BINARY)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/Git.cmake)
endif()
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/google/googletest/CMakeLists.txt")
    message(STATUS "Google Test submodule update")
    execute_process(COMMAND ${GIT_BINARY} submodule update --init --recursive
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Build google test (target: gtest)
# Do not install testing lib, we don't need it, as we run them from build directory with correct rpath embedded.
set(INSTALL_GTEST OFF CACHE BOOL "Suppressing gtest install" FORCE)
# Link shared library
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared gtest lib" FORCE)
add_subdirectory(vendor/google/googletest)

# Warn if build type is not release.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # NDEBUG may be excluded from build options globally, if LLVM build doesn't have that.
    # But building Google Test, we need it to let it know that optimized build is used.
    target_compile_definitions(gtest PRIVATE NDEBUG)
endif()
# This is WA for https://github.com/google/googletest/issues/1305
target_compile_options(gmock PRIVATE -Wno-deprecated-copy)

####################### Configure and add submodules  ########################

# Meta target for tests
add_custom_target(${ISPCRT_TESTS_PROJECT_NAME})

# Level Zero API mock library
add_subdirectory(level_zero_mock)
# Tests using Level Zero mock library
add_subdirectory(mock_tests)

# Install gtest libraries
install(
    TARGETS gtest gtest_main
    LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/tests)