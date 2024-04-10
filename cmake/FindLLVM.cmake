#
#  Copyright (c) 2018-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc FindLLVM.cmake
#
if (NOT LLVM_CONFIG_EXECUTABLE)
    find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config)
    if (NOT LLVM_CONFIG_EXECUTABLE)
        message(FATAL_ERROR "Failed to find llvm-config")
    endif()
endif()
message(STATUS "LLVM_CONFIG_EXECUTABLE: ${LLVM_CONFIG_EXECUTABLE}")
# It is better to use cmake_path here if we are OK to raise CMake version up to 3.20.
# cmake_path(GET LLVM_CONFIG_EXECUTABLE PARENT_PATH LLVM_TOOLS_BINARY_DIR)
get_filename_component(LLVM_TOOLS_BINARY_DIR "${LLVM_CONFIG_EXECUTABLE}" DIRECTORY)
message(STATUS "LLVM_TOOLS_BINARY_DIR: ${LLVM_TOOLS_BINARY_DIR}")

if (NOT LLVM_DIR)
    # It is better to use cmake_path here if we are OK to raise CMake version up to 3.20.
    # cmake_path(SET LLVM_DIR NORMALIZE "${LLVM_TOOLS_BINARY_DIR}/../lib/cmake/llvm")
    set(LLVM_DIR "${LLVM_TOOLS_BINARY_DIR}/../lib/cmake/llvm")
endif()
message(STATUS "LLVM_DIR is ${LLVM_DIR}")

find_program(CLANG_EXECUTABLE NAMES clang
    PATHS ${LLVM_TOOLS_BINARY_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
    if (NOT CLANG_EXECUTABLE)
        message(FATAL_ERROR "Failed to find clang" )
    endif()
    message(STATUS "CLANG_EXECUTABLE: ${CLANG_EXECUTABLE}")

find_program(CLANGPP_EXECUTABLE NAMES clang++
    PATHS ${LLVM_TOOLS_BINARY_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
    if (NOT CLANGPP_EXECUTABLE)
        message(FATAL_ERROR "Failed to find clang++" )
    endif()
    message(STATUS "CLANGPP_EXECUTABLE: ${CLANGPP_EXECUTABLE}")

find_program(LLVM_AS_EXECUTABLE NAMES llvm-as
    PATHS ${LLVM_TOOLS_BINARY_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
    if (NOT LLVM_AS_EXECUTABLE)
        message(FATAL_ERROR "Failed to find llvm-as" )
    endif()
    message(STATUS "LLVM_AS_EXECUTABLE: ${LLVM_AS_EXECUTABLE}")

if (ISPC_INCLUDE_TESTS)
    find_program(FILE_CHECK_EXECUTABLE NAMES FileCheck
        PATHS ${LLVM_TOOLS_BINARY_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
        if (NOT FILE_CHECK_EXECUTABLE)
            message(FATAL_ERROR "Failed to find FileCheck" )
        endif()
        message(STATUS "FILE_CHECK_EXECUTABLE: ${FILE_CHECK_EXECUTABLE}")
endif()

function(str_to_list inStr outStr)
    string(REPLACE " " ";" tmpOutput "${inStr}")
    set(${outStr} ${tmpOutput} PARENT_SCOPE)
endfunction()

function(run_llvm_config output_var)
    set(command "${LLVM_CONFIG_EXECUTABLE}" ${ARGN})
    execute_process(COMMAND ${command}
        RESULT_VARIABLE exit_code
        OUTPUT_VARIABLE ${output_var}
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    if (NOT ("${exit_code}" EQUAL "0"))
        message(FATAL_ERROR "Failed running ${command}")
    endif()
    set(${output_var} ${${output_var}} PARENT_SCOPE)
endfunction()

if (WIN32)
  # For windows build - need catch CRT flags
  include(${LLVM_DIR}/ChooseMSVCCRT.cmake)
endif()

run_llvm_config(LLVM_VERSION_NUMBER "--version")
message(STATUS "Detected LLVM version: ${LLVM_VERSION_NUMBER}")

run_llvm_config(ASSERTIONS "--assertion-mode")

run_llvm_config(CXX_FLAGS "--cxxflags")
# Check DNDEBUG flag
if (NOT CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
    string(FIND CXX_FLAGS "NDEBUG" NDEBUG_DEF)
    # If LLVM was built without NDEBUG flag remove it from Cmake flags
    if (NOT ${NDEBUG_DEF} GREATER -1)
        foreach (cmake_flags_to_update
            CMAKE_CXX_FLAGS_RELEASE
            CMAKE_C_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL
            CMAKE_C_FLAGS_MINSIZEREL
            CMAKE_CXX_FLAGS_RELWITHDEBINFO
            CMAKE_C_FLAGS_RELWITHDEBINFO)
            string (REGEX REPLACE "(^| )[/-]D *NDEBUG($| )" " " "${cmake_flags_to_update}" "${${cmake_flags_to_update}}")
        endforeach()
    endif()
endif()

run_llvm_config(LLVM_LIBRARY_DIRS "--libdir")
run_llvm_config(LLVM_INCLUDE_DIRS "--includedir")

function(get_llvm_libfiles resultList)
    run_llvm_config(LLVM_LIBS "--libfiles" ${ARGN})
    str_to_list("${LLVM_LIBS}" tmpList)
    set(${resultList} ${tmpList} PARENT_SCOPE)
endfunction()

function(get_llvm_cppflags resultList)
    run_llvm_config(CPP_FLAGS "--cppflags")
    str_to_list("${CPP_FLAGS}" tmpList)
    set(${resultList} ${tmpList} PARENT_SCOPE)
endfunction()
