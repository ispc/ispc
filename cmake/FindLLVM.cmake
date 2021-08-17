#
#  Copyright (c) 2018-2020, Intel Corporation
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
# ispc FindLLVM.cmake
#
find_package(LLVM REQUIRED CONFIG)
    if (NOT LLVM_FOUND )
        message(FATAL_ERROR "LLVM package can't be found. \
                Set CMAKE_PREFIX_PATH variable to LLVM's installation prefix.")
    endif()
    set(LLVM_VERSION "LLVM_${LLVM_VERSION_MAJOR}_${LLVM_VERSION_MINOR}")
    message(STATUS "Found LLVM ${LLVM_VERSION}")

find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config
    PATHS ${LLVM_TOOLS_BINARY_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
    if (NOT LLVM_CONFIG_EXECUTABLE)
        message(FATAL_ERROR "Failed to find llvm-config")
    endif()
    message(STATUS "LLVM_CONFIG_EXECUTABLE: ${LLVM_CONFIG_EXECUTABLE}")

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

if (XE_ENABLED)
    find_program(CMC_EXECUTABLE NAMES cmc
        PATHS ${LLVM_TOOLS_BINARY_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
    if (NOT CMC_EXECUTABLE)
        message(STATUS "Failed to find cmc" )
    endif()
    message(STATUS "CMC_EXECUTABLE: ${CMC_EXECUTABLE}")
    get_filename_component(CM_INSTALL_PATH ${CMC_EXECUTABLE} DIRECTORY)
    set(CM_INSTALL_PATH ${CM_INSTALL_PATH}/..)
    set(CM_INCLUDE_PATH ${CM_INSTALL_PATH}/include)
    if (NOT EXISTS ${CM_INCLUDE_PATH} OR NOT EXISTS ${CM_INCLUDE_PATH}/cm)
        message(STATUS "Cannot find path to CM library headers (CM_INCLUDE_PATH)")
    endif()
    message(STATUS "CM_INCLUDE_PATH: ${CM_INCLUDE_PATH}")
    set(CM_LIBRARY_PATH ${CM_INSTALL_PATH}/lib)
endif()

find_program(LLVM_DIS_EXECUTABLE NAMES llvm-dis
    PATHS ${LLVM_TOOLS_BINARY_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
    if (NOT LLVM_DIS_EXECUTABLE)
        message(FATAL_ERROR "Failed to find llvm-dis" )
    endif()
    message(STATUS "LLVM_DIS_EXECUTABLE: ${LLVM_DIS_EXECUTABLE}")

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
if (NOT ${ASSERTIONS} STREQUAL "ON")
    message(WARNING "LLVM was built without assertions enabled (-DLLVM_ENABLE_ASSERTIONS=OFF). This disables dumps, which are required for ISPC to be fully functional.")
endif()

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
