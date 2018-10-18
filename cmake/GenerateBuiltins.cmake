#
#  Copyright (c) 2018, Intel Corporation
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
# ispc GenerateBuiltins.cmake
#
find_program(M4_EXECUTABLE m4)
    if (NOT M4_EXECUTABLE)
        message(FATAL_ERROR "Failed to find M4 macro processor" )
    endif()
    message(STATUS "M4 macro processor: " ${M4_EXECUTABLE})

if (WIN32)
    set(OS_NAME "WINDOWS")
elseif (UNIX)
    set(OS_NAME "UNIX")
endif()

function(ll_to_cpp llFileName bit resultFileName)
    set(inputFilePath ${CMAKE_CURRENT_SOURCE_DIR}/builtins/${llFileName}.ll)
    set(includePath ${CMAKE_CURRENT_SOURCE_DIR}/builtins)
    if (WIN32)
        win_path_to_cygwin(${inputFilePath} inputFilePath)
        win_path_to_cygwin(${includePath} includePath)
    endif()
    if ("${bit}" STREQUAL "")
        set(output ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/builtins-${llFileName}.cpp)
        add_custom_command(
            OUTPUT ${output}
            COMMAND ${M4_EXECUTABLE} -I${includePath}
                -DLLVM_VERSION=${LLVM_VERSION} -DBUILD_OS=${OS_NAME} ${inputFilePath}
                | ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/bitcode2cpp.py ${inputFilePath} --llvm_as ${LLVM_AS_EXECUTABLE}
                > ${output}
            DEPENDS ${inputFilePath}
        )
    else ()
        set(output ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/builtins-${llFileName}-${bit}bit.cpp)
        add_custom_command(
            OUTPUT ${output}
            COMMAND ${M4_EXECUTABLE} -I${includePath}
                -DLLVM_VERSION=${LLVM_VERSION} -DBUILD_OS=${OS_NAME} -DRUNTIME=${bit} ${inputFilePath}
                | ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/bitcode2cpp.py ${inputFilePath} ${bit}bit --llvm_as ${LLVM_AS_EXECUTABLE}
                > ${output}
            DEPENDS ${inputFilePath}
        )
    endif()
    set(${resultFileName} ${output} PARENT_SCOPE)
    set_source_files_properties(${resultFileName} PROPERTIES GENERATED true)
endfunction()

function(builtin_to_cpp bit resultFileName)
    set(inputFilePath ${CMAKE_CURRENT_SOURCE_DIR}/builtins/builtins.c)
    set(output ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/builtins-c-${bit}.cpp)
    add_custom_command(
        OUTPUT ${output}
        COMMAND ${CLANG_EXECUTABLE} -m${bit} -emit-llvm -c ${inputFilePath} -o - | \"${LLVM_DIS_EXECUTABLE}\" -
            | ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/bitcode2cpp.py c ${bit} --llvm_as ${LLVM_AS_EXECUTABLE}
            > ${output}
        DEPENDS ${inputFilePath}
        )
    set(${resultFileName} ${output} PARENT_SCOPE)
    set_source_files_properties(${resultFileName} PROPERTIES GENERATED true)
endfunction()

function (generate_target_builtins resultList)
    ll_to_cpp(dispatch "" output)
    list(APPEND tmpList ${output})
    foreach (ispc_target ${ARGN})
        foreach (bit 32 64)
            ll_to_cpp(target-${ispc_target} ${bit} output${bit})
            list(APPEND tmpList ${output${bit}})
        endforeach()
    endforeach()
    set(${resultList} ${tmpList} PARENT_SCOPE)
endfunction()

function (generate_common_builtins resultList)
    foreach (bit 32 64)
        builtin_to_cpp(${bit} res${bit})
        list(APPEND tmpList ${res${bit}} )
    endforeach()
    set(${resultList} ${tmpList} PARENT_SCOPE)
endfunction()