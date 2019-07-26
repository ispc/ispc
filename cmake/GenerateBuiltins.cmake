#
#  Copyright (c) 2018-2019, Intel Corporation
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
    set(TARGET_OS_LIST "windows" "unix")
elseif (UNIX)
    set(TARGET_OS_LIST "unix")
endif()

function(ll_to_cpp llFileName bit os_name resultFileName)
    set(inputFilePath builtins/${llFileName}.ll)
    set(includePath builtins)
    string(TOUPPER ${os_name} os_name_macro)
    if ("${bit}" STREQUAL "")
        set(output ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/builtins-${llFileName}-${os_name}.cpp)
        add_custom_command(
            OUTPUT ${output}
            COMMAND ${M4_EXECUTABLE} -I${includePath}
                -DLLVM_VERSION=${LLVM_VERSION} -DBUILD_OS=${os_name_macro} ${inputFilePath}
                | \"${Python3_EXECUTABLE}\" bitcode2cpp.py ${inputFilePath} --os=${os_name_macro} --llvm_as ${LLVM_AS_EXECUTABLE}
                > ${output}
            DEPENDS ${inputFilePath}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    else ()
        set(output ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/builtins-${llFileName}-${bit}bit-${os_name}.cpp)
        add_custom_command(
            OUTPUT ${output}
            COMMAND ${M4_EXECUTABLE} -I${includePath}
                -DLLVM_VERSION=${LLVM_VERSION} -DBUILD_OS=${os_name_macro} -DRUNTIME=${bit} ${inputFilePath}
                | \"${Python3_EXECUTABLE}\" bitcode2cpp.py ${inputFilePath} --runtime=${bit} --os=${os_name_macro} --llvm_as ${LLVM_AS_EXECUTABLE}
                > ${output}
            DEPENDS ${inputFilePath}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
    set(${resultFileName} ${output} PARENT_SCOPE)
    set_source_files_properties(${resultFileName} PROPERTIES GENERATED true)
endfunction()

function(builtin_to_cpp bit os_name resultFileName)
    set(inputFilePath builtins/builtins.c)
    set(output ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/builtins-c-${bit}-${os_name}.cpp)
    ## note, that clang will adjust target triple to make 32 bit when -m32 is passed.
    if (${os_name} STREQUAL "windows")
        set(target_flags --target="x86_64-pc-win32")
    else()
        set(target_flags --target="x86_64-unknown-linux-gnu" -fPIC)
    endif()
    string(TOUPPER ${os_name} os_name_macro)
    add_custom_command(
        OUTPUT ${output}
        COMMAND ${CLANG_EXECUTABLE} ${target_flags} -m${bit} -emit-llvm -c ${inputFilePath} -o - | \"${LLVM_DIS_EXECUTABLE}\" -
            | \"${Python3_EXECUTABLE}\" bitcode2cpp.py c --runtime=${bit} --os=${os_name_macro} --llvm_as ${LLVM_AS_EXECUTABLE}
            > ${output}
        DEPENDS ${inputFilePath}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    set(${resultFileName} ${output} PARENT_SCOPE)
    set_source_files_properties(${resultFileName} PROPERTIES GENERATED true)
endfunction()

function (generate_target_builtins resultList)
    foreach (os_name ${TARGET_OS_LIST})
        ll_to_cpp(dispatch "" ${os_name} output${os_name})
        list(APPEND tmpList ${output${os_name}})
        if(MSVC)
            # Group generated files inside Visual Studio
            source_group("Generated Builtins" FILES ${output}${os_name})
        endif()
    endforeach()
    foreach (ispc_target ${ARGN})
        foreach (bit 32 64)
            foreach (os_name ${TARGET_OS_LIST})
                ll_to_cpp(target-${ispc_target} ${bit} ${os_name} output${os_name}${bit})
                list(APPEND tmpList ${output${os_name}${bit}})
                if(MSVC)
                    # Group generated files inside Visual Studio
                    source_group("Generated Builtins" FILES ${output${os_name}${bit}})
                endif()
            endforeach()
        endforeach()
    endforeach()
    set(${resultList} ${tmpList} PARENT_SCOPE)
endfunction()

function (generate_common_builtins resultList)
    foreach (bit 32 64)
        foreach (os_name ${TARGET_OS_LIST})
            builtin_to_cpp(${bit} ${os_name} res${bit}${os_name})
            list(APPEND tmpList ${res${bit}${os_name}} )
            if(MSVC)
                # Group generated files inside Visual Studio
                source_group("Generated Builtins" FILES ${res${bit}${os_name}})
            endif()
        endforeach()
    endforeach()
    set(${resultList} ${tmpList} PARENT_SCOPE)
endfunction()
