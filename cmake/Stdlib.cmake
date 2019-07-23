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
# ispc Stdlib.cmake
#
function(create_stdlib mask outputPath)
    set(output ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/stdlib_mask${mask}_ispc.cpp)
    add_custom_command(
        OUTPUT ${output}
        COMMAND ${CLANG_EXECUTABLE} -E -x c -DISPC_MASK_BITS=${mask} -DISPC=1 -DPI=3.14159265358979
            stdlib.ispc | \"${Python3_EXECUTABLE}\" stdlib2cpp.py mask${mask}
            > ${output}
        DEPENDS stdlib.ispc
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set(${outputPath} ${output} PARENT_SCOPE)
    set_source_files_properties(${outputPath} PROPERTIES GENERATED true)
endfunction()

function(generate_stdlib resultList)
    foreach (m ${ARGN})
        create_stdlib(${m} outputPath)
        list(APPEND tmpList "${outputPath}")
        if(MSVC)
            # Group generated files inside Visual Studio
            source_group("Generated Stdlib" FILES ${outputPath})
        endif()
    endforeach()
    set(${resultList} ${tmpList} PARENT_SCOPE)
endfunction()
