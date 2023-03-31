#
#  Copyright (c) 2018-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

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
