#
#  Copyright (c) 2018-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc FixWindowsPath.cmake
#
if (WIN32)
    find_program(CYGPATH_EXECUTABLE cygpath
        PATHS ${CYGWIN_INSTALL_PATH}/bin)
        if (NOT CYGPATH_EXECUTABLE)
            message(WARNING "Failed to find cygpath" )
        endif()
    # To avoid cygwin warnings about dos-style path during VS build
    function (win_path_to_cygwin inPath execPath outPath)
        set(cygwinPath ${inPath})
        # Need to update path only if tool was installed as cygwin package
        if (${execPath} MATCHES ".*cygwin.*")
            if (${CMAKE_GENERATOR} MATCHES "Visual Studio*")
                execute_process(
                    COMMAND ${CYGPATH_EXECUTABLE} -u ${inPath}
                    OUTPUT_VARIABLE cygwinPath
                )
                string(STRIP "${cygwinPath}" cygwinPath)
            endif()
        endif()
        set(${outPath} ${cygwinPath} PARENT_SCOPE)
    endfunction()
endif()
