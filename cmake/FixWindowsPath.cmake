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