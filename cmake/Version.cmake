#
#  Copyright (c) 2022, Intel Corporation
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
# ispc Version.cmake
#
# Get ispc version
function(get_ispc_version VERSION_FILE)
    file(READ ${VERSION_FILE} ispc_ver)
    string(REGEX MATCH "ISPC_VERSION \"([0-9]*)\.([0-9]*)\.([0-9]*)([a-z]*)" _ ${ispc_ver})
    set(ISPC_VERSION_MAJOR ${CMAKE_MATCH_1} PARENT_SCOPE)
    set(ISPC_VERSION_MINOR ${CMAKE_MATCH_2} PARENT_SCOPE)
    set(ISPC_VERSION_PATCH ${CMAKE_MATCH_3} PARENT_SCOPE)
    set(ISPC_VERSION_SUFFIX ${CMAKE_MATCH_4} PARENT_SCOPE)
    if (${CMAKE_MATCH_4} MATCHES ".*dev")
        set (ISPC_DOC_REPO_TAG "main" PARENT_SCOPE)
    else()
        set (ISPC_DOC_REPO_TAG "v${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}${CMAKE_MATCH_4}" PARENT_SCOPE)
    endif()
endfunction()