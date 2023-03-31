#
#  Copyright (c) 2022-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

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
