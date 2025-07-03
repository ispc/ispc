#
#  Copyright (c) 2025, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

# ispcConfig.cmake

# Compute the installation prefix relative to this file
get_filename_component(ISPC_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(ISPC_PREFIX "${ISPC_CMAKE_DIR}" PATH)
get_filename_component(ISPC_PREFIX "${ISPC_PREFIX}" PATH)
get_filename_component(ISPC_PREFIX "${ISPC_PREFIX}" PATH)

# Set up paths
set(ISPC_INCLUDE_DIRS "${ISPC_PREFIX}/include")
set(ISPC_LIBRARIES "${ISPC_PREFIX}/lib")
set(ISPC_EXECUTABLE "${ISPC_PREFIX}/bin/ispc")

# Platform-specific library names
if(WIN32)
    set(ISPC_LIBRARY_NAME "ispc.dll")
    set(ISPC_IMPORT_LIBRARY "${ISPC_LIBRARIES}/ispc.lib")
else()
    set(ISPC_LIBRARY_NAME "libispc.so")
endif()

# Full path to the library
set(ISPC_LIBRARY "${ISPC_LIBRARIES}/${ISPC_LIBRARY_NAME}")

# Verify that the files exist
set(_ispc_required_files 
    "${ISPC_EXECUTABLE}"
    "${ISPC_INCLUDE_DIRS}/ispc/compiler.h"
    "${ISPC_LIBRARY}"
)

foreach(_file ${_ispc_required_files})
    if(NOT EXISTS "${_file}")
        set(ispc_FOUND FALSE)
        set(ispc_NOT_FOUND_MESSAGE "ISPC installation appears to be broken: missing ${_file}")
        return()
    endif()
endforeach()

# Create imported target for the library
if(NOT TARGET ispc::lib)
    add_library(ispc::lib SHARED IMPORTED)
    
    # Set target properties
    set_target_properties(ispc::lib PROPERTIES
        IMPORTED_LOCATION "${ISPC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ISPC_INCLUDE_DIRS}"
    )
    
    # Windows-specific: set import library
    if(WIN32 AND EXISTS "${ISPC_IMPORT_LIBRARY}")
        set_target_properties(ispc::lib PROPERTIES
            IMPORTED_IMPLIB "${ISPC_IMPORT_LIBRARY}"
        )
    endif()
endif()

# Create executable target
if(NOT TARGET ispc::ispc)
    add_executable(ispc::ispc IMPORTED)
    set_target_properties(ispc::ispc PROPERTIES
        IMPORTED_LOCATION "${ISPC_EXECUTABLE}"
    )
endif()

# Set standard variables
set(ISPC_FOUND TRUE)

# Provide summary
if(NOT ispc_FIND_QUIETLY)
    message(STATUS "Found ISPC:")
    message(STATUS "  Version: ${ispc_VERSION}")
    message(STATUS "  Executable: ${ISPC_EXECUTABLE}")
    message(STATUS "  Library: ${ISPC_LIBRARY}")
    message(STATUS "  Include dirs: ${ISPC_INCLUDE_DIRS}")
endif()