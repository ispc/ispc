## Copyright 2020-2025, Intel Corporation
## SPDX-License-Identifier: BSD-3-Clause

# Debug: Print initial LEVEL_ZERO_ROOT value
message(STATUS "Findlevel_zero: Initial LEVEL_ZERO_ROOT = '${LEVEL_ZERO_ROOT}'")
message(STATUS "Findlevel_zero: ENV{LEVEL_ZERO_ROOT} = '$ENV{LEVEL_ZERO_ROOT}'")

# Save the user-provided LEVEL_ZERO_ROOT before find_path potentially overwrites it
if(DEFINED LEVEL_ZERO_ROOT)
  set(_LEVEL_ZERO_ROOT_HINT ${LEVEL_ZERO_ROOT})
endif()

find_path(LEVEL_ZERO_ROOT include/level_zero/ze_api.h
  DOC "Root of level_zero installation"
  HINTS ${_LEVEL_ZERO_ROOT_HINT} $ENV{LEVEL_ZERO_ROOT}
  PATHS
    ${PROJECT_SOURCE_DIR}/level_zero
    /opt/level_zero
)

# Debug: Print LEVEL_ZERO_ROOT after find_path
message(STATUS "Findlevel_zero: After find_path, LEVEL_ZERO_ROOT = '${LEVEL_ZERO_ROOT}'")

find_path(LEVEL_ZERO_INCLUDE_DIR level_zero/ze_api.h
  HINTS
    ${LEVEL_ZERO_ROOT}/include
    ${_LEVEL_ZERO_ROOT_HINT}/include
    ${LEVEL_ZERO_ROOT}
    ${_LEVEL_ZERO_ROOT_HINT}
  PATH_SUFFIXES
    include
)

# Debug: Print include dir result and check if file exists
message(STATUS "Findlevel_zero: LEVEL_ZERO_INCLUDE_DIR = '${LEVEL_ZERO_INCLUDE_DIR}'")
message(STATUS "Findlevel_zero: Checking if ${_LEVEL_ZERO_ROOT_HINT}/include/level_zero/ze_api.h exists")
if(EXISTS "${_LEVEL_ZERO_ROOT_HINT}/include/level_zero/ze_api.h")
  message(STATUS "Findlevel_zero: File EXISTS at ${_LEVEL_ZERO_ROOT_HINT}/include/level_zero/ze_api.h")
else()
  message(STATUS "Findlevel_zero: File NOT FOUND at ${_LEVEL_ZERO_ROOT_HINT}/include/level_zero/ze_api.h")
endif()

find_library(LEVEL_ZERO_LIB_LOADER ze_loader
  HINTS
    ${LEVEL_ZERO_ROOT}/x86_64-linux-gnu
    ${LEVEL_ZERO_ROOT}/lib64
    ${LEVEL_ZERO_ROOT}/lib
    ${LEVEL_ZERO_ROOT}/bin
    ${_LEVEL_ZERO_ROOT_HINT}/x86_64-linux-gnu
    ${_LEVEL_ZERO_ROOT_HINT}/lib64
    ${_LEVEL_ZERO_ROOT_HINT}/lib
    ${_LEVEL_ZERO_ROOT_HINT}/bin
  PATH_SUFFIXES
    x86_64-linux-gnu
    lib64
    lib
    bin
)

# Debug: Print library result and check if files exist
message(STATUS "Findlevel_zero: LEVEL_ZERO_LIB_LOADER = '${LEVEL_ZERO_LIB_LOADER}'")
message(STATUS "Findlevel_zero: Checking lib paths:")
if(EXISTS "${_LEVEL_ZERO_ROOT_HINT}/lib/ze_loader.lib")
  message(STATUS "Findlevel_zero: Found ${_LEVEL_ZERO_ROOT_HINT}/lib/ze_loader.lib")
else()
  message(STATUS "Findlevel_zero: NOT found ${_LEVEL_ZERO_ROOT_HINT}/lib/ze_loader.lib")
endif()
if(EXISTS "${_LEVEL_ZERO_ROOT_HINT}/bin/ze_loader.dll")
  message(STATUS "Findlevel_zero: Found ${_LEVEL_ZERO_ROOT_HINT}/bin/ze_loader.dll")
else()
  message(STATUS "Findlevel_zero: NOT found ${_LEVEL_ZERO_ROOT_HINT}/bin/ze_loader.dll")
endif()
# List contents of the root directory to see what's actually there
file(GLOB _L0_ROOT_CONTENTS "${_LEVEL_ZERO_ROOT_HINT}/*")
message(STATUS "Findlevel_zero: Contents of ${_LEVEL_ZERO_ROOT_HINT}: ${_L0_ROOT_CONTENTS}")

set(LEVEL_ZERO_ERROR_MSG
"
Could not find level_zero!
Use LEVEL_ZERO_ROOT to point to your level_zero installation
"
)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(level_zero
  ${LEVEL_ZERO_ERROR_MSG} LEVEL_ZERO_INCLUDE_DIR LEVEL_ZERO_LIB_LOADER)

add_library(level_zero SHARED IMPORTED)
set_target_properties(level_zero PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${LEVEL_ZERO_INCLUDE_DIR}
)
set_target_properties(level_zero PROPERTIES
  IMPORTED_LOCATION ${LEVEL_ZERO_LIB_LOADER}
  IMPORTED_NO_SONAME TRUE
)
