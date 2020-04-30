## Copyright 2020 Intel Corporation
## SPDX-License-Identifier: BSD-3-Clause

find_path(LEVEL_ZERO_ROOT include/level_zero/ze_api.h
  DOC "Root of level_zero installation"
  HINTS ${LEVEL_ZERO_ROOT} $ENV{LEVEL_ZERO_ROOT}
  PATHS
    ${PROJECT_SOURCE_DIR}/level_zero
    /opt/level_zero
)

find_path(LEVEL_ZERO_INCLUDE_DIR level_zero/ze_api.h
  PATHS
    ${LEVEL_ZERO_ROOT}/include
)

find_library(LEVEL_ZERO_LIB_LOADER ze_loader HINTS ${LEVEL_ZERO_ROOT}/lib)

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
