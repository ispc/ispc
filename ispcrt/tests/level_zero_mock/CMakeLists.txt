## Copyright 2020 Intel Corporation
## SPDX-License-Identifier: BSD-3-Clause

set(TARGET_NAME ze_null)

add_library(${TARGET_NAME}
    SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/ze_mock.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ze_mock.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ze_mock_driver.cpp
)

set_target_properties(${TARGET_NAME} PROPERTIES
    # VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    VERSION "1.0.0"
    SOVERSION "1"
)

target_include_directories(${TARGET_NAME}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${LEVEL_ZERO_INCLUDE_DIR}
)

install(
    TARGETS ${TARGET_NAME}
    LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/tests)