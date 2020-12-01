## Copyright 2020 Intel Corporation
## SPDX-License-Identifier: BSD-3-Clause

# set the project name
project(ispcrt_mock_tests)

# add the executable
add_executable(ispcrt_mock_tests ispcrt_mock_main.cpp)

link_directories(${CMAKE_BINARY_DIR})

target_link_libraries(ispcrt_mock_tests PUBLIC gtest_main ispcrt ze_null)
target_include_directories(ispcrt_mock_tests PUBLIC ${CMAKE_SOURCE_DIR}/ispcrt/tests/level_zero_mock)

install(
    TARGETS ispcrt_mock_tests
    RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/tests)