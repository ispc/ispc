## Copyright 2020 Intel Corporation
## SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.1)

project(test_ispcrt LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

find_package(ispcrt REQUIRED)

add_executable(${PROJECT_NAME} test_ispcrt.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ispcrt::ispcrt_static)
