#  Copyright (c) 2025, Intel Corporation
#  SPDX-License-Identifier: BSD-3-Clause

set(example_name point_transform_nanobind)
project(${example_name})

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python 3.7 COMPONENTS Interpreter Development REQUIRED)

# Try to find nanobind
set(NANOBIND_FOUND FALSE)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE NB_DIR
    RESULT_VARIABLE NB_RESULT
    ERROR_VARIABLE NB_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NB_RESULT EQUAL 0 AND NB_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
    find_package(nanobind CONFIG QUIET)
    if(nanobind_FOUND)
        set(NANOBIND_FOUND TRUE)
    endif()
endif()

# Check if nanobind was found, exit early if not
if(NOT NANOBIND_FOUND)
    message(STATUS "nanobind not found. Skipping the point_transform_nanobind example.")
    message(STATUS "To build this example, please install nanobind with: pip install nanobind")
    return()  # Exit early, stopping processing of this CMakeLists.txt file
endif()

# Set ISPC source file
set(ISPC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/point_transform.ispc)

# Set ISPC output files
set(ISPC_OBJ ${CMAKE_CURRENT_BINARY_DIR}/point_transform.o)
set(ISPC_HEADER ${CMAKE_CURRENT_BINARY_DIR}/point_transform_ispc.h)

# Set ISPC flags
set(ISPC_FLAGS -O2 --target=host)
if (WIN32)
    set(ISPC_FLAGS ${ISPC_FLAGS} --dllexport)
else()
    set(ISPC_FLAGS ${ISPC_FLAGS} --pic)
endif()

# Add custom command to compile ISPC file
add_custom_command(
    OUTPUT ${ISPC_OBJ} ${ISPC_HEADER}
    COMMAND ${ISPC_EXECUTABLE} ${ISPC_FLAGS} ${ISPC_SRC} -o ${ISPC_OBJ} -h ${ISPC_HEADER}
    DEPENDS ${ISPC_SRC}
    COMMENT "Compiling ISPC file: ${ISPC_SRC}"
)

# Make sure the transform_ispc.h file is created before it's needed
add_custom_target(generate_ispc_files
    DEPENDS ${ISPC_OBJ} ${ISPC_HEADER}
)

# Add nanobind extension module
nanobind_add_module(
    ispc_transform
    nanobind_wrapper.cpp
)

# Make sure the ISPC files are generated before building the module
add_dependencies(ispc_transform generate_ispc_files)

# Link with the ISPC object file
target_link_libraries(ispc_transform PRIVATE ${ISPC_OBJ})

# Add include directories to find the ISPC header
target_include_directories(ispc_transform PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# Add a target to copy Python file
add_custom_target(point_transform_nanobind_py
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/point_transform.py
        ${CMAKE_BINARY_DIR}/point_transform_nanobind/point_transform.py
    COMMENT "Copying point_transform.py example to build directory"
)

add_dependencies(ispc_transform point_transform_nanobind_py)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/point_transform_nanobind)
# On Windows, link with appropriate libraries to provide DLL entry point
if(WIN32)
    target_link_libraries(ispc_transform PRIVATE msvcrt)
    set_target_properties(ispc_transform PROPERTIES
        LINKER_LANGUAGE CXX
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    if (MSVC)
        add_custom_command(TARGET ispc_transform POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                $<TARGET_FILE:ispc_transform>
                ${CMAKE_BINARY_DIR}/point_transform_nanobind/$<TARGET_FILE_NAME:ispc_transform>
            COMMENT "Copying ispc_transform DLL to build directory"
    )
    endif()
endif()

# Installation targets
if (NOT ISPC_PREPARE_PACKAGE)
    install(TARGETS ispc_transform LIBRARY DESTINATION examples/point_transform_nanobind
                                   RUNTIME DESTINATION examples/point_transform_nanobind)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/point_transform.py DESTINATION examples/point_transform_nanobind)
endif()