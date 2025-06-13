#  Copyright (c) 2025, Intel Corporation
#  SPDX-License-Identifier: BSD-3-Clause

set(example_name point_transform_ctypes)
project(${example_name})

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

# Create shared library
add_library(${example_name} SHARED ${ISPC_OBJ})
set_target_properties(${example_name} PROPERTIES LINKER_LANGUAGE CXX)

# Add a target to copy Python file
add_custom_target(point_transform_ctypes_py
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/point_transform.py
        ${CMAKE_BINARY_DIR}/${example_name}/point_transform.py
    COMMENT "Copying point_transform.py example to build directory"
)

add_dependencies(${example_name} point_transform_ctypes_py)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/point_transform_ctypes)
# On Windows, link with appropriate libraries to provide DLL entry point
if(WIN32)
    target_link_libraries(${example_name} PRIVATE msvcrt)
    set_target_properties(${example_name} PROPERTIES
        LINKER_LANGUAGE CXX
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    if (MSVC)
        add_custom_command(TARGET ${example_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                $<TARGET_FILE:${example_name}>
                ${CMAKE_BINARY_DIR}/${example_name}/$<TARGET_FILE_NAME:${example_name}>
            COMMENT "Copying ${example_name} DLL to build directory"
    )
    endif()
endif()

# Installation targets
if (NOT ISPC_PREPARE_PACKAGE)
    install(TARGETS ${example_name} LIBRARY DESTINATION examples/${example_name}
                                    RUNTIME DESTINATION examples/${example_name})
    install(FILES ${ISPC_HEADER} DESTINATION examples/${example_name})
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/point_transform.py DESTINATION examples/${example_name})
endif()
