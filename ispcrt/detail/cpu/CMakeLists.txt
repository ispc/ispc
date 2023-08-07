## Copyright 2023 Intel Corporation
## SPDX-License-Identifier: BSD-3-Clause

set(ISPCRT_BUILD_TASK_MODELS "OpenMP;TBB;Threads")

if (ISPCRT_BUILD_TASKING)
    # Set default value for ISPCRT_BUILD_TASK_MODEL if it is not set externally
    if (NOT ISPCRT_BUILD_TASK_MODEL)
        if (WIN32 OR APPLE)
            set (ISPCRT_BUILD_TASK_MODEL "Threads")
        else()
            set (ISPCRT_BUILD_TASK_MODEL "TBB")
        endif()
    endif()

    # Report error if unsupported thread model is requested.
    string(FIND "${ISPCRT_BUILD_TASK_MODELS}" "${ISPCRT_BUILD_TASK_MODEL}" MATCHED_TASK_MODEL)

    if (${MATCHED_TASK_MODEL} EQUAL -1)
        message(FATAL_ERROR "ISPCRT_BUILD_TASK_MODEL (${ISPCRT_BUILD_TASK_MODEL}) allows only the following values: ${ISPCRT_BUILD_TASK_MODELS}")
    endif()

    message(STATUS "ISPC Runtime will be built with support of " ${ISPCRT_BUILD_TASK_MODEL} " tasking model on CPU")

    add_library(ispcrt_tasking INTERFACE)

    if (ISPCRT_BUILD_TASK_MODEL STREQUAL "OpenMP")
        find_package(OpenMP REQUIRED)
        if (OpenMP_FOUND)
            target_link_libraries(ispcrt_tasking INTERFACE OpenMP::OpenMP_CXX)
            target_compile_definitions(ispcrt_tasking INTERFACE ISPC_USE_OMP)
        endif()
    elseif (ISPCRT_BUILD_TASK_MODEL STREQUAL "TBB")
        set (TBB_HEADER "tbb/tbb.h")
        set (TBB_FOUND)
        # If path to TBB is provided by user
        if (TBB_ROOT)
            message(STATUS "TBB root:" ${TBB_ROOT})
            # Try to find TBB library and header files
            find_library(TBB_LIB NAMES libtbb.so.2 tbb PATHS ${TBB_ROOT}/lib NO_DEFAULT_PATH)
            find_path(TBB_INCLUDE_DIR NAMES ${TBB_HEADER} PATHS ${TBB_ROOT}/include NO_DEFAULT_PATH)
            # If TBB library and header files are found, use them.
            if (TBB_LIB AND TBB_INCLUDE_DIR)
                message(STATUS "Found TBB: ${TBB_LIB}")
                message(STATUS "TBB_INCLUDE_DIR is ${TBB_INCLUDE_DIR}")
                target_link_libraries(ispcrt_tasking INTERFACE ${TBB_LIB})
                target_include_directories(ispcrt_tasking INTERFACE ${TBB_INCLUDE_DIR})
                set (TBB_FOUND TRUE)
            endif()
        else()
            # Otherwise find TBB on the system
            find_package(TBB QUIET COMPONENTS tbb)
            if (TBB_FOUND)
                # TBB package does not produce any info when found, so print it here.
                message(STATUS "Found TBB: ${TBB_VERSION} at ${TBB_DIR}")
                target_link_libraries(ispcrt_tasking INTERFACE TBB::tbb)
            else()
                # On some OS (Ubuntu 18.04, CentOS) TBB is installed without CMake support
                # Try to find TBB library and header files
                find_library(TBB_LIB NAMES libtbb.so.2 tbb)
                get_filename_component(TBB_LIB_DIR "${TBB_LIB}" DIRECTORY)
                find_path(TBB_INCLUDE_DIR NAMES ${TBB_HEADER} PATHS ${TBB_LIB_DIR}/../include)
                # If TBB library and header files are found, use them.
                if (TBB_LIB AND TBB_INCLUDE_DIR)
                    message(STATUS "Found TBB: ${TBB_LIB}")
                    message(STATUS "TBB_INCLUDE_DIR is ${TBB_INCLUDE_DIR}")
                    target_link_libraries(ispcrt_tasking INTERFACE ${TBB_LIB})
                    target_include_directories(ispcrt_tasking INTERFACE ${TBB_INCLUDE_DIR})
                    set (TBB_FOUND TRUE)
                endif()
            endif()
        endif()
        if (NOT TBB_FOUND)
            message(FATAL_ERROR "TBB is not found! Please install TBB or set the TBB_ROOT pointing to TBB location")
        endif()
        target_compile_definitions(ispcrt_tasking INTERFACE ISPC_USE_TBB_PARALLEL_FOR)
    else()
        find_package(Threads REQUIRED)
        if (Threads_FOUND)
            # Use default compile definitions from ispc_tasking.cpp
            target_link_libraries(ispcrt_tasking INTERFACE Threads::Threads)
        endif()
    endif()
else()
    message(STATUS "ISPC Runtime will be built without support of tasking model on CPU")
endif()

set(TARGET ${PROJECT_NAME}_device_cpu)
add_library(${TARGET} SHARED
    CPUDevice.cpp
    $<$<BOOL:${ISPCRT_BUILD_TASKING}>:ispc_tasking.cpp>
    )

target_link_libraries(${TARGET} PRIVATE
    $<$<BOOL:${ISPCRT_BUILD_TASKING}>:ispcrt_tasking>
    )
target_include_directories(${TARGET} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>/../../
    )
target_link_libraries(${TARGET} PRIVATE ispcrt_interface_lib)

set_target_properties(${TARGET} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    )

install(TARGETS ${TARGET}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
