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
        find_package(TBB REQUIRED COMPONENTS tbb)
        if (TBB_FOUND)
            # TBB package does not produce any info when found, so print it here.
            message(STATUS "Found TBB: ${TBB_VERSION} at ${TBB_DIR}")
            target_link_libraries(ispcrt_tasking INTERFACE TBB::tbb)
            target_compile_definitions(ispcrt_tasking INTERFACE ISPC_USE_TBB_PARALLEL_FOR)
        endif()
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
target_compile_definitions(${TARGET} PRIVATE ISPCRT_BUILD_CPU)
set_target_properties(${TARGET} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    )
target_include_directories(${TARGET} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>/../../
    )

set_target_properties(${TARGET} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    )
install(TARGETS ${TARGET}
    EXPORT ${TARGET}_Exports
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      NAMELINK_SKIP
    # on Windows put the dlls into bin
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    # ... and the import lib into the devel package
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
install(TARGETS ${TARGET}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      NAMELINK_ONLY
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
