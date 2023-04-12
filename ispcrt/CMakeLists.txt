## Copyright 2020-2023 Intel Corporation
## SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.1)

## Global setup ##

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_STANDARD_REQUIRED   ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

## Establish project ##########################################################

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/Version.cmake)
get_ispc_version("${CMAKE_CURRENT_LIST_DIR}/../common/version.h")
project(ispcrt VERSION ${ISPC_VERSION_MAJOR}.${ISPC_VERSION_MINOR}.${ISPC_VERSION_PATCH} LANGUAGES CXX)

include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

## Add library and executable targets #########################################

option(ISPCRT_BUILD_STATIC "Build ispcrt static library" ON)
option(ISPCRT_BUILD_CPU "Enable CPU support in ispcrt" ON)
option(ISPCRT_BUILD_TASKING "Enable CPU tasking targets in ispcrt" ON)

# Do not build ISPCRT with GPU support on FreeBSD even if XE_ENABLED is ON.
# All dependencies, such as L0 and compute-runtime, are theoretically available
# on FreeBSD, but were never tested. If this switch is enabled, the full recipe to
# build/test the full stack is needed.
if (XE_ENABLED AND NOT (CMAKE_SYSTEM_NAME STREQUAL "FreeBSD"))
  option(ISPCRT_BUILD_GPU "Enable Level0 GPU support in ispcrt" ON)
else()
  option(ISPCRT_BUILD_GPU "Enable Level0 GPU support in ispcrt" OFF)
endif()

if (ISPCRT_BUILD_GPU)
  if (WIN32)
    option(ISPCRT_BUILD_TESTS "Enable ispcrt tests" OFF)
  else()
    option(ISPCRT_BUILD_TESTS "Enable ispcrt tests" ON)
  endif()
endif()

if (NOT ISPCRT_BUILD_CPU AND NOT ISPCRT_BUILD_GPU)
  message(FATAL_ERROR "You must enable either CPU or GPU support!")
endif()


if (WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_subdirectory(detail)

# RPATH to find devices libraries when run mock tests from the build directory.
list(APPEND CMAKE_BUILD_RPATH ${CMAKE_BINARY_DIR}/ispcrt/detail/cpu ${CMAKE_BINARY_DIR}/ispcrt/detail/gpu)
# RPATH for ispcrt library to find devices libraries.
set(CMAKE_INSTALL_RPATH $ORIGIN)

# Shared library target.
list(APPEND LIB_LIST ${PROJECT_NAME})
add_library(${PROJECT_NAME} SHARED
  ispcrt.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../common/version.rc
)
target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})

# Static library target.
if (ISPCRT_BUILD_STATIC)
    list(APPEND LIB_LIST ${PROJECT_NAME}_static)
    add_library(${PROJECT_NAME}_static STATIC
      ispcrt.cpp
      $<$<BOOL:${ISPCRT_BUILD_CPU}>:detail/cpu/CPUDevice.cpp>
      $<$<BOOL:${ISPCRT_BUILD_GPU}>:detail/gpu/GPUDevice.cpp>
      $<$<BOOL:${ISPCRT_BUILD_TASKING}>:detail/cpu/ispc_tasking.cpp>
      ${CMAKE_CURRENT_LIST_DIR}/../common/version.rc
    )
    target_compile_definitions(${PROJECT_NAME}_static PRIVATE ISPCRT_BUILD_STATIC)
    target_include_directories(${PROJECT_NAME}_static PUBLIC
      $<BUILD_INTERFACE:${LEVEL_ZERO_INCLUDE_DIR}>
    )
endif()

foreach(PRNAME ${LIB_LIST})
    set_target_properties(${PRNAME} PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
    )

    target_include_directories(${PRNAME} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../common>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ispcrt>
    )

    if (ISPCRT_BUILD_GPU)
        target_compile_definitions(${PRNAME} PRIVATE ISPCRT_BUILD_GPU)
    endif()

    if (ISPCRT_BUILD_CPU)
        target_compile_definitions(${PRNAME} PRIVATE ISPCRT_BUILD_CPU)
    endif()

    if (ISPCRT_BUILD_TASKING)
        target_compile_definitions(${PRNAME} PRIVATE ISPCRT_BUILD_TASKING)
    endif()

    target_compile_definitions(${PRNAME} PRIVATE ISPCRT_VERSION_MAJOR="${PROJECT_VERSION_MAJOR}")
    target_compile_definitions(${PRNAME} PRIVATE ISPCRT_VERSION_FULL="${PROJECT_VERSION}")


    # Security options
    if (MSVC)
      # Stack canaries
      target_compile_options(${PRNAME} PRIVATE /GS)
      # Control flow guard
      target_link_options(${PRNAME} PRIVATE /GUARD:CF)
    elseif (APPLE)
    else()
      # Enable stack protector
      # NOT to assume that null pointer deference does not exist
      # Assume that signed overflow always wraps
      target_compile_options(${PRNAME} PRIVATE -fstack-protector-strong -fno-delete-null-pointer-checks -fwrapv)
    endif()

    ## Install targets + exports ##

    install(TARGETS ${PRNAME}
      EXPORT ${PROJECT_NAME}_Exports
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        NAMELINK_SKIP
      # on Windows put the dlls into bin
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      # ... and the import lib into the devel package
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    install(TARGETS ${PRNAME}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        NAMELINK_ONLY
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endforeach()

install(EXPORT ${PROJECT_NAME}_Exports
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
  NAMESPACE ${PROJECT_NAME}::
)

if (WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
endif()

## Build tests ############################################################

if (ISPCRT_BUILD_TESTS)
  add_subdirectory(tests)
endif()

## Install headers ############################################################

install(FILES ispcrt.h ispcrt.hpp ispcrt.isph
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ispcrt
)

## Configure CMake find_package() config files ################################

include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
INSTALL_DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
)

write_basic_package_version_file(
    "${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  cmake/Findlevel_zero.cmake
  cmake/Finddpcpp_compiler.cmake
  cmake/ispc.cmake
  cmake/interop.cmake
DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
)
