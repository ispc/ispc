## Copyright 2020-2022 Intel Corporation
## SPDX-License-Identifier: BSD-3-Clause

###############################################################################
## Generic ISPC macros/options ################################################
###############################################################################

option(BUILD_GPU "Build GPU code paths?" ON)

set(ISPC_INCLUDE_DIR "")
macro (include_directories_ispc)
  list(APPEND ISPC_INCLUDE_DIR ${ARGN})
endmacro ()

set(ISPC_DEFINITIONS "")
macro (add_definitions_ispc)
  list(APPEND ISPC_DEFINITIONS ${ARGN})
endmacro ()

# We can't get the file dependencies property of the custom targets
# so we use a custom property to propoage this information through
define_property(TARGET PROPERTY ISPC_CUSTOM_DEPENDENCIES
    BRIEF_DOCS "Tracks list of custom target dependencies"
    FULL_DOCS "Tracks list of custom target dependencies"
)

###############################################################################
## CPU specific macros/options ################################################
###############################################################################

## Find ISPC ##
find_program(ISPC_EXECUTABLE ispc HINTS ${ISPC_DIR_HINT} DOC "Path to the ISPC executable.")
if (NOT ISPC_EXECUTABLE)
  message(FATAL_ERROR "Could not find ISPC. Exiting.")
else()
  # Execute "ispc --version" and parse the version
  execute_process(COMMAND ${ISPC_EXECUTABLE} "--version"
                  OUTPUT_VARIABLE ISPC_INFO)
  string(REGEX MATCH "(.*), ([0-9]*\.[0-9]*\.[0-9]*[a-z]*) (.*)" _ ${ISPC_INFO})
  set(ISPC_VERSION ${CMAKE_MATCH_2})
  message(STATUS "Found ISPC v${ISPC_VERSION}: ${ISPC_EXECUTABLE}")
endif()

## ISPC config options ##

option(ISPC_FAST_MATH "enable ISPC fast-math optimizations" OFF)
mark_as_advanced(ISPC_FAST_MATH)

set(ISPC_ADDRESSING 32 CACHE STRING "32 vs 64 bit addressing in ispc")
set_property(CACHE ISPC_ADDRESSING PROPERTY STRINGS 32 64)
mark_as_advanced(ISPC_ADDRESSING)

macro(define_ispc_isa_options ISA_NAME)
  set(ISPC_TARGET_${ISA_NAME} ${ARGV1} CACHE STRING "ispc target used for ${ISA_NAME} ISA")
  set_property(CACHE ISPC_TARGET_${ISA_NAME} PROPERTY STRINGS ${ARGN} NONE)
  #mark_as_advanced(ISPC_TARGET_${ISA_NAME})
endmacro()

# TODO: query ISPC for available targets to be added here
define_ispc_isa_options(SSE4 sse4-i32x4 sse4-i32x8 sse4-i16x8 sse4-i8x16)
define_ispc_isa_options(AVX avx1-i32x8 avx1-i32x4 avx1-i32x16 avx1-i64x4)
define_ispc_isa_options(AVX2 avx2-i32x8 avx2-i32x4 avx2-i32x16 avx2-i64x4)
define_ispc_isa_options(AVX512KNL avx512knl-x16)
define_ispc_isa_options(AVX512SKX avx512skx-x16 avx512skx-x8)

macro(append_ispc_target_list ISA_NAME)
  set(_TARGET_NAME ISPC_TARGET_${ISA_NAME})
  if (NOT ${_TARGET_NAME} STREQUAL "NONE")
    list(APPEND ISPC_TARGET_LIST ${${_TARGET_NAME}})
  endif()
  unset(_TARGET_NAME)
endmacro()

unset(ISPC_TARGET_LIST)
append_ispc_target_list(SSE4)
append_ispc_target_list(AVX)
append_ispc_target_list(AVX2)
append_ispc_target_list(AVX512KNL)
append_ispc_target_list(AVX512SKX)

## Macros ##

macro (ispc_read_dependencies ISPC_DEPENDENCIES_FILE)
  set(ISPC_DEPENDENCIES "")
  if (EXISTS ${ISPC_DEPENDENCIES_FILE})
    file(READ ${ISPC_DEPENDENCIES_FILE} contents)
    string(REPLACE " " ";"     contents "${contents}")
    string(REPLACE ";" "\\\\;" contents "${contents}")
    string(REPLACE "\n" ";"    contents "${contents}")
    foreach(dep ${contents})
      if (EXISTS ${dep})
        set(ISPC_DEPENDENCIES ${ISPC_DEPENDENCIES} ${dep})
      endif (EXISTS ${dep})
    endforeach(dep ${contents})
  endif ()
endmacro()

macro (ispc_compile)
  set(ISPC_ADDITIONAL_ARGS "")
  # Check if CPU target is passed externally
  if (NOT ISPC_TARGET_CPU)
    set(ISPC_TARGETS ${ISPC_TARGET_LIST})
  else()
    set(ISPC_TARGETS ${ISPC_TARGET_CPU})
  endif()

  set(ISPC_TARGET_EXT ${CMAKE_CXX_OUTPUT_EXTENSION})
  string(REPLACE ";" "," ISPC_TARGET_ARGS "${ISPC_TARGETS}")

  set(ISPC_ARCHITECTURE "x86-64")

  set(ISPC_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR})
  include_directories(${ISPC_TARGET_DIR})

  if(ISPC_INCLUDE_DIR)
    string(REPLACE ";" ";-I;" ISPC_INCLUDE_DIR_PARMS "${ISPC_INCLUDE_DIR}")
    set(ISPC_INCLUDE_DIR_PARMS "-I" ${ISPC_INCLUDE_DIR_PARMS})
  endif()

  if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release")
  endif()
  #CAUTION: -O0/1 -g with ispc seg faults
  set(ISPC_FLAGS_DEBUG "-g" CACHE STRING "ISPC Debug flags")
  mark_as_advanced(ISPC_FLAGS_DEBUG)
  set(ISPC_FLAGS_RELEASE "-O3" CACHE STRING "ISPC Release flags")
  mark_as_advanced(ISPC_FLAGS_RELEASE)
  set(ISPC_FLAGS_RELWITHDEBINFO "-O2 -g" CACHE STRING "ISPC Release with Debug symbols flags")
  mark_as_advanced(ISPC_FLAGS_RELWITHDEBINFO)
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(ISPC_OPT_FLAGS ${ISPC_FLAGS_RELEASE})
  elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(ISPC_OPT_FLAGS ${ISPC_FLAGS_DEBUG})
  elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    set(ISPC_OPT_FLAGS ${ISPC_FLAGS_RELWITHDEBINFO})
  else ()
    message(FATAL_ERROR "CMAKE_BUILD_TYPE (${CMAKE_BUILD_TYPE}) allows only the following values: Debug;Release;RelWithDebInfo")
  endif()

  # turn space sparated list into ';' separated list
  string(REPLACE " " ";" ISPC_OPT_FLAGS "${ISPC_OPT_FLAGS}")

  if (NOT WIN32)
    list(APPEND ISPC_ADDITIONAL_ARGS --pic)
  endif()

  if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    list(APPEND ISPC_ADDITIONAL_ARGS --opt=disable-assertions)
  endif()

  if (ISPC_FAST_MATH)
    list(APPEND ISPC_ADDITIONAL_ARGS --opt=fast-math)
  endif()

  set(ISPC_OBJECTS "")

  foreach(src ${ARGN})
    get_filename_component(fname ${src} NAME_WE)
    # The src_relpath will usually be the same as getting the DIRECTORY
    # component of the src file name, but we go through the full path
    # computation path to handle cases where the src name is an absolute
    # file path in some other location, where we need to compute a path to
    get_filename_component(src_dir ${src} ABSOLUTE)
    file(RELATIVE_PATH src_relpath ${CMAKE_CURRENT_LIST_DIR} ${src_dir})
    get_filename_component(src_relpath ${src_relpath} DIRECTORY)
    # Remove any relative paths up from the relative path
    string(REPLACE "../" "_/" src_relpath "${src_relpath}")

    set(outdir "${ISPC_TARGET_DIR}/${src_relpath}")
    set(input ${CMAKE_CURRENT_SOURCE_DIR}/${src})

    set(ISPC_DEPENDENCIES_FILE ${outdir}/${fname}.dev.idep)
    ispc_read_dependencies(${ISPC_DEPENDENCIES_FILE})

    set(results "${outdir}/${fname}.dev${ISPC_TARGET_EXT}")
    # if we have multiple targets add additional object files
    list(LENGTH ISPC_TARGETS NUM_TARGETS)
    if (NUM_TARGETS GREATER 1)
      foreach(target ${ISPC_TARGETS})
        string(REGEX REPLACE "-(i(8|16|32|64))?x(4|8|16|32|64)" "" target ${target})
        string(REPLACE "avx1" "avx" target ${target})
        list(APPEND results "${outdir}/${fname}.dev_${target}${ISPC_TARGET_EXT}")
      endforeach()
    endif()

    add_custom_command(
      OUTPUT ${results} ${outdir}/${fname}_ispc.h
      COMMAND ${CMAKE_COMMAND} -E make_directory ${outdir}
      COMMAND ${ISPC_EXECUTABLE}
        ${ISPC_DEFINITIONS}
        -I ${CMAKE_CURRENT_SOURCE_DIR}
        ${ISPC_INCLUDE_DIR_PARMS}
        --arch=${ISPC_ARCHITECTURE}
        --addressing=${ISPC_ADDRESSING}
        ${ISPC_OPT_FLAGS}
        --target=${ISPC_TARGET_ARGS}
        --woff
        ${ISPC_ADDITIONAL_ARGS}
        -h ${outdir}/${fname}_ispc.h
        -MMM ${ISPC_DEPENDENCIES_FILE}
        -o ${outdir}/${fname}.dev${ISPC_TARGET_EXT}
        ${input}
      DEPENDS ${input} ${ISPC_DEPENDENCIES}
      COMMENT "Building ISPC object ${outdir}/${fname}.dev${ISPC_TARGET_EXT}"
    )

    set(ISPC_OBJECTS ${ISPC_OBJECTS} ${results})
  endforeach()
endmacro()

function(ispc_target_add_sources name)
  ## Split-out C/C++ from ISPC files ##

  set(ISPC_SOURCES "")
  set(OTHER_SOURCES "")

  foreach(src ${ARGN})
    get_filename_component(ext ${src} EXT)
    if (ext STREQUAL ".ispc")
      set(ISPC_SOURCES ${ISPC_SOURCES} ${src})
    else()
      set(OTHER_SOURCES ${OTHER_SOURCES} ${src})
    endif()
  endforeach()

  ## Get existing target definitions and include dirs ##

  # NOTE(jda) - This needs work: BUILD_INTERFACE vs. INSTALL_INTERFACE isn't
  #             handled automatically.

  #get_property(TARGET_DEFINITIONS TARGET ${name} PROPERTY COMPILE_DEFINITIONS)
  #get_property(TARGET_INCLUDES TARGET ${name} PROPERTY INCLUDE_DIRECTORIES)

  #set(ISPC_DEFINITIONS ${TARGET_DEFINITIONS})
  #set(ISPC_INCLUDE_DIR ${TARGET_INCLUDES})

  ## Compile ISPC files ##

  ispc_compile(${ISPC_SOURCES})

  ## Set final sources on target ##

  get_property(TARGET_SOURCES TARGET ${name} PROPERTY SOURCES)
  list(APPEND TARGET_SOURCES ${ISPC_OBJECTS} ${OTHER_SOURCES})
  set_target_properties(${name} PROPERTIES SOURCES "${TARGET_SOURCES}")
endfunction()


###############################################################################
## GPU specific macros/options ################################################
###############################################################################

define_ispc_isa_options(XE gen9-x8 gen9-x16 xelp-x8 xelp-x16)

set(ISPC_XE_ADDITIONAL_ARGS "" CACHE STRING "extra arguments to pass to ISPC for Xe targets")

function (ispc_compile_gpu parent_target output_prefix)
  if(ISPC_INCLUDE_DIR)
    string(REPLACE ";" ";-I;" ISPC_INCLUDE_DIR_PARMS "${ISPC_INCLUDE_DIR}")
    set(ISPC_INCLUDE_DIR_PARMS "-I" ${ISPC_INCLUDE_DIR_PARMS})
  endif()

  # Check If GPU target is passed externally
  if (NOT ISPC_TARGET_XE)
    set(ISPC_TARGET_XE "gen9-x8")
  endif()

  if (NOT ISPC_TARGET_DIR)
    set(ISPC_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  set(ISPC_PROGRAM_COUNT 16)
  if ("${ISPC_TARGET_XE}" STREQUAL "gen9-x8" OR
      "${ISPC_TARGET_XE}" STREQUAL "xelp-x8")
    set(ISPC_PROGRAM_COUNT 8)
  endif()

  set(ISPC_XE_FLAGS_DEBUG "-g" CACHE STRING "ISPC Xe Debug flags")
  mark_as_advanced(ISPC_XE_FLAGS_DEBUG)
  set(ISPC_XE_FLAGS_RELEASE "-O3" CACHE STRING "ISPC Xe Release flags")
  mark_as_advanced(ISPC_XE_FLAGS_RELEASE)
  set(ISPC_XE_FLAGS_RELWITHDEBINFO "-O2 -g" CACHE STRING "ISPC Xe Release with Debug symbols flags")
  mark_as_advanced(ISPC_XE_FLAGS_RELWITHDEBINFO)

  if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release")
  endif()
  if (WIN32 OR "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(ISPC_XE_OPT_FLAGS ${ISPC_XE_FLAGS_RELEASE})
  elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(ISPC_XE_OPT_FLAGS ${ISPC_XE_FLAGS_DEBUG})
  elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    set(ISPC_XE_OPT_FLAGS ${ISPC_XE_FLAGS_RELWITHDEBINFO})
  else ()
    message(FATAL_ERROR "CMAKE_BUILD_TYPE (${CMAKE_BUILD_TYPE}) allows only the following values: Debug;Release;RelWithDebInfo")
  endif()
  # turn space sparated list into ';' separated list
  string(REPLACE " " ";" ISPC_XE_OPT_FLAGS "${ISPC_XE_OPT_FLAGS}")

  # Additional flags passed by user
  if (NOT ISPC_XE_ADDITIONAL_ARGS)
    set(ISPC_XE_ADDITIONAL_ARGS "")
  endif()

  # Output ISPC module format passed by user
  if (NOT ISPC_XE_FORMAT)
    set (ISPC_XE_FORMAT "spv")
  endif()

  if (ISPC_XE_FORMAT STREQUAL "spv")
    set(ISPC_GPU_OUTPUT_OPT "--emit-spirv")
    set(ISPC_GPU_TARGET_NAME ${parent_target}_spv)
  elseif (ISPC_XE_FORMAT STREQUAL "zebin")
    set(ISPC_GPU_OUTPUT_OPT "--emit-zebin")
    set(ISPC_GPU_TARGET_NAME ${parent_target}_bin)
  elseif (ISPC_XE_FORMAT STREQUAL "bc")
    set(ISPC_GPU_OUTPUT_OPT "--emit-llvm")
    set(ISPC_GPU_TARGET_NAME ${parent_target}_bc)
  endif()

  set(ISPC_XE_COMPILE_OUTPUTS "")
  foreach(src ${ARGN})
    get_filename_component(fname ${src} NAME_WE)
    get_filename_component(dir ${src} PATH)
    # The src_relpath will usually be the same as getting the DIRECTORY
    # component of the src file name, but we go through the full path
    # computation path to handle cases where the src name is an absolute
    # file path in some other location, where we need to compute a path to
    get_filename_component(src_dir ${src} ABSOLUTE)
    file(RELATIVE_PATH src_relpath ${CMAKE_CURRENT_LIST_DIR} ${src_dir})
    get_filename_component(src_relpath ${src_relpath} DIRECTORY)
    # Remove any relative paths up from the relative path
    string(REPLACE "../" "_/" src_relpath "${src_relpath}")

    set(outdir "${ISPC_TARGET_DIR}/${src_relpath}")

    set(input ${CMAKE_CURRENT_LIST_DIR}/${dir}/${fname}.ispc)

    set(ISPC_DEPENDENCIES_FILE ${outdir}/${fname}.gpu.dev.idep)
    ispc_read_dependencies(${ISPC_DEPENDENCIES_FILE})

    # We don't do any separate compilation or linking for SPV files right now,
    # so we treat the output spv/bin/bc files as final targets and output to
    # the ISPC_TARGET_DIR root
    if (ISPC_XE_FORMAT STREQUAL "spv")
      set(result "${ISPC_TARGET_DIR}/${output_prefix}${parent_target}.spv")
    elseif (ISPC_XE_FORMAT STREQUAL "zebin")
      set(result "${ISPC_TARGET_DIR}/${output_prefix}${parent_target}.bin")
    elseif (ISPC_XE_FORMAT STREQUAL "bc")
      set(result "${ISPC_TARGET_DIR}/${output_prefix}${parent_target}.bc")
    endif()

    add_custom_command(
      OUTPUT ${result}
      DEPENDS ${input} ${ISPC_DEPENDENCIES}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${outdir}
      COMMAND ${ISPC_EXECUTABLE}
        -I ${CMAKE_CURRENT_SOURCE_DIR}
        ${ISPC_INCLUDE_DIR_PARMS}
        ${ISPC_XE_OPT_FLAGS}
        -DISPC_GPU
        ${ISPC_DEFINITIONS}
        --addressing=64
        --target=${ISPC_TARGET_XE}
        ${ISPC_GPU_OUTPUT_OPT}
        --woff
        ${ISPC_XE_ADDITIONAL_ARGS}
        -o ${result}
        ${input}
        -MMM ${ISPC_DEPENDENCIES_FILE}
      COMMENT "Building ISPC GPU object ${result}"
    )
    set_source_files_properties(${result} PROPERTIES GENERATED true)

    list(APPEND ISPC_XE_COMPILE_OUTPUTS ${result})
  endforeach()

  add_custom_target(${ISPC_GPU_TARGET_NAME} DEPENDS ${ISPC_XE_COMPILE_OUTPUTS} )
  set_target_properties(${ISPC_GPU_TARGET_NAME} PROPERTIES
      ISPC_CUSTOM_DEPENDENCIES ${ISPC_XE_COMPILE_OUTPUTS}
  )

  add_dependencies(${parent_target} ${ISPC_GPU_TARGET_NAME})

  target_compile_definitions(${parent_target}
  PRIVATE
    ISPC_GPU_PROGRAM_COUNT=${ISPC_PROGRAM_COUNT}
  )

  unset(ISPC_PROGRAM_COUNT)
endfunction()

###############################################################################
## Generic kernel compilation #################################################
###############################################################################

function(add_ispc_kernel TARGET_NAME SOURCE PREFIX)
if (WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
  add_library(${TARGET_NAME} SHARED)
  set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "${PREFIX}${TARGET_NAME}")
  if (WIN32)
    target_link_libraries(${TARGET_NAME} PRIVATE msvcrt.lib)
    if (ISPC_BUILD)
       target_link_libraries(${TARGET_NAME} PRIVATE ${ISPCRT_LIB})
    else()
       target_link_libraries(${TARGET_NAME} PRIVATE ispcrt::ispcrt)
    endif()
  endif()
  ispc_target_add_sources(${TARGET_NAME} ${SOURCE})
  if (BUILD_GPU)
    ispc_compile_gpu(${TARGET_NAME} "${PREFIX}" ${SOURCE})
  endif()
endfunction()
