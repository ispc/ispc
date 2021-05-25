## Copyright 2020 Intel Corporation
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

###############################################################################
## CPU specific macros/options ################################################
###############################################################################

## Find ISPC ##
find_program(ISPC_EXECUTABLE ispc HINTS ${ISPC_DIR_HINT} DOC "Path to the ISPC executable.")
if (NOT ISPC_EXECUTABLE)
  message(FATAL_ERROR "Could not find ISPC. Exiting.")
else()
  message(STATUS "Found Intel(r) Implicit SPMD Compiler (Intel(r) ISPC): ${ISPC_EXECUTABLE}")
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
define_ispc_isa_options(AVX512KNL avx512knl-i32x16)
define_ispc_isa_options(AVX512SKX avx512skx-i32x16 avx512skx-i32x8)

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

  #CAUTION: -O0/1 -g with ispc seg faults
  set(ISPC_FLAGS_DEBUG "-g" CACHE STRING "ISPC Debug flags")
  mark_as_advanced(ISPC_FLAGS_DEBUG)
  set(ISPC_FLAGS_RELEASE "-O3" CACHE STRING "ISPC Release flags")
  mark_as_advanced(ISPC_FLAGS_RELEASE)
  set(ISPC_FLAGS_RELWITHDEBINFO "-O2 -g" CACHE STRING "ISPC Release with Debug symbols flags")
  mark_as_advanced(ISPC_FLAGS_RELWITHDEBINFO)
  if (WIN32 OR "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(ISPC_OPT_FLAGS ${ISPC_FLAGS_RELEASE})
  elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(ISPC_OPT_FLAGS ${ISPC_FLAGS_DEBUG})
  else()
    set(ISPC_OPT_FLAGS ${ISPC_FLAGS_RELWITHDEBINFO})
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

    set(outdir "${ISPC_TARGET_DIR}/local_ispc")
    set(input ${CMAKE_CURRENT_SOURCE_DIR}/${src})

    set(deps "")
    if (EXISTS ${outdir}/${fname}.dev.idep)
      file(READ ${outdir}/${fname}.dev.idep contents)
      string(REPLACE " " ";"     contents "${contents}")
      string(REPLACE ";" "\\\\;" contents "${contents}")
      string(REPLACE "\n" ";"    contents "${contents}")
      foreach(dep ${contents})
        if (EXISTS ${dep})
          set(deps ${deps} ${dep})
        endif (EXISTS ${dep})
      endforeach(dep ${contents})
    endif ()

    set(results "${outdir}/${fname}.dev${ISPC_TARGET_EXT}")
    # if we have multiple targets add additional object files
    list(LENGTH ISPC_TARGETS NUM_TARGETS)
    if (NUM_TARGETS GREATER 1)
      foreach(target ${ISPC_TARGETS})
        string(REPLACE "-i8x16"  "" target ${target})
        string(REPLACE "-i32x4"  "" target ${target})
        string(REPLACE "-i32x8"  "" target ${target})
        string(REPLACE "-i32x16" "" target ${target})
        string(REPLACE "-i64x4"  "" target ${target})
        string(REPLACE "-i64x8"  "" target ${target})
        string(REPLACE "avx1" "avx" target ${target})
        list(APPEND results "${outdir}/${fname}.dev_${target}${ISPC_TARGET_EXT}")
      endforeach()
    endif()

    add_custom_command(
      OUTPUT ${results} ${ISPC_TARGET_DIR}/${fname}_ispc.h
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
      -h ${ISPC_TARGET_DIR}/${fname}_ispc.h
      -MMM  ${outdir}/${fname}.dev.idep
      -o ${outdir}/${fname}.dev${ISPC_TARGET_EXT}
      ${input}
      DEPENDS ${input} ${deps}
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

find_program(ISPC_EXECUTABLE_GPU ispc-gpu DOC "Path to GEN enabled ISPC.")

define_ispc_isa_options(GEN genx-x8 genx-x16)

set(ISPC_GENX_ADDITIONAL_ARGS "" CACHE STRING "extra arguments to pass to ISPC for GEN targets")

macro (ispc_compile_gpu parent_target output_prefix)
  if(ISPC_INCLUDE_DIR)
    string(REPLACE ";" ";-I;" ISPC_INCLUDE_DIR_PARMS "${ISPC_INCLUDE_DIR}")
    set(ISPC_INCLUDE_DIR_PARMS "-I" ${ISPC_INCLUDE_DIR_PARMS})
  endif()

  # Check If GPU target is passed externally
  if (NOT ISPC_TARGET_GEN)
    set(ISPC_TARGET_GEN "genx-x8")
  endif()

  foreach(src ${ARGN})
    get_filename_component(fname ${src} NAME_WE)
    get_filename_component(dir ${src} PATH)

    message(STATUS "ISPC-GPU source file to be compiled: ${src}")

    set(input ${CMAKE_CURRENT_LIST_DIR}/${dir}/${fname}.ispc)

    if (NOT ISPC_TARGET_DIR)
      set(ISPC_TARGET_DIR ${CMAKE_BINARY_DIR})
    endif()

    set(outdir ${ISPC_TARGET_DIR})

    set(ISPC_PROGRAM_COUNT 16)
    if ("${ISPC_TARGET_GEN}" STREQUAL "genx-x8")
      set(ISPC_PROGRAM_COUNT 8)
    endif()

    set(ISPC_GENX_FLAGS_DEBUG "-g" CACHE STRING "ISPC GENX Debug flags")
    mark_as_advanced(ISPC_GENX_FLAGS_DEBUG)
    set(ISPC_GENX_FLAGS_RELEASE "-O3" CACHE STRING "ISPC GENX Release flags")
    mark_as_advanced(ISPC_GENX_FLAGS_RELEASE)
    set(ISPC_GENX_FLAGS_RELWITHDEBINFO "-O2 -g" CACHE STRING "ISPC GENX Release with Debug symbols flags")
    mark_as_advanced(ISPC_GENX_FLAGS_RELWITHDEBINFO)

    if (WIN32 OR "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
      set(ISPC_GENX_OPT_FLAGS ${ISPC_GENX_FLAGS_RELEASE})
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
      set(ISPC_GENX_OPT_FLAGS ${ISPC_GENX_FLAGS_DEBUG})
    else()
      set(ISPC_GENX_OPT_FLAGS ${ISPC_GENX_FLAGS_RELWITHDEBINFO})
    endif()
    # turn space sparated list into ';' separated list
    string(REPLACE " " ";" ISPC_GENX_OPT_FLAGS "${ISPC_GENX_OPT_FLAGS}")

    # Additional flags passed by user
    if (NOT ISPC_GENX_ADDITIONAL_ARGS)
      set(ISPC_GENX_ADDITIONAL_ARGS "")
    endif()

    # Output ISPC module format passed by user
    if (NOT ISPC_GENX_FORMAT)
      set (ISPC_GENX_FORMAT "spv")
    endif()

    if (ISPC_GENX_FORMAT STREQUAL "spv")
      set(ISPC_GPU_TARGET_NAME ${parent_target}_${fname}_spv)
      set(ISPC_GPU_OUTPUT_OPT "--emit-spirv")
      set(result "${outdir}/${output_prefix}${parent_target}.spv")
    elseif (ISPC_GENX_FORMAT STREQUAL "zebin")
      set(ISPC_GPU_TARGET_NAME ${parent_target}_${fname}_bin)
      set(ISPC_GPU_OUTPUT_OPT "--emit-zebin")
      set(result "${outdir}/${output_prefix}${parent_target}.bin")
    endif()

    add_custom_target(${ISPC_GPU_TARGET_NAME}
      COMMAND ${ISPC_EXECUTABLE_GPU}
        -I ${CMAKE_CURRENT_SOURCE_DIR}
        ${ISPC_INCLUDE_DIR_PARMS}
        ${ISPC_GENX_OPT_FLAGS}
        -DISPC_GPU
        ${ISPC_DEFINITIONS}
        --addressing=64
        --target=${ISPC_TARGET_GEN}
        ${ISPC_GPU_OUTPUT_OPT}
        --woff
        ${ISPC_GENX_ADDITIONAL_ARGS}
        -o ${result}
        ${input}
      COMMENT "Building ISPC GPU object ${result}"
    )

    add_dependencies(${parent_target} ${ISPC_GPU_TARGET_NAME})

    target_compile_definitions(${parent_target}
    PRIVATE
      ISPC_GPU_PROGRAM_COUNT=${ISPC_PROGRAM_COUNT}
    )

    unset(ISPC_PROGRAM_COUNT)
  endforeach()
endmacro()

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
