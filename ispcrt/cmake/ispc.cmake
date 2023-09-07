## Copyright 2020-2023 Intel Corporation
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

# We track DPCPP/ESIMD libraries that an ISPC GPU target links separately to
# make some of the generator expressions easier to write. Instead of having
# to do some list filtering on the single LINK_LIBRARIES property and do some
# name matching heuristics we can just check this property to see if we link
# and DPCPP libraries.
define_property(TARGET PROPERTY ISPC_DPCPP_LINK_LIBRARIES
  BRIEF_DOCS "Tracks list of DPCPP libraries linked by an ISPC GPU target"
  FULL_DOCS "Tracks list of DPCPP libraries linked by an ISPC GPU target"
)

define_property(TARGET PROPERTY ISPC_DPCPP_LINKING_ESIMD
  BRIEF_DOCS "Tracks if the DPCPP libraries linked by the ISPC library are ESIMD"
  FULL_DOCS "Tracks if the DPCPP libraries linked by the ISPC library are ESIMD"
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
  # Execute "ispc --help" and parse supported archs
  execute_process(COMMAND ${ISPC_EXECUTABLE} "--help"
                  OUTPUT_VARIABLE ISPC_HELP)
  string(REGEX MATCH "--arch={((([a-z,0-9,-])+, |([a-z,0-9,-])+)+)}" _ ${ISPC_HELP})
  set(ISPC_ARCHS ${CMAKE_MATCH_1})
  if ("${ISPC_ARCHS}" STREQUAL "")
    message(WARNING "Can't determine ISPC supported architectures.")
  else()
    message(STATUS "ISPC supports: ${ISPC_ARCHS}")
    string(REPLACE ", " ";" ISPC_ARCHS_LIST ${ISPC_ARCHS})
  endif()
endif()

## ISPC config options ##

option(ISPC_FAST_MATH "enable ISPC fast-math optimizations" OFF)
mark_as_advanced(ISPC_FAST_MATH)

set(ISPC_ADDRESSING 32 CACHE STRING "32 vs 64 bit addressing in ispc")
set_property(CACHE ISPC_ADDRESSING PROPERTY STRINGS 32 64)
mark_as_advanced(ISPC_ADDRESSING)

macro(define_ispc_supported_arch ARCH_NAME ARCH_FILTER)
  set(ISPC_ARCHS_${ARCH_NAME}_LIST ${ISPC_ARCHS_LIST})
  list(FILTER ISPC_ARCHS_${ARCH_NAME}_LIST INCLUDE REGEX ${ARCH_FILTER})
  list(LENGTH ISPC_ARCHS_${ARCH_NAME}_LIST ARCH_LENGTH)
  if (${ARCH_LENGTH} GREATER 0)
    set(ISPC_${ARCH_NAME}_ENABLED TRUE)
  endif()
endmacro()

define_ispc_supported_arch(X86 "x86|x86-64")
define_ispc_supported_arch(ARM "arm|aarch64")
define_ispc_supported_arch(XE "xe64")

macro(define_ispc_isa_options ISA_NAME)
  set(ISPC_TARGET_${ISA_NAME} ${ARGV1} CACHE STRING "ispc target used for ${ISA_NAME} ISA")
  set_property(CACHE ISPC_TARGET_${ISA_NAME} PROPERTY STRINGS ${ARGN} NONE)
  #mark_as_advanced(ISPC_TARGET_${ISA_NAME})
endmacro()

if (ISPC_X86_ENABLED)
  define_ispc_isa_options(SSE4 sse4-i32x4 sse4-i32x8 sse4-i16x8 sse4-i8x16)
  define_ispc_isa_options(AVX avx1-i32x8 avx1-i32x4 avx1-i32x16 avx1-i64x4)
  define_ispc_isa_options(AVX2 avx2-i32x8 avx2-i32x4 avx2-i32x16 avx2-i64x4)
  define_ispc_isa_options(AVX512KNL avx512knl-x16)
  define_ispc_isa_options(AVX512SKX avx512skx-x16 avx512skx-x8)
  define_ispc_isa_options(AVX512SPR avx512spr-x16 avx512spr-x8)
endif()

macro(append_ispc_target_list ISA_NAME)
  set(_TARGET_NAME ISPC_TARGET_${ISA_NAME})
  if (NOT ${_TARGET_NAME} STREQUAL "NONE")
    list(APPEND ISPC_TARGET_LIST ${${_TARGET_NAME}})
  endif()
  unset(_TARGET_NAME)
endmacro()

unset(ISPC_TARGET_LIST)
if (ISPC_X86_ENABLED)
  append_ispc_target_list(SSE4)
  append_ispc_target_list(AVX)
  append_ispc_target_list(AVX2)
  append_ispc_target_list(AVX512KNL)
  append_ispc_target_list(AVX512SKX)
  if (NOT APPLE)
    append_ispc_target_list(AVX512SPR)
  endif()
endif()

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
  cmake_parse_arguments(ISPC_COMPILE "" "TARGET" "" ${ARGN})

  set(ISPC_ADDITIONAL_ARGS "")
  # Check if CPU target is passed externally
  if (NOT ISPC_TARGET_CPU)
    set(ISPC_TARGETS ${ISPC_TARGET_LIST})
  else()
    set(ISPC_TARGETS ${ISPC_TARGET_CPU})
  endif()

  set(ISPC_TARGET_EXT ${CMAKE_CXX_OUTPUT_EXTENSION})
  string(REPLACE ";" "," ISPC_TARGET_ARGS "${ISPC_TARGETS}")

  if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64|aarch64")
      set(ISPC_ARCHITECTURE "aarch64")
    else()
      set(ISPC_ARCHITECTURE "x86-64")
    endif()
  else()
    set(ISPC_ARCHITECTURE "x86")
  endif()

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

  # Also set the target-local include directories and defines if
  # we were given a target name
  set(ISPC_INCLUDE_DIRS_EXPR "")
  set(ISPC_COMPILE_DEFINITIONS_EXPR "")
  if (NOT "${ISPC_COMPILE_TARGET}" STREQUAL "")
    set(ISPC_INCLUDE_DIRS_EXPR
      "$<TARGET_PROPERTY:${ISPC_COMPILE_TARGET},INCLUDE_DIRECTORIES>")
    set(ISPC_COMPILE_DEFINITIONS_EXPR
      "$<TARGET_PROPERTY:${ISPC_COMPILE_TARGET},COMPILE_DEFINITIONS>")
  endif()

  set(ISPC_OBJECTS "")
  foreach(src ${ISPC_COMPILE_UNPARSED_ARGUMENTS})
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
        "$<$<BOOL:${ISPC_INCLUDE_DIRS_EXPR}>:-I$<JOIN:${ISPC_INCLUDE_DIRS_EXPR},;-I>>"
        ${ISPC_INCLUDE_DIR_PARMS}
        "$<$<BOOL:${ISPC_COMPILE_DEFINITIONS_EXPR}>:-D$<JOIN:${ISPC_COMPILE_DEFINITIONS_EXPR},;-D>>"
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
      COMMAND_EXPAND_LISTS
      VERBATIM
    )

    set(ISPC_OBJECTS ${ISPC_OBJECTS} ${results})
  endforeach()
endmacro()

###############################################################################
## GPU specific macros/options ################################################
###############################################################################

define_ispc_isa_options(XE gen9-x8 gen9-x16 xelp-x8 xelp-x16 xehpc-x16 xehpc-x32 xehpg-x8 xehpg-x16)

set(ISPC_XE_ADDITIONAL_ARGS "" CACHE STRING "extra arguments to pass to ISPC for Xe targets")

function(ispc_gpu_target_add_sources TARGET_NAME PARENT_TARGET_NAME)
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
    set(TARGET_OUTPUT_EXT "spv")
  elseif (ISPC_XE_FORMAT STREQUAL "zebin")
    set(ISPC_GPU_OUTPUT_OPT "--emit-zebin")
    set(TARGET_OUTPUT_EXT "bin")
  elseif (ISPC_XE_FORMAT STREQUAL "bc")
    set(ISPC_GPU_OUTPUT_OPT "--emit-llvm")
    set(TARGET_OUTPUT_EXT "bc")
  endif()

  # Support old global includes as well
  if (ISPC_INCLUDE_DIR)
    string(REPLACE ";" ";-I;" ISPC_INCLUDE_DIR_PARMS "${ISPC_INCLUDE_DIR}")
    set(ISPC_INCLUDE_DIR_PARMS "-I" ${ISPC_INCLUDE_DIR_PARMS})
  endif()

  set(ISPC_XE_COMPILE_OUTPUTS "")
  set(ISPC_INCLUDE_DIRS_EXPR "$<TARGET_PROPERTY:${TARGET_NAME},INCLUDE_DIRECTORIES>")
  set(ISPC_COMPILE_DEFINITIONS_EXPR "$<TARGET_PROPERTY:${TARGET_NAME},COMPILE_DEFINITIONS>")
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

    if (ISPC_XE_FORMAT STREQUAL "spv")
      # Add a .tmp suffix so we don't conflict with final SPV target name if it's the same
      # as the file
      set(result "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_${fname}.tmp.spv")
    elseif (ISPC_XE_FORMAT STREQUAL "zebin")
      set(result "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_${fname}.tmp.bin")
    elseif (ISPC_XE_FORMAT STREQUAL "bc")
      set(result "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_${fname}.tmp.bc")
    endif()

    add_custom_command(
      OUTPUT ${result}
      DEPENDS ${input} ${ISPC_DEPENDENCIES}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${outdir}
      COMMAND ${ISPC_EXECUTABLE}
        -I ${CMAKE_CURRENT_SOURCE_DIR}
        "$<$<BOOL:${ISPC_INCLUDE_DIRS_EXPR}>:-I$<JOIN:${ISPC_INCLUDE_DIRS_EXPR},;-I>>"
        ${ISPC_INCLUDE_DIR_PARMS}
        ${ISPC_XE_OPT_FLAGS}
        -DISPC_GPU
        "$<$<BOOL:${ISPC_COMPILE_DEFINITIONS_EXPR}>:-D$<JOIN:${ISPC_COMPILE_DEFINITIONS_EXPR},;-D>>"
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
      COMMAND_EXPAND_LISTS
      VERBATIM
    )
    set_source_files_properties(${result} PROPERTIES GENERATED true)

    list(APPEND ISPC_XE_COMPILE_OUTPUTS ${result})
  endforeach()

  add_custom_target(${TARGET_NAME} DEPENDS
    ${ISPC_TARGET_DIR}/${PARENT_TARGET_NAME}.${TARGET_OUTPUT_EXT})
  set_target_properties(${TARGET_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${ISPC_TARGET_DIR}
    LIBRARY_OUTPUT_NAME ${PARENT_TARGET_NAME}.${TARGET_OUTPUT_EXT}
  )

  set(LINK_GPU_LIBRARIES_PROP "$<TARGET_PROPERTY:${TARGET_NAME},LINK_LIBRARIES>")
  set(LINK_DPCPP_LIBRARIES_PROP "$<TARGET_PROPERTY:${TARGET_NAME},ISPC_DPCPP_LINK_LIBRARIES>")

  set(NEEDS_ISPC_LINK_EXPR
      "$<OR:$<BOOL:${LINK_GPU_LIBRARIES_PROP}>,$<BOOL:${LINK_DPCPP_LIBRARIES_PROP}>>")
  set(LINKS_DPCPP_LIBS "$<BOOL:${LINK_DPCPP_LIBRARIES_PROP}>")

  set(LINKS_DPCPP_ESIMD_LIBS
    "$<BOOL:$<TARGET_PROPERTY:${TARGET_NAME},ISPC_DPCPP_LINKING_ESIMD>>")
  set(LINKS_DPCPP_SCALAR_LIBS
    "$<AND:${LINKS_DPCPP_LIBS},$<NOT:${LINKS_DPCPP_ESIMD_LIBS}>>")

  # If we have multiple files we need to link, even if we don't link any libraries.
  list(LENGTH ISPC_XE_COMPILE_OUTPUTS NUM_ISPC_XE_COMPILE_OUTPUTS)
  if (${NUM_ISPC_XE_COMPILE_OUTPUTS} GREATER 1)
    set(NEEDS_ISPC_LINK_EXPR 1)
  endif()

  list(APPEND SYCL_POST_LINK_ARGS
    "-split=auto"
    "-symbols"
    "-lower-esimd"
    "-emit-param-info"
    "-emit-exported-symbols"
    "-spec-const=native"
    "-device-globals"
    "-O2"
  )

  list(APPEND SPV_EXTENSIONS
    "-all"
    "+SPV_EXT_shader_atomic_float_add"
    "+SPV_EXT_shader_atomic_float_min_max"
    "+SPV_KHR_no_integer_wrap_decoration"
    "+SPV_KHR_float_controls"
    "+SPV_INTEL_subgroups"
    "+SPV_INTEL_media_block_io"
    "+SPV_INTEL_fpga_reg"
    "+SPV_INTEL_device_side_avc_motion_estimation"
    "+SPV_INTEL_fpga_loop_controls"
    "+SPV_INTEL_fpga_memory_attributes"
    "+SPV_INTEL_fpga_memory_accesses"
    "+SPV_INTEL_unstructured_loop_controls"
    "+SPV_INTEL_blocking_pipes"
    "+SPV_INTEL_io_pipes"
    "+SPV_INTEL_function_pointers"
    "+SPV_INTEL_kernel_attributes"
    "+SPV_INTEL_float_controls2"
    "+SPV_INTEL_inline_assembly"
    "+SPV_INTEL_optimization_hints"
    "+SPV_INTEL_arbitrary_precision_integers"
    "+SPV_INTEL_vector_compute"
    "+SPV_INTEL_fast_composite"
    "+SPV_INTEL_fpga_buffer_location"
    "+SPV_INTEL_arbitrary_precision_fixed_point"
    "+SPV_INTEL_arbitrary_precision_floating_point"
    "+SPV_INTEL_variable_length_array"
    "+SPV_INTEL_fp_fast_math_mode"
    "+SPV_INTEL_fpga_cluster_attributes"
    "+SPV_INTEL_loop_fuse"
    "+SPV_INTEL_long_constant_composite"
    "+SPV_INTEL_fpga_invocation_pipelining_attributes"
  )
  string(REPLACE ";" "," SPV_EXT_PARMS "${SPV_EXTENSIONS}")

  list(APPEND DPCPP_LLVM_SPIRV_ARGS
    "-spirv-debug-info-version=ocl-100"
    "-spirv-allow-extra-diexpressions"
    "-spirv-allow-unknown-intrinsics=llvm.genx."
    # Not all extenstion are supported yet by VC backend
    # so list here which are supported
    #-spirv-ext=+all
    "-spirv-ext=${SPV_EXT_PARMS}"
  )

  set(TARGET_OUTPUT_FILE "${ISPC_TARGET_DIR}/${PARENT_TARGET_NAME}")
  add_custom_command(
    DEPENDS
      ${ISPC_XE_COMPILE_OUTPUTS}
      ${LINK_GPU_LIBRARIES_PROP}
      ${LINK_DPCPP_LIBRARIES_PROP}
    OUTPUT ${ISPC_TARGET_DIR}/${PARENT_TARGET_NAME}.${TARGET_OUTPUT_EXT}
    COMMAND
      # True case: we're linking and run ispc link. Args we want separate with a space
      # should be separate with a ; here so when the lists are expanded we get the
      # desired space separate in the command
      # False case: we're just going to copy the file with cmake -E copy
      "$<IF:${NEEDS_ISPC_LINK_EXPR},${ISPC_EXECUTABLE};link,cmake;-E;copy>"

      # True case arguments for ispc link
      "$<${NEEDS_ISPC_LINK_EXPR}:${ISPC_XE_COMPILE_OUTPUTS};${LINK_GPU_LIBRARIES_PROP}>"
      "$<${NEEDS_ISPC_LINK_EXPR}:--emit-spirv>"
      "$<${NEEDS_ISPC_LINK_EXPR}:-o>"
      "$<${NEEDS_ISPC_LINK_EXPR}:${TARGET_OUTPUT_FILE}.spv>"

      # False case arguments for cmake -E copy
      # We also pick between zebin/spv suffixes here, zebin and spv are both valid
      # outputs for single-file targets.
      "$<$<NOT:${NEEDS_ISPC_LINK_EXPR}>:${ISPC_XE_COMPILE_OUTPUTS}>"
      "$<$<NOT:${NEEDS_ISPC_LINK_EXPR}>:${TARGET_OUTPUT_FILE}.${TARGET_OUTPUT_EXT}>"

    # For targets doing DPCPP linking we need to do the dpcpp link step against the
    # extracted DPCPP library bitcode and then translate to the final SPV output target
    # First we link the bitcode extracted from DPCPP using DPCPP LLVM link.
    COMMAND
      "$<${LINKS_DPCPP_LIBS}:${DPCPP_LLVM_LINK}>"
      ${LINK_DPCPP_LIBRARIES_PROP}
      "$<${LINKS_DPCPP_LIBS}:-o;${TARGET_OUTPUT_FILE}.linked.dpcpp.bc>"

    # Now we run SYCL post link if we're linking against a scalar DPCPP library
    # ESIMD linking skips this step
    COMMAND
      "$<${LINKS_DPCPP_SCALAR_LIBS}:${DPCPP_SYCL_POST_LINK};${TARGET_OUTPUT_FILE}.linked.dpcpp.bc>"
      "$<${LINKS_DPCPP_SCALAR_LIBS}:${SYCL_POST_LINK_ARGS}>"
      "$<${LINKS_DPCPP_SCALAR_LIBS}:-o;${TARGET_OUTPUT_FILE}.postlink.bc>"

    # Now link with ISPC bitcode with DPCPP extracted and post-processed bitcode.
    COMMAND
      "$<${LINKS_DPCPP_LIBS}:${DPCPP_LLVM_LINK};${result}>"
      "$<${LINKS_DPCPP_SCALAR_LIBS}:${TARGET_OUTPUT_FILE}.postlink_0.bc>"
      "$<${LINKS_DPCPP_ESIMD_LIBS}:${TARGET_OUTPUT_FILE}.linked.dpcpp.bc>"
      "$<${LINKS_DPCPP_LIBS}:-o;${TARGET_OUTPUT_FILE}.linked.bc>"

    # And finally back to SPV to the original expected target SPV name
    COMMAND
      "$<${LINKS_DPCPP_LIBS}:${DPCPP_LLVM_SPIRV}>"
      # Pick the right input to llvm-spirv based on if we're linking scalar or esimd
      # DPCPP libraries.
      "$<${LINKS_DPCPP_SCALAR_LIBS}:${TARGET_OUTPUT_FILE}.linked.bc>"
      "$<${LINKS_DPCPP_ESIMD_LIBS}:${TARGET_OUTPUT_FILE}.linked.bc>"

      "$<${LINKS_DPCPP_LIBS}:${DPCPP_LLVM_SPIRV_ARGS}>"
      "$<${LINKS_DPCPP_LIBS}:-o;${TARGET_OUTPUT_FILE}.spv>"

    COMMENT "Making ISPC library ${TARGET_OUTPUT_FILE}.${TARGET_OUTPUT_EXT}"
    COMMAND_EXPAND_LISTS
    VERBATIM
  )
  unset(ISPC_PROGRAM_COUNT)
endfunction()


###############################################################################
## Generic kernel compilation #################################################
###############################################################################

# Kept for backwards compatibility with existing CPU projects
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


# Compile ISPC files
function(ispc_cpu_target_add_sources TARGET_NAME)
  # We'll only have ISPC files coming through this codepath now
  ispc_compile(${ARGN} TARGET ${TARGET_NAME})

  ## Set final sources on target ##
  get_property(TARGET_SOURCES TARGET ${TARGET_NAME} PROPERTY SOURCES)
  list(APPEND TARGET_SOURCES ${ISPC_OBJECTS})
  set_target_properties(${TARGET_NAME} PROPERTIES SOURCES "${TARGET_SOURCES}")
endfunction()

function(add_ispc_library TARGET_NAME)
  if (WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif()
  add_library(${TARGET_NAME} SHARED)
  set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)

  if (WIN32)
    target_link_libraries(${TARGET_NAME} PRIVATE msvcrt.lib)
    if (ISPC_BUILD)
       target_link_libraries(${TARGET_NAME} PRIVATE ${ISPCRT_LIB})
    else()
       target_link_libraries(${TARGET_NAME} PRIVATE ispcrt::ispcrt)
    endif()
  endif()

  ispc_cpu_target_add_sources(${TARGET_NAME} ${ARGN})
  if (BUILD_GPU)
    ispc_gpu_target_add_sources(${TARGET_NAME}_gpu ${TARGET_NAME} ${ARGN})
    add_dependencies(${TARGET_NAME} ${TARGET_NAME}_gpu)
    # TODO: Do still need to propage up the ISPC program width define
  endif()
endfunction()

function(ispc_target_include_directories TARGET_NAME)
  get_property(ISPC_TARGET_INCLUDE_DIRS TARGET ${TARGET_NAME} PROPERTY INCLUDE_DIRECTORIES)
  # Get the absolute path for each include directory
  foreach (dir ${ARGN})
    get_filename_component(ABS_PATH ${dir} ABSOLUTE)
    list(APPEND ISPC_TARGET_INCLUDE_DIRS ${ABS_PATH})
  endforeach()
  set_target_properties(${TARGET_NAME}
    PROPERTIES INCLUDE_DIRECTORIES "${ISPC_TARGET_INCLUDE_DIRS}")
  if (BUILD_GPU)
    set_target_properties(${TARGET_NAME}_gpu
      PROPERTIES INCLUDE_DIRECTORIES "${ISPC_TARGET_INCLUDE_DIRS}")
  endif()
endfunction()

function(ispc_target_link_libraries TARGET_NAME)
    # For CPU I think we can just link normally
    target_link_libraries(${TARGET_NAME} ${ARGN})
    if (BUILD_GPU)
      get_property(GPU_LINK_LIBRARIES TARGET ${TARGET_NAME}_gpu PROPERTY LINK_LIBRARIES)
      # Get the library file name for each library we want to link
      foreach (lib ${ARGN})
        if (TARGET ${lib}_gpu)
          # TODO: Need to propagate include directories from the targets over as well
          # to match CMake public include behavior
          get_property(LIB_OUTPUT_PATH TARGET ${lib}_gpu PROPERTY LIBRARY_OUTPUT_DIRECTORY)
          get_property(LIB_OUTPUT_NAME TARGET ${lib}_gpu PROPERTY LIBRARY_OUTPUT_NAME)
          list(APPEND GPU_LINK_LIBRARIES ${LIB_OUTPUT_PATH}/${LIB_OUTPUT_NAME})
        endif()
      endforeach()
      set_target_properties(${TARGET_NAME}_gpu
          PROPERTIES LINK_LIBRARIES "${GPU_LINK_LIBRARIES}")
    endif()
endfunction()

function(ispc_target_compile_definitions TARGET_NAME)
  get_property(ISPC_TARGET_COMPILE_DEFINITIONS
    TARGET ${TARGET_NAME}
    PROPERTY COMPILE_DEFINITIONS)
  list(APPEND ISPC_TARGET_COMPILE_DEFINITIONS ${ARGN})
  set_target_properties(${TARGET_NAME}
    PROPERTIES COMPILE_DEFINITIONS "${ISPC_TARGET_COMPILE_DEFINITIONS}")
  if (BUILD_GPU)
    set_target_properties(${TARGET_NAME}_gpu
      PROPERTIES COMPILE_DEFINITIONS "${ISPC_TARGET_COMPILE_DEFINITIONS}")
  endif()
endfunction()

