## Copyright 2021-2023 Intel Corporation
## SPDX-License-Identifier: BSD-3-Clause

###############################################################################
## Definititions for ISPC/ESIMD and ISPC/DPC++ interoperability ###############
###############################################################################

# This module contains helper functions allowing to link several modules on
# LLVM IR level and translate final module to SPIR-V format.
# For ISPC and DPC++ SYCL the workflow is easy:
#   1. compile to bitcode
# This bitcode file is ready to be linked with others and translated to .spv.
# For DPC++ ESIMD:
#   1. extract ESIMD bitcode using clang-offload-bundler from DPC++ library
#   2. lower extracted bitcode to real VC backend intrinsics using sycl-post-link
# Lowered bitcode file can be linked with ISPC bitcode using llvm-link and
# translated then to .spv with llvm-spirv.

# Find DPCPP compiler
set(OLD_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
find_package(dpcpp_compiler REQUIRED)
set(CMAKE_MODULE_PATH ${OLD_CMAKE_MODULE_PATH})
unset(OLD_CMAKE_MODULE_PATH)

# Create DPCPP library or SPIR-V file
# target_name: name of the target to use for the created library
# ARGN: DPCPP source files
function (add_dpcpp_library target_name)
    cmake_parse_arguments(PARSE_ARGV 1 DPCPP "SPV" "" "")
    set(outdir ${CMAKE_CURRENT_BINARY_DIR})

    set(DPCPP_CXX_FLAGS "")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(DPCPP_CXX_FLAGS "-O3")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(DPCPP_CXX_FLAGS "-g")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
        set(DPCPP_CXX_FLAGS "-O2" "-g")
    else()
        message(FATAL_ERROR "add_dpcpp_library only supports Debug;Release;RelWithDebInfo build configs")
    endif()

    # Compile each DPCPP file
    set(DPCPP_RESULTS "")
    foreach(src ${DPCPP_UNPARSED_ARGUMENTS})
        get_filename_component(fname ${src} NAME_WE)

        # If input path is not absolute, prepend ${CMAKE_CURRENT_LIST_DIR}
        if(NOT IS_ABSOLUTE ${src})
            set(input ${CMAKE_CURRENT_LIST_DIR}/${src})
        else()
            set(input ${src})
        endif()
        set (TARGET_OUTPUT_FILE "${outdir}/${fname}")
        if (DPCPP_SPV)
            set(TARGET_OUTPUT_FILE_RESULT "${TARGET_OUTPUT_FILE}.spv")
        else()
            set(TARGET_OUTPUT_FILE_RESULT "${TARGET_OUTPUT_FILE}.o")
        endif()

        if(DPCPP_CUSTOM_INCLUDE_DIR)
            string(REPLACE ";" ";-I;" DPCPP_CUSTOM_INCLUDE_DIR_PARMS "${DPCPP_CUSTOM_INCLUDE_DIR}")
            set(DPCPP_CUSTOM_INCLUDE_DIR_PARMS "-I" ${DPCPP_CUSTOM_INCLUDE_DIR_PARMS})
        endif()

        # Allow function pointers in DPC++ and do not instrument SYCL code.
        list(APPEND DPCPP_CUSTOM_FLAGS "-Xclang" "-fsycl-allow-func-ptr" "-fno-sycl-instrument-device-code")
        if (DPCPP_SPV)
            # Get only SYCL device code to bicode first
            # WA: SYCL assert implementation should be treated separately.
            # Disable usage of asserts for now with "-DSYCL_DISABLE_FALLBACK_ASSERT=1"
            list(APPEND DPCPP_CUSTOM_FLAGS "-fsycl-device-only" "-DSYCL_DISABLE_FALLBACK_ASSERT=1")
        else()
            list(APPEND DPCPP_CUSTOM_FLAGS "-c")
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
            "-spirv-ext=${SPV_EXT_PARMS}"
        )

        add_custom_command(
            DEPENDS ${input}
            OUTPUT "${TARGET_OUTPUT_FILE_RESULT}"
            COMMAND ${DPCPP_COMPILER}
                -fsycl
                -fPIE
                ${DPCPP_CXX_FLAGS}
                ${DPCPP_CUSTOM_INCLUDE_DIR_PARMS}
                ${DPCPP_CUSTOM_FLAGS}
                -I ${CMAKE_CURRENT_SOURCE_DIR}
                -o "$<IF:$<BOOL:${DPCPP_SPV}>,${TARGET_OUTPUT_FILE}.bc,${TARGET_OUTPUT_FILE}.o>"
                ${input}

            # Run sycl-post-link on it
            COMMAND
            "$<$<BOOL:${DPCPP_SPV}>:${DPCPP_SYCL_POST_LINK}>"
            "$<$<BOOL:${DPCPP_SPV}>:${SYCL_POST_LINK_ARGS};${TARGET_OUTPUT_FILE}.bc>"
            "$<$<BOOL:${DPCPP_SPV}>:-o;${TARGET_OUTPUT_FILE}.postlink.bc>"

            # And finally back to SPV to the original expected target SPV name
            COMMAND
            "$<$<BOOL:${DPCPP_SPV}>:${DPCPP_LLVM_SPIRV}>"
            # Pick the right input to llvm-spirv based on if we're linking scalar or esimd
            # DPCPP libraries.
            "$<$<BOOL:${DPCPP_SPV}>:${TARGET_OUTPUT_FILE}.postlink_0.bc>"
            "$<$<BOOL:${DPCPP_SPV}>:${DPCPP_LLVM_SPIRV_ARGS}>"
            "$<$<BOOL:${DPCPP_SPV}>:-o;${TARGET_OUTPUT_FILE}.spv>"
            COMMENT "Building DPCPP object ${TARGET_OUTPUT_FILE_RESULT}"
            COMMAND_EXPAND_LISTS
            VERBATIM
        )

        list(APPEND DPCPP_RESULTS ${TARGET_OUTPUT_FILE_RESULT})
    endforeach()

    if (DPCPP_SPV)
        add_custom_target(${target_name} DEPENDS ${DPCPP_RESULTS})
    else()
        add_library(${target_name} STATIC)
        set_target_properties(${target_name} PROPERTIES
            LINKER_LANGUAGE CXX
            SOURCES "${DPCPP_RESULTS}")
    endif()
endfunction()

# Extract esimd bitcode
# target_name: name of the target to use for the extracted bitcode. The bitcode
#              file output path will be set as the LIBRARY_OUTPUT_DIRECTORY property
#              and the file name set to the LIBRARY_OUTPUT_NAME property
# library: the library to extract bitcode from
function (dpcpp_get_esimd_bitcode target_name library)
    set(outdir ${CMAKE_CURRENT_BINARY_DIR})
    # Result after unbundle command
    set(bundler_result_tmp "${outdir}/${target_name}.out")
    # Result after bitcode linking of unbundled output
    set(bundler_result "${outdir}/${target_name}.bc")
    # Intermediate bitcode file with link to the real one
    set(lower_post_link "${outdir}/${target_name}_lower.bc")
    # Final esimd bitcode file
    set(post_link_result "${outdir}/${target_name}_lower_esimd_0.bc")

    add_custom_command(
        DEPENDS ${library}
        OUTPUT ${lower_post_link} ${post_link_result}
        COMMAND ${DPCPP_CLANG_BUNDLER}
            --input=$<TARGET_FILE:${library}>
            --unbundle
            --targets=sycl-spir64-unknown-unknown-sycldevice
            --type=a
            --output=${bundler_result_tmp}
        COMMAND ${DPCPP_LLVM_LINK}
            ${bundler_result_tmp}
            -o ${bundler_result}
        COMMAND ${DPCPP_SYCL_POST_LINK}
            -split=auto
            -symbols
            -lower-esimd
            -emit-param-info
            -emit-exported-symbols
            -spec-const=native
            -device-globals
            -O2
            -o ${lower_post_link}
            ${bundler_result}
        COMMENT "Extracting ESIMD Bitcode ${bundler_result_tmp}"
    )
    add_custom_target(${target_name} DEPENDS ${post_link_result})
    set_target_properties(${target_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${outdir}
        LIBRARY_OUTPUT_NAME ${target_name}_lower_esimd_0.bc
    )
endfunction()

# Extract dpcpp sycl bitcode
# target_name: name of the target to use for the extracted bitcode. The bitcode
#              file output path will be set as the LIBRARY_OUTPUT_DIRECTORY property
#              and the file name set to the LIBRARY_OUTPUT_NAME property
# library: the library to extract bitcode from
function (dpcpp_get_sycl_bitcode target_name library)
    set(outdir ${CMAKE_CURRENT_BINARY_DIR})
    # Result after unbundle command
    set(bundler_result_tmp "${outdir}/${target_name}.out")
    # Result after bitcode linking of unbundled output
    set(bundler_result "${outdir}/${target_name}.bc")

    add_custom_command(
        DEPENDS ${library}
        OUTPUT ${bundler_result}
        COMMAND ${DPCPP_CLANG_BUNDLER}
            --input=$<TARGET_FILE:${library}>
            --unbundle
            --targets=sycl-spir64-unknown-unknown-sycldevice
            --type=a
            --output=${bundler_result_tmp}
        COMMAND ${DPCPP_LLVM_LINK}
            ${bundler_result_tmp}
            -o ${bundler_result}
        COMMENT "Extracting DPC++ Bitcode ${bundler_result}"
    )

    add_custom_target(${target_name} DEPENDS ${bundler_result})
    set_target_properties(${target_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${outdir}
        LIBRARY_OUTPUT_NAME ${target_name}.bc
    )
endfunction()

function(ispc_target_link_dpcpp_libraries TARGET_NAME)
    if (NOT BUILD_GPU)
        message(FATAL_ERROR "Linking of ISPC/DPC++ modules is supported on GPU only")
    endif()
    get_property(GPU_LINK_LIBRARIES TARGET ${TARGET_NAME}_gpu PROPERTY ISPC_DPCPP_LINK_LIBRARIES)
    # Get the library file name for each library we want to link
    foreach (lib ${ARGN})
      # Make the dpcpp_bc target to extract the bitcode so we can link it
      dpcpp_get_sycl_bitcode(${lib}_dpcpp_bc ${lib})

      get_property(LIB_OUTPUT_PATH TARGET ${lib}_dpcpp_bc PROPERTY LIBRARY_OUTPUT_DIRECTORY)
      get_property(LIB_OUTPUT_NAME TARGET ${lib}_dpcpp_bc PROPERTY LIBRARY_OUTPUT_NAME)
      list(APPEND GPU_LINK_LIBRARIES ${LIB_OUTPUT_PATH}/${LIB_OUTPUT_NAME})
    endforeach()
    set_target_properties(${TARGET_NAME}_gpu
      PROPERTIES ISPC_DPCPP_LINK_LIBRARIES "${GPU_LINK_LIBRARIES}")
endfunction()

function(ispc_target_link_dpcpp_esimd_libraries TARGET_NAME)
    if (NOT BUILD_GPU)
        message(FATAL_ERROR "Linking of ISPC/DPC++ modules is supported on GPU only")
    endif()
    get_property(GPU_LINK_LIBRARIES TARGET ${TARGET_NAME}_gpu PROPERTY ISPC_DPCPP_LINK_LIBRARIES)
    # Get the library file name for each library we want to link
    foreach (lib ${ARGN})
      # Make the dpcpp_bc target to extract the bitcode so we can link it
      dpcpp_get_esimd_bitcode(${lib}_dpcpp_esimd_bc ${lib})

      get_property(LIB_OUTPUT_PATH TARGET ${lib}_dpcpp_esimd_bc PROPERTY LIBRARY_OUTPUT_DIRECTORY)
      get_property(LIB_OUTPUT_NAME TARGET ${lib}_dpcpp_esimd_bc PROPERTY LIBRARY_OUTPUT_NAME)
      list(APPEND GPU_LINK_LIBRARIES ${LIB_OUTPUT_PATH}/${LIB_OUTPUT_NAME})
    endforeach()
    set_target_properties(${TARGET_NAME}_gpu PROPERTIES
      ISPC_DPCPP_LINK_LIBRARIES "${GPU_LINK_LIBRARIES}"
      ISPC_DPCPP_LINKING_ESIMD 1
     )
endfunction()
