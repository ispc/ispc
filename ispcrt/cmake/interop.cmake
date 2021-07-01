## Copyright 2021 Intel Corporation
## SPDX-License-Identifier: BSD-3-Clause

###############################################################################
## Definititions for ISPC/ESIMD interoperability ##############################
###############################################################################

# This module contains helper functions allowing to link several modules on
# LLVM IR level and translate final module to SPIR-V format.
# For ISPC the workflow is easy:
#   1. compile to .bc
# This bitcode file is ready to be linked with others and translated to .spv
# For DPC++:
#   1. compile source to object file using dpc++ compiler
#   2. extract ESIMD bitcode using clang-offload-bundler
#   3. lower extracted bitcode to real VC backend intrinsics using sycl-post-link
# Lowered bitcode file can be linked with ISPC bitcode using llvm-link and
# translated then to .spv with llvm-spirv.
# Note that we can extract bitcode during dpc++ compilation in one step
# instead of 1. and 2. But we assume that the most frequent case will be to link
# with some library which contains precompiled objects. So getting bitcode is split
# to two steps for more flexibility.

set(COMMON_FLAGS_DEBUG "-g" CACHE STRING "Debug flags")
mark_as_advanced(COMMON_FLAGS_DEBUG)
set(COMMON_FLAGS_RELEASE "-O3" CACHE STRING "Release flags")
mark_as_advanced(COMMON_FLAGS_RELEASE)
set(COMMON_FLAGS_RELWITHDEBINFO "-O2 -g" CACHE STRING "Release with Debug symbols flags")
mark_as_advanced(COMMON_FLAGS_RELWITHDEBINFO)
if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release")
endif()
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(COMMON_OPT_FLAGS ${COMMON_FLAGS_RELEASE})
elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(COMMON_OPT_FLAGS ${COMMON_FLAGS_DEBUG})
elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    set(COMMON_OPT_FLAGS ${COMMON_FLAGS_RELWITHDEBINFO})
else ()
    message(FATAL_ERROR "CMAKE_BUILD_TYPE (${CMAKE_BUILD_TYPE}) allows only the following values: Debug;Release;RelWithDebInfo")
endif()


# Compile dpcpp source to object
# parent_target: parent target to set dependency on
# dpcpp_source: dpcpp source file name
# output_name: output parameter - name of resulting object file
function (dpcpp_compile_source parent_target dpcpp_source output_name)
    get_filename_component(fname ${dpcpp_source} NAME_WE)

    # If input path is not absolute, prepend ${CMAKE_CURRENT_LIST_DIR}
    if(NOT IS_ABSOLUTE ${dpcpp_source})
        set(input ${CMAKE_CURRENT_LIST_DIR}/${dpcpp_source})
    else()
        set(input ${dpcpp_source})
    endif()

    set(outdir ${CMAKE_CURRENT_BINARY_DIR})
    set(result "${outdir}/${fname}.o")

    if(DPCPP_ESIMD_INCLUDE_DIR)
        string(REPLACE ";" ";-I;" DPCPP_ESIMD_INCLUDE_DIR_PARMS "${DPCPP_ESIMD_INCLUDE_DIR}")
        set(DPCPP_ESIMD_INCLUDE_DIR_PARMS "-I" ${DPCPP_ESIMD_INCLUDE_DIR_PARMS})
    endif()

    if (NOT DPCPP_ESIMD_FLAGS)
        set (DPCPP_ESIMD_FLAGS "")
    endif()

    add_custom_command(
        OUTPUT ${result}
        DEPENDS ${input}
        COMMAND ${DPCPP_COMPILER}
            -fsycl
            -fPIE
            -c
            ${COMMON_OPT_FLAGS}
            ${DPCPP_ESIMD_INCLUDE_DIR_PARMS}
            ${DPCPP_ESIMD_FLAGS}
            -I ${CMAKE_CURRENT_SOURCE_DIR}
            -o ${result}
            ${input}
        COMMENT "Building DPCPP object ${input}"
    )
    set_source_files_properties(${result} PROPERTIES GENERATED true)

    # Wrapper custom target to add dependency to the parent target
    set(dpcpp_compile_target_name ${parent_target}_${dpcpp_source}_obj)
    add_custom_target(${dpcpp_compile_target_name} DEPENDS ${result})
    add_dependencies(${parent_target} ${dpcpp_compile_target_name})
    # Pass path to resulting dpcpp object to the parent target
    set(${output_name} ${result} PARENT_SCOPE)
endfunction()

# Extrect esimd bitcode
# parent_target: parent target to set dependency on
# dpcpp_obj: dpcpp object file name
# output_name: output parameter - name of resulting bitcode file
function (dpcpp_get_esimd_bitcode parent_target dpcpp_obj output_name)
    get_filename_component(fname ${dpcpp_obj} NAME_WE)

    # If input path is not absolute, prepend ${CMAKE_CURRENT_LIST_DIR}
    if(NOT IS_ABSOLUTE ${dpcpp_obj})
        set(input ${CMAKE_CURRENT_LIST_DIR}/${dpcpp_obj})
    else()
        set(input ${dpcpp_obj})
    endif()

    set(outdir ${CMAKE_CURRENT_BINARY_DIR})
    set(bundler_result "${outdir}/${fname}.bc")

    add_custom_command(
        OUTPUT ${bundler_result}
        DEPENDS ${input}
        COMMAND ${DPCPP_CLANG_BUNDLER}
            --inputs=${input}
            --unbundle
            --targets=sycl-spir64-unknown-unknown-sycldevice
            --type=o
            --outputs=${bundler_result}
        COMMENT "Extracting ESIMD Bitcode ${bundler_result}"
    )
    set_source_files_properties(${bundler_result} PROPERTIES GENERATED true)

    # This is intermediate bitcode file with link to the real one
    set(lower_post_link "${outdir}/${fname}_lower.bc")
    # This is final esimd bitcode file
    set(post_link_result "${outdir}/${fname}_lower_esimd_0.bc")

    add_custom_command(
        OUTPUT ${post_link_result}
        DEPENDS ${bundler_result}
        BYPRODUCTS ${lower_post_link}
        COMMAND ${DPCPP_SYCL_POST_LINK}
            -split=auto
            -symbols
            -split-esimd
            -lower-esimd
            -spec-const=rt
            -o ${lower_post_link}
            ${bundler_result}
        COMMENT "Lowering ESIMD ${post_link_result}"
    )
    set_source_files_properties(${post_link_result} PROPERTIES GENERATED true)

    # Wrapper custom target to add dependency to the parent target
    set(dpcpp_esimd_target_name ${fname}_bc)
    add_custom_target(${dpcpp_esimd_target_name} DEPENDS ${post_link_result})
    add_dependencies(${parent_target} ${dpcpp_esimd_target_name})

    # Pass path to resulting esimd bitcode to the parent target
    set(${output_name} ${post_link_result} PARENT_SCOPE)
endfunction()

# Link bitcode files into one
# parent_target: parent target to set dependency on
# output_name: output parameter - name of resulting bitcode file. It will have
# suffix _ispc2esimd
# ARGN: names of bitcode file to link
function (link_bitcode parent_target output_name)
    # Join all bitcode inputs to one string
    set(input "")
    foreach(src ${ARGN})
        # If input path is not absolute, prepend ${CMAKE_CURRENT_LIST_DIR}
        if(NOT IS_ABSOLUTE ${src})
            list(APPEND input "${CMAKE_CURRENT_LIST_DIR}/${src}")
        else()
            list(APPEND input "${src}")
        endif()
    endforeach()
    set(outdir ${CMAKE_CURRENT_BINARY_DIR})
    set(result "${outdir}/${parent_target}_ispc2esimd.bc")

    add_custom_command(
        OUTPUT ${result}
        DEPENDS ${input}
        COMMAND ${DPCPP_LLVM_LINK}
            ${input}
            -o ${result}
        COMMENT "Linking LLVM Bitcode ${result}"
    )
    set_source_files_properties(${result} PROPERTIES GENERATED true)

    # Wrapper custom target to add dependency to the parent target
    set(llvm_link_target_name ${parent_target}_ispc2esimd_bc)
    add_custom_target(${llvm_link_target_name} DEPENDS ${result})
    add_dependencies(${parent_target} ${llvm_link_target_name})

    # Pass path to resulting linked bitcode file to the parent target
    set(${output_name} ${result} PARENT_SCOPE)
endfunction()

# Link bitcode files into one
# parent_target: parent target to set dependency on
# bc_input: name of bitcode fiel to translate
function (translate_to_spirv parent_target bc_input)
    get_filename_component(fname ${bc_input} NAME_WE)

    # If input path is not absolute, prepend ${CMAKE_CURRENT_LIST_DIR}
    if(NOT IS_ABSOLUTE ${bc_input})
        set(input ${CMAKE_CURRENT_LIST_DIR}/${bc_input})
    else()
        set(input ${bc_input})
    endif()

    set(outdir ${CMAKE_CURRENT_BINARY_DIR})
    set(result "${outdir}/${fname}.spv")

    list(APPEND SPV_EXT "-all"
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
                        "+SPV_INTEL_fpga_invocation_pipelining_attributes")
    string(REPLACE ";" "," SPV_EXT_PARMS "${SPV_EXT}")
    add_custom_command(
        OUTPUT ${result}
        DEPENDS ${input}
        COMMAND ${DPCPP_LLVM_SPIRV}
            ${input}
            -o ${result}
            -spirv-debug-info-version=ocl-100
            -spirv-allow-extra-diexpressions
            -spirv-allow-unknown-intrinsics=llvm.genx.
            # Not all extenstion are supported yet by VC backend
            # so list here which are supported
            #-spirv-ext=+all
            -spirv-ext=${SPV_EXT_PARMS}
        COMMENT "Translating LLVM Bitcode to SPIR-V ${result}"
    )
    set_source_files_properties(${result} PROPERTIES GENERATED true)

    # Wrapper custom target to add dependency to the parent target
    set(llvm_spirv_target_name ${parent_target}_${fname}_spv)
    add_custom_target(${llvm_spirv_target_name} DEPENDS ${result})
    add_dependencies(${parent_target} ${llvm_spirv_target_name})
endfunction()

# Link ISPC and ESIMD GPU modules to SPIR-V and produce libraries required for CPU
# parent_target: parent target to set dependency on
# ARGN: list of files (ISPC source, DPCPP source, DPCPP objects) to link
function (link_ispc_esimd parent_target)
    if (NOT BUILD_GPU)
        message(FATAL_ERROR "Linking of ISPC/ESIMD modules is supported on GPU only")
    endif()

    if (WIN32)
        message(FATAL_ERROR "Linking of ISPC/ESIMD modules is currently supported on Linux only")
    endif()

    set(ISPC_OUTPUTS "")
    set(DPCPP_OUTPUTS "")

    set(ISPC_GENX_FORMAT "bc")
    set(ISPC_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR})

    foreach(src ${ARGN})
        get_filename_component(ext ${src} LAST_EXT)
        # ISPC file
        if (ext STREQUAL ".ispc")
            ispc_compile_gpu(${parent_target} "ispc_" ispc_bc ${src})
            list(APPEND ISPC_OUTPUTS ${ispc_bc})
        # DPCPP source
        elseif (ext STREQUAL ".cpp")
            dpcpp_compile_source(${parent_target} ${src} dpcpp_obj)
            dpcpp_get_esimd_bitcode(${parent_target} ${dpcpp_obj} esimd_bc)
            list(APPEND DPCPP_OUTPUTS ${esimd_bc})
        # Precompiled DPCPP object
        elseif (ext STREQUAL ".o")
            dpcpp_get_esimd_bitcode(${parent_target} ${src} esimd_bc)
            list(APPEND DPCPP_OUTPUTS ${esimd_bc})
        endif()
    endforeach()

    link_bitcode(${parent_target} ispc2esimd_bc ${ISPC_OUTPUTS} ${DPCPP_OUTPUTS})

    translate_to_spirv(${parent_target} ${ispc2esimd_bc})

    unset(ISPC_GENX_FORMAT)
    unset(ISPC_TARGET_DIR)

endfunction()