## Copyright 2021 Intel Corporation
## SPDX-License-Identifier: BSD-3-Clause

###############################################################################
## Definititions for ISPC/ESIMD interoperability ##############################
###############################################################################

# This module contains helper functions allowing to link several modules on
# LLVM IR level and translate final module to SPIR-V format.
# For ISPC the workflow is easy:
#   1. compile to bitcode
# This bitcode file is ready to be linked with others and translated to .spv.
# For DPC++:
#   1. extract ESIMD bitcode using clang-offload-bundler from DPC++ library
#   3. lower extracted bitcode to real VC backend intrinsics using sycl-post-link
# Lowered bitcode file can be linked with ISPC bitcode using llvm-link and
# translated then to .spv with llvm-spirv.

# Find DPCPP compiler
set(OLD_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
find_package(dpcpp_compiler)
set(CMAKE_MODULE_PATH ${OLD_CMAKE_MODULE_PATH})
unset(OLD_CMAKE_MODULE_PATH)

# Create DPCPP library
# target_name: name of the target to use for the created library
# ARGN: DPCPP source files
function (add_dpcpp_library target_name)
    set(outdir ${CMAKE_CURRENT_BINARY_DIR})

    set(DPCPP_CXX_FLAGS "")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(DPCPP_CXX_FLAGS "-O3")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(DPCPP_CXX_FLAGS "-g")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
        set(DPCPP_CXX_FLAGS "-O2 -g")
    else()
        message(FATAL_ERROR "add_dpcpp_library only supports Debug;Release;RelWithDebInfo build configs")
    endif()

    # Compile each DPCPP file
    set(DPCPP_OBJECTS "")
    foreach(src ${ARGN})
        get_filename_component(fname ${src} NAME_WE)

        # If input path is not absolute, prepend ${CMAKE_CURRENT_LIST_DIR}
        if(NOT IS_ABSOLUTE ${src})
            set(input ${CMAKE_CURRENT_LIST_DIR}/${src})
        else()
            set(input ${src})
        endif()

        set(result "${outdir}/${fname}.o")

        if(DPCPP_ESIMD_INCLUDE_DIR)
            string(REPLACE ";" ";-I;" DPCPP_ESIMD_INCLUDE_DIR_PARMS "${DPCPP_ESIMD_INCLUDE_DIR}")
            set(DPCPP_ESIMD_INCLUDE_DIR_PARMS "-I" ${DPCPP_ESIMD_INCLUDE_DIR_PARMS})
        endif()

        if (NOT DPCPP_ESIMD_FLAGS)
            set (DPCPP_ESIMD_FLAGS "")
        endif()

        add_custom_command(
            DEPENDS ${input}
            OUTPUT ${result}
            COMMAND ${DPCPP_COMPILER}
                -fsycl
                -fPIE
                -c
                ${DPCPP_CXX_FLAGS}
                ${DPCPP_ESIMD_INCLUDE_DIR_PARMS}
                ${DPCPP_ESIMD_FLAGS}
                -I ${CMAKE_CURRENT_SOURCE_DIR}
                -o ${result}
                ${input}
            COMMENT "Building DPCPP object ${result}"
        )

        list(APPEND DPCPP_OBJECTS ${result})
    endforeach()

    add_library(${target_name} STATIC)
    set_target_properties(${target_name} PROPERTIES
        LINKER_LANGUAGE CXX
        SOURCES "${DPCPP_OBJECTS}")
endfunction()

# Extract esimd bitcode
# target_name: name of the target to use for the extracted bitcode. The bitcode
#              file will be set as a ISPC_CUSTOM_DEPENDENCIES property on this target
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
            --inputs=$<TARGET_FILE:${library}>
            --unbundle
            --targets=sycl-spir64-unknown-unknown-sycldevice
            --type=a
            --outputs=${bundler_result_tmp}
        COMMAND ${DPCPP_LLVM_LINK}
            ${bundler_result_tmp}
            -o ${bundler_result}
        COMMAND ${DPCPP_SYCL_POST_LINK}
            -split=auto
            -symbols
            -split-esimd
            -lower-esimd
            -spec-const=rt
            -o ${lower_post_link}
            ${bundler_result}
        COMMENT "Extracting ESIMD Bitcode ${bundler_result_tmp}"
    )

    add_custom_target(${target_name} DEPENDS ${post_link_result})
    set_target_properties(${target_name} PROPERTIES
        ISPC_CUSTOM_DEPENDENCIES ${post_link_result}
    )
endfunction()

# Link bitcode files into one
# target_name: target to generate for the linked bitcode output
# ARGN: targets whose ISPC_CUSTOM_DEPENDENCIES properties are the bitcode files to link
function (link_bitcode target_name)
    # Join all bitcode inputs to one string
    # ispc_compile_gpu will set the outputs as the sources of the GPU target
    # Not sure if there's a cleaner/more sensible property to get this one,
    # technically it'd only be set as the "depends" for the target
    set(input "")
    foreach(bc_target ${ARGN})
        get_target_property(BC_OUTPUTS ${bc_target} ISPC_CUSTOM_DEPENDENCIES)
        foreach (bc ${BC_OUTPUTS})
            get_filename_component(ext ${bc} LAST_EXT)
            if (ext STREQUAL ".bc")
                list(APPEND input "${bc}")
            else()
                # Since the ISPC_CUSTOM_DEPENDENCIES is a custom property we control and write,
                # if the files here aren't all .bc we know that it was built for the wrong target
                # initially.
                message(FATAL_ERROR "Non-bitcode (bc) file found on target ${bc_target}")
            endif()
        endforeach()
    endforeach()

    set(outdir ${CMAKE_CURRENT_BINARY_DIR})
    set(result "${outdir}/${target_name}.bc")

    # Propagate both the target and file level dependencies
    add_custom_command(
        DEPENDS ${ARGN} ${input}
        OUTPUT ${result}
        COMMAND ${DPCPP_LLVM_LINK}
            ${input}
            -o ${result}
        COMMENT "Linking LLVM Bitcode ${result}"
    )

    add_custom_target(${target_name} DEPENDS ${result})
    set_target_properties(${target_name} PROPERTIES
        ISPC_CUSTOM_DEPENDENCIES ${result}
    )
endfunction()

# Translate the linked bitcode files to SPV
# target_name: target name to use for the output spv command, should have a single bc file
#              as its ISPC_CUSTOM_DEPENDENCIES
# ispc_target: the original ISPC target being linked
# bc_target: name of bitcode file to translate
function (translate_to_spirv target_name ispc_target bc_target)
    # Name will be the ISPC targets bc file, but with the extension swapped to spv
    # There may be multiple ISPC files (and so bc files) on the original ISPC target,
    # the linked output will take the name of the first one
    get_target_property(ISPC_BC_SOURCES ${ispc_target} ISPC_CUSTOM_DEPENDENCIES)
    list(GET ISPC_BC_SOURCES 0 ISPC_BC_SOURCE)
    get_filename_component(outdir ${ISPC_BC_SOURCE} DIRECTORY)
    get_filename_component(fname ${ISPC_BC_SOURCE} NAME_WE)
    set(spv_output "${outdir}/${fname}.spv")

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

    # Get the BC file we want to translate to SPV
    get_target_property(input ${bc_target} ISPC_CUSTOM_DEPENDENCIES)

    # Propagate both the target and file level dependencies
    add_custom_command(
        DEPENDS ${ispc_target} ${bc_target} ${input}
        OUTPUT ${spv_output}
        COMMAND ${DPCPP_LLVM_SPIRV}
            ${input}
            -o ${spv_output}
            -spirv-debug-info-version=ocl-100
            -spirv-allow-extra-diexpressions
            -spirv-allow-unknown-intrinsics=llvm.genx.
            # Not all extenstion are supported yet by VC backend
            # so list here which are supported
            #-spirv-ext=+all
            -spirv-ext=${SPV_EXT_PARMS}
        COMMENT "Translating LLVM Bitcode to SPIR-V ${spv_output}"
    )

    add_custom_target(${target_name} DEPENDS ${spv_output})
    set_target_properties(${target_name} PROPERTIES
        ISPC_CUSTOM_DEPENDENCIES ${spv_output}
    )
endfunction()

# Link ISPC and ESIMD GPU modules to SPIR-V
# ispc_target: the ispc kernel target previously compiled to bitcode
# ARGN: list of compiled DPCPP targets to link the ispc target with
function (link_ispc_esimd ispc_target)
    if (NOT BUILD_GPU)
        message(FATAL_ERROR "Linking of ISPC/ESIMD modules is supported on GPU only")
    endif()

    if (WIN32)
        message(FATAL_ERROR "Linking of ISPC/ESIMD modules is currently supported on Linux only")
    endif()

    if (NOT TARGET ${ispc_target}_bc)
        message(FATAL_ERROR "ISPC target ${ispc_target} must be compiled to bc for ISPC/ESIMD linking")
    endif()

    set(DPCPP_BC_TARGETS "")
    foreach(lib ${ARGN})
        set(bc_target_name ${lib}_bc)
        if (NOT TARGET ${bc_target_name})
            dpcpp_get_esimd_bitcode(${bc_target_name} ${lib})
        endif()
        list(APPEND DPCPP_BC_TARGETS ${bc_target_name})
    endforeach()

    link_bitcode(${ispc_target}_ispc2esimd_bc
        ${ispc_target}_bc
        ${DPCPP_BC_TARGETS})

    translate_to_spirv(${ispc_target}_ispc2esimd_spv
        ${ispc_target}_bc
        ${ispc_target}_ispc2esimd_bc)

    add_dependencies(${ispc_target} ${ispc_target}_ispc2esimd_spv)
endfunction()
