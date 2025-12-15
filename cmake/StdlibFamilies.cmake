#
#  Copyright (c) 2025, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

# This file defines stdlib families used by both CMake (for stdlib compilation)
# and C++ (for runtime lookup). Each family maps multiple targets to a single
# representative (base) target whose stdlib they share.
#
# Two types of base targets:
#   1. generic-* targets: For families where targets produce stdlib identical to generic
#   2. Real ISA targets: For inheritance families where child ISAs produce stdlib
#      identical to parent ISA (but different from generic, usually due to alignment)
#

# Define width families in a unified declarative format
# Format: "family_name:base_target:member1,member2,member3,..."
# where base_target is either:
#   - generic-* for families that match generic implementation
#   - real ISA target for inheritance families (e.g., avx1-i32x8 for avx2/avx2vnni)
#
# This function creates variables for each width family:
#   STDLIB_FAMILY_<family_name>: List of member targets
#   STDLIB_FAMILY_BASE_<family_name>: Base target
#   STDLIB_FAMILIES: List of all family names
#   STDLIB_FAMILY_ALL_MEMBERS: Flat list of all targets in any family
function(define_stdlib_families)
    set(all_families "")
    set(all_members "")

    # Define AVX10 target lists - can be appended to for future LLVM versions
    set(AVX10_X4 "")
    set(AVX10_X8 "")
    set(AVX10_X16 "")
    set(AVX10_X32 "")
    set(AVX10_X64 "")
    if (${LLVM_VERSION_NUMBER} VERSION_GREATER_EQUAL "20.1.2")
        list(APPEND AVX10_X4 "avx10_2dmr-x4")
        list(APPEND AVX10_X8 "avx10_2dmr-x8")
        list(APPEND AVX10_X16 "avx10_2dmr-x16")
        list(APPEND AVX10_X32 "avx10_2dmr-x32")
        list(APPEND AVX10_X64 "avx10_2dmr-x64")
    endif()
    # Convert lists to comma-separated strings with leading comma
    foreach(width IN ITEMS X4 X8 X16 X32 X64)
        if (AVX10_${width})
            list(JOIN AVX10_${width} "," AVX10_${width})
            set(AVX10_${width} ",${AVX10_${width}}")
        endif()
    endforeach()

    # Single declarative list of all families
    # Note: nozmm targets are excluded - they produce different stdlib that avoids ZMM instructions
    set(FAMILY_DEFINITIONS
        # AVX512 families - use generic targets
        "i1x4:generic-i1x4:avx512skx-x4,avx512icl-x4,avx512spr-x4,avx512gnr-x4${AVX10_X4}"
        "i1x8:generic-i1x8:avx512skx-x8,avx512icl-x8,avx512spr-x8,avx512gnr-x8${AVX10_X8}"
        "i1x16:generic-i1x16:avx512skx-x16,avx512icl-x16,avx512spr-x16,avx512gnr-x16${AVX10_X16}"
        "i1x32:generic-i1x32:avx512skx-x32,avx512icl-x32,avx512spr-x32,avx512gnr-x32${AVX10_X32}"
        "i1x64:generic-i1x64:avx512skx-x64,avx512icl-x64,avx512spr-x64,avx512gnr-x64${AVX10_X64}"

        # SSE/AVX families that match generic implementation
        "i32x4:generic-i32x4:sse2-i32x4,sse4-i32x4"
        "i32x8:generic-i32x8:sse2-i32x8,sse4-i32x8"
        "i32x16:generic-i32x16:avx1-i32x16,avx2-i32x16,avx2vnni-i32x16"
        "i64x4:generic-i64x4:avx1-i64x4,avx2-i64x4"
        "i16x8:generic-i16x8:sse4-i16x8"
        "i16x16:generic-i16x16:avx2-i16x16"
        "i8x16:generic-i8x16:sse4-i8x16"
        "i8x32:generic-i8x32:avx2-i8x32"

        # Inheritance families - use parent ISA target (NOT generic!)
        # These produce stdlib different from generic but identical to parent ISA
        "avx_i32x8:avx1-i32x8:avx2-i32x8,avx2vnni-i32x8"
        "avx_i32x4:avx2-i32x4:avx2vnni-i32x4"

        # NEON families - use generic targets
        # NEON stdlib is identical to generic except for target-cpu attribute
        "neon_i8x16:generic-i8x16:neon-i8x16"
        "neon_i16x8:generic-i16x8:neon-i16x8"
        "neon_i16x16:generic-i16x16:neon-i16x16"
        "neon_i32x4:generic-i32x4:neon-i32x4"
        "neon_i32x8:generic-i32x8:neon-i32x8"
        "neon_i8x32:generic-i8x32:neon-i8x32"
    )

    # Separate families by architecture for efficient filtering
    set(x86_families "")
    set(arm_families "")

    # Parsing loop - store family info locally and classify by architecture
    foreach(family_def ${FAMILY_DEFINITIONS})
        # Parse: family_name:base:members
        string(REPLACE ":" ";" parts ${family_def})
        list(LENGTH parts parts_len)
        if(NOT parts_len EQUAL 3)
            message(WARNING "Invalid family definition (expected 3 parts): ${family_def}")
            continue()
        endif()

        list(GET parts 0 family_name)
        list(GET parts 1 base)
        list(GET parts 2 members_str)

        # Convert comma-separated members to CMake list
        string(REPLACE "," ";" members "${members_str}")

        # Store family info locally for architecture classification
        set(local_family_${family_name}_members "${members}")
        set(local_family_${family_name}_base "${base}")

        # Export to parent scope
        set(STDLIB_FAMILY_${family_name} "${members}" PARENT_SCOPE)
        set(STDLIB_FAMILY_BASE_${family_name} "${base}" PARENT_SCOPE)
        list(APPEND all_families ${family_name})
        list(APPEND all_members ${members})

        # Classify family by architecture based on first member
        list(GET members 0 first_member)
        list(FIND X86_TARGETS ${first_member} is_x86)
        list(FIND ARM_TARGETS ${first_member} is_arm)

        if(is_x86 GREATER -1)
            list(APPEND x86_families ${family_name})
        elseif(is_arm GREATER -1)
            list(APPEND arm_families ${family_name})
        else()
            message(WARNING "Family ${family_name} doesn't match any known architecture")
        endif()

        # Display what we registered if needed
        if(ISPC_STDLIB_DEBUG_PRINT)
            if(base MATCHES "^generic-")
                message(STATUS "Generic family ${family_name}: ${members} -> ${base}")
            else()
                message(STATUS "Inheritance family ${family_name}: ${members} -> ${base}")
            endif()
        endif()
    endforeach()

    list(REMOVE_DUPLICATES all_members)
    set(STDLIB_FAMILIES "${all_families}" PARENT_SCOPE)
    set(STDLIB_FAMILY_ALL_MEMBERS "${all_members}" PARENT_SCOPE)
    set(STDLIB_X86_FAMILIES "${x86_families}" PARENT_SCOPE)
    set(STDLIB_ARM_FAMILIES "${arm_families}" PARENT_SCOPE)

    validate_stdlib_families()
endfunction()

# Generate C++ code for stdlib target map
# This creates a C++ header file with the mapping from specific targets to generic targets
function(generate_stdlib_target_map_cpp)
    set(output_file ${CMAKE_BINARY_DIR}/stdlib_target_map_generated.h)

    file(WRITE ${output_file}
        "// Auto-generated from cmake/StdlibFamilies.cmake\n"
        "// DO NOT EDIT MANUALLY - regenerated on each CMake run\n"
        "//\n"
        "// This file contains the mapping from specific targets\n"
        "// to their corresponding generic stdlib targets.\n"
        "\n"
        "#pragma once\n"
        "\n"
        "#include \"target_enums.h\"\n"
        "#include <unordered_map>\n"
        "\n"
        "namespace ispc {\n"
        "\n"
        "// Mapping from each target to its stdlib base target\n"
        "// Since we've made stdlib capability-neutral, targets in the same width family\n"
        "// produce identical stdlib bitcode. We compile stdlib once with a generic target\n"
        "// and reuse it for all targets in the family (x86, ARM, etc.).\n"
        "inline const std::unordered_map<ISPCTarget, ISPCTarget> stdlibTargetMap = {\n"
    )

    # Generate map entries for each family
    foreach(family ${STDLIB_FAMILIES})
        set(base_target ${STDLIB_FAMILY_BASE_${family}})
        string(REPLACE "-" "_" generic_enum ${base_target})

        file(APPEND ${output_file}
            "    // ${family} family -> ${base_target}\n"
        )

        foreach(target ${STDLIB_FAMILY_${family}})
            string(REPLACE "-" "_" target_enum ${target})
            file(APPEND ${output_file}
                "    {ISPCTarget::${target_enum}, ISPCTarget::${generic_enum}},\n"
            )
        endforeach()

        file(APPEND ${output_file} "\n")
    endforeach()

    file(APPEND ${output_file}
        "};\n"
        "\n"
        "} // namespace ispc\n"
    )

    message(STATUS "Generated stdlib target map: ${output_file}")
endfunction()

# Validation: Ensure all targets referenced in families exist and belong to same architecture
function(validate_stdlib_families)
    foreach(family ${STDLIB_FAMILIES})
        set(family_arch "")
        foreach(target ${STDLIB_FAMILY_${family}})
            list(FIND X86_TARGETS ${target} x86_idx)
            list(FIND ARM_TARGETS ${target} arm_idx)

            if(x86_idx EQUAL -1 AND arm_idx EQUAL -1)
                message(FATAL_ERROR
                    "Width family ${family} references non-existent target: ${target}\n"
                    "This target is not in X86_TARGETS or ARM_TARGETS list.")
            endif()

            # Verify family members are from same architecture
            if(x86_idx GREATER -1)
                set(target_arch "x86")
            else()
                set(target_arch "arm")
            endif()

            if(family_arch STREQUAL "")
                set(family_arch ${target_arch})
            elseif(NOT family_arch STREQUAL target_arch)
                message(FATAL_ERROR
                    "Width family ${family} mixes different architectures!\n"
                    "Target ${target} is ${target_arch} but family was ${family_arch}.\n"
                    "All family members must be from the same architecture.")
            endif()
        endforeach()
    endforeach()
endfunction()
