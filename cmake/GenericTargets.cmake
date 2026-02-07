#
#  Copyright (c) 2025-2026, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

# Define custom commands to generate bitcode for a specific generic target and
# a corresponding cpp wrapper for bitcode. All they are generated from
# builtins/generic.ispc file. For slim binary, an entry to the
# bitcode_libs_generated.cpp file is added.
function(generate_generic_target_builtin ispc_name target arch bit os)
    generate_generic(${ispc_name} ${target} ${arch} ${bit} ${os} "builtins")

    # Propagate the updated lists back up through this scope
    set(GENERIC_TARGET_BC_FILE ${GENERIC_TARGET_BC_FILE} PARENT_SCOPE)
    set(GENERIC_TARGET_CPP_FILE ${GENERIC_TARGET_CPP_FILE} PARENT_SCOPE)
endfunction()

# Define custom commands to generate bitcode for a specific generic target
# stdlib and a corresponding cpp wrapper for its bitcode. All they are
# generated from stdlib/stdlib.ispc file. For slim binary, an entry to the
# bitcode_libs_generated.cpp file is added.
function(generate_generic_target_stdlib ispc_name target arch bit os)
    generate_generic(${ispc_name} ${target} ${arch} ${bit} ${os} "stdlib")

    # Propagate the updated lists back up through this scope
    set(GENERIC_STDLIB_BC_FILE ${GENERIC_STDLIB_BC_FILE} PARENT_SCOPE)
    set(GENERIC_STDLIB_CPP_FILE ${GENERIC_STDLIB_CPP_FILE} PARENT_SCOPE)
endfunction()

# Generate custom commands to generate bitcode and corresponding cpp files for
# all generic targets and define a library of cpp files. It just traverses all
# targets/archs/os combinations and call generate_generic_target_builtin for
# each of them.
function (generate_generic_builtins ispc_name)
    list(APPEND os_list)
    if (ISPC_WINDOWS_TARGET)
        list(APPEND os_list "windows")
    endif()
    if (ISPC_UNIX_TARGET)
        list(APPEND os_list "unix")
    endif()

    list(APPEND TARGET_LIST
        "i1x4"
        "i1x8"
        "i1x16"
        "i1x32"
        "i1x64"
        "i8x16"
        "i8x32"
        "i16x8"
        "i16x16"
        "i32x4"
        "i32x8"
        "i32x16"
        "i64x4"
    )

    list(APPEND ARCH_LIST)
    if (X86_ENABLED)
        list(APPEND ARCH_LIST
            "x86_64,64"
        )
        # Generate 32-bit x86 generic stdlib/builtins when:
        # - Not on Apple (native Linux/Windows), OR
        # - On Apple with cross-compilation enabled (ISPC_LINUX_TARGET or ISPC_WINDOWS_TARGET)
        if (NOT APPLE OR ISPC_LINUX_TARGET OR ISPC_WINDOWS_TARGET)
            list(APPEND ARCH_LIST
                "x86,32"
            )
        endif()
    endif()

    if (ARM_ENABLED)
        list(APPEND ARCH_LIST
            "aarch64,64"
            "arm,32"
        )
    endif()

    if (RISCV_ENABLED AND ISPC_LINUX_TARGET)
        list(APPEND ARCH_LIST
            "riscv64,64"
        )
    endif()

    if (PPC64_ENABLED AND ISPC_LINUX_TARGET)
        list(APPEND ARCH_LIST
            "ppc64le,64"
        )
    endif()

    foreach(os ${os_list})
        foreach(target ${TARGET_LIST})
            foreach(pair ${ARCH_LIST})
                string(REGEX REPLACE "," ";" pair_split ${pair})
                list(GET pair_split 0 arch)
                list(GET pair_split 1 bit)
                # Skip unsupported cases, see Target::GetTripleString for more details.
                if (${os} STREQUAL "windows" AND (${arch} STREQUAL "arm" OR ${arch} STREQUAL "riscv64" OR ${arch} STREQUAL "ppc64le"))
                    continue()
                endif()
                generate_generic_target_builtin(${ispc_name} ${target} ${arch} ${bit} ${os})
                generate_generic_target_stdlib(${ispc_name} ${target} ${arch} ${bit} ${os})
            endforeach()
        endforeach()
    endforeach()

    if (WASM_ENABLED)
        foreach(target ${TARGET_LIST})
            generate_generic_target_builtin(${ispc_name} ${target} wasm64 64 web)
            generate_generic_target_stdlib(${ispc_name} ${target} wasm64 64 web)
            generate_generic_target_builtin(${ispc_name} ${target} wasm32 32 web)
            generate_generic_target_stdlib(${ispc_name} ${target} wasm32 32 web)
        endforeach()
    endif()

    add_custom_target(generic-target-bc DEPENDS ${GENERIC_TARGET_BC_FILE})
    add_custom_target(generic-target-cpp DEPENDS ${GENERIC_TARGET_CPP_FILE})

    add_custom_target(generic-stdlib-bc DEPENDS ${GENERIC_STDLIB_BC_FILE})
    add_custom_target(generic-stdlib-cpp DEPENDS ${GENERIC_STDLIB_CPP_FILE})

    set_source_files_properties(${GENERIC_TARGET_CPP_FILE} ${GENERIC_STDLIB_CPP_FILE} PROPERTIES GENERATED true)
    add_library(generic-target OBJECT EXCLUDE_FROM_ALL ${GENERIC_TARGET_CPP_FILE} ${GENERIC_STDLIB_CPP_FILE})
    add_dependencies(generic-target generic-target-cpp generic-stdlib-cpp)
endfunction()
