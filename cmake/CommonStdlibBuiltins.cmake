#
#  Copyright (c) 2024-2025, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc CommonStdlibBuiltins.cmake
#

# Common function to determine architecture and OS
function(determine_arch_and_os target bit os out_arch out_os)
    set(arch "error")
    if ("${target}" MATCHES "sse|avx")
        if ("${bit}" STREQUAL "32")
            set(arch "x86")
        elseif ("${bit}" STREQUAL "64")
            set(arch "x86_64")
        else()
            set(arch "error")
        endif()
    elseif ("${target}" MATCHES "neon")
        if ("${bit}" STREQUAL "32")
            set(arch "arm")
        elseif ("${bit}" STREQUAL "64")
            set(arch "aarch64")
        else()
            set(arch "error")
        endif()
    elseif ("${target}" MATCHES "rvv")
        if ("${bit}" STREQUAL "64")
            set(arch "riscv64")
        else()
            set(arch "error")
        endif()
    elseif ("${target}" MATCHES "wasm")
        if ("${bit}" STREQUAL "32")
            set(arch "wasm32")
        elseif ("${bit}" STREQUAL "64")
            set(arch "wasm64")
        else()
            set(arch "error")
        endif()
    elseif ("${target}" MATCHES "gen9|xe")
        set(arch "xe64")
    endif()

    if ("${arch}" STREQUAL "error")
        message(FATAL_ERROR "Incorrect target or bit passed: ${target} ${os} ${bit}")
    endif()

    if ("${os}" STREQUAL "unix")
        set(os "linux")
    endif()

    set(${out_arch} ${arch} PARENT_SCOPE)
    set(${out_os} ${os} PARENT_SCOPE)
endfunction()

# This function is a common entry to generate stdlib or target builtins via
# stdlib_to_cpp or target_ll_to_cpp
function (disp_target_stdlib func ispc_name target bit os CPP_LIST BC_LIST)
    if (${func} STREQUAL "stdlib_to_cpp")
        stdlib_to_cpp(${ispc_name} ${target} ${bit} ${os} ${CPP_LIST} ${BC_LIST})
    elseif(${func} STREQUAL "target_ll_to_cpp")
        target_ll_to_cpp(${target} ${bit} ${os} ${CPP_LIST} ${BC_LIST})
    else()
        message(FATAL_ERROR "Incorrect func name")
    endif()

    set(${CPP_LIST} ${${CPP_LIST}} PARENT_SCOPE)
    set(${BC_LIST} ${${BC_LIST}} PARENT_SCOPE)
endfunction()

# Common function to call stdlib_to_cpp or target_ll_to_cpp. The actual name
# is passed in func argument.
function (generate_stdlib_or_target_builtins func ispc_name CPP_LIST BC_LIST)
    list(APPEND os_list)
    if (ISPC_WINDOWS_TARGET)
        list(APPEND os_list "windows")
    endif()
    if (ISPC_UNIX_TARGET)
        list(APPEND os_list "unix")
    endif()

    if (NOT os_list)
        message(FATAL_ERROR "Windows or Linux target has to be enabled")
    endif()

    # "Regular" targets, targeting specific real ISA: sse/avx
    if (X86_ENABLED)
        foreach (target ${X86_TARGETS})
            foreach (bit 32 64)
                foreach (os ${os_list})
                    disp_target_stdlib(${func} ${ispc_name} ${target} ${bit} ${os} ${CPP_LIST} ${BC_LIST})
                endforeach()
            endforeach()
        endforeach()
    endif()

    # XE targets
    if (XE_ENABLED)
        foreach (target ${XE_TARGETS})
            if (${func} STREQUAL "stdlib_to_cpp")
                # No cross-compilation for XE targets
                if (WIN32)
                    disp_target_stdlib(${func} ${ispc_name} ${target} 64 windows ${CPP_LIST} ${BC_LIST})
                elseif (APPLE)
                    # no xe support
                else()
                    disp_target_stdlib(${func} ${ispc_name} ${target} 64 unix ${CPP_LIST} ${BC_LIST})
                endif()
            elseif(${func} STREQUAL "target_ll_to_cpp")
                foreach (os ${os_list})
                    disp_target_stdlib(${func} ${ispc_name} ${target} 64 ${os} ${CPP_LIST} ${BC_LIST})
                endforeach()
            else()
                message(FATAL_ERROR "Incorrect func name")
            endif()
        endforeach()
    endif()

    # ARM targets
    if (ARM_ENABLED)
        foreach (os ${os_list})
            foreach (target ${ARM_TARGETS})
                disp_target_stdlib(${func} ${ispc_name} ${target} 64 ${os} ${CPP_LIST} ${BC_LIST})
                # On windows only 64-bit neon targets is supported
                if (${os} STREQUAL "windows")
                    continue()
                endif()
                disp_target_stdlib(${func} ${ispc_name} ${target} 32 ${os} ${CPP_LIST} ${BC_LIST})
            endforeach()
        endforeach()
    endif()

    # RISC-V targets (Linux only)
    if (RISCV_ENABLED AND ISPC_LINUX_TARGET)
        foreach (target ${RISCV_TARGETS})
            disp_target_stdlib(${func} ${ispc_name} ${target} 64 unix ${CPP_LIST} ${BC_LIST})
        endforeach()
    endif()

    # WASM targets.
    if (WASM_ENABLED)
        foreach (target ${WASM_TARGETS})
            foreach (bit 32 64)
                disp_target_stdlib(${func} ${ispc_name} ${target} ${bit} web ${CPP_LIST} ${BC_LIST})
            endforeach()
        endforeach()
    endif()

    set(${CPP_LIST} ${${CPP_LIST}} PARENT_SCOPE)
    set(${BC_LIST} ${${BC_LIST}} PARENT_SCOPE)
endfunction()

# Generate bitcode and corresponding C++ wrapper files for generic targets.
function(generate_generic ispc_name target arch bit os component)
    set(include ${CMAKE_CURRENT_SOURCE_DIR}/stdlib/include)

    # Handle OS-specific settings
    if("${os}" STREQUAL "unix")
        set(fixed_os "linux")
        set(OS_UP UNIX)
    elseif("${os}" STREQUAL "windows")
        set(fixed_os "windows")
        set(OS_UP WINDOWS)
    elseif("${os}" STREQUAL "web")
        set(fixed_os "web")
        set(OS_UP WEB)
    else()
        message(FATAL_ERROR "Error: unknown OS for target ${os}")
    endif()

    # Set component-specific variables
    if("${component}" STREQUAL "builtins")
        set(name "builtins_target_generic_${target}_${arch}_${os}")
        set(input_ispc "builtins/generic.ispc")
        set(bitcode_type "ispc-target")
        set(bitcode_lib_entry "static BitcodeLib ${name}(\"${name}.bc\", ISPCTarget::generic_${target}, TargetOS::${fixed_os}, Arch::${arch});\n")
        set(bc_list_var "GENERIC_TARGET_BC_FILE")
        set(cpp_list_var "GENERIC_TARGET_CPP_FILE")
    else() # stdlib
        set(name "stdlib_generic_${target}_${arch}_${os}")
        set(input_ispc "stdlib/stdlib.ispc")
        set(bitcode_type "stdlib")
        set(bitcode_lib_entry "static BitcodeLib ${name}(BitcodeLib::BitcodeLibType::Stdlib, \"${name}.bc\", ISPCTarget::generic_${target}, TargetOS::${fixed_os}, Arch::${arch});\n")
        set(bc_list_var "GENERIC_STDLIB_BC_FILE")
        set(cpp_list_var "GENERIC_STDLIB_CPP_FILE")
    endif()

    # Handle Linux target special case
    if("${fixed_os}" STREQUAL "linux" AND NOT ISPC_LINUX_TARGET)
        if(APPLE)
            set(fixed_os "macos")
            if (${arch} STREQUAL "arm")
                return()
            endif()
        elseif(CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
            set(fixed_os "freebsd")
        else()
            message(FATAL_ERROR "Error: unknown OS for target ${fixed_os}")
        endif()
    endif()

    file(APPEND ${CMAKE_BINARY_DIR}/bitcode_libs_generated.cpp "${bitcode_lib_entry}")

    set(target "generic-${target}")
    set(cpp ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${name}.cpp)
    set(bc ${BITCODE_FOLDER}/${name}.bc)

    # Generate bitcode
    add_custom_command(
        OUTPUT ${bc}
        COMMAND ${ispc_name} -I ${include} --enable-llvm-intrinsics --nostdlib --gen-stdlib
                --target=${target} --arch=${arch} --target-os=${fixed_os} --emit-llvm
                -o ${bc} ${input_ispc} -DBUILD_OS=${OS_UP} -DRUNTIME=${bit}
        DEPENDS ${ispc_name} ${input_ispc}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Generate CPP wrapper
    add_custom_command(
        OUTPUT ${cpp}
        COMMAND ${Python3_EXECUTABLE} ${BITCODE2CPP} ${bc} --type=${bitcode_type}
                --runtime=${bit} --os=${OS_UP} --arch=${arch} ${cpp}
        DEPENDS ${bc} ${BITCODE2CPP}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Update parent scope variables
    list(APPEND ${bc_list_var} ${bc})
    set(${bc_list_var} ${${bc_list_var}} PARENT_SCOPE)

    list(APPEND ${cpp_list_var} ${cpp})
    set(${cpp_list_var} ${${cpp_list_var}} PARENT_SCOPE)
endfunction()
