#
#  Copyright (c) 2024, Intel Corporation
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
                if (${os} STREQUAL "windows")
                    continue()
                endif()
                disp_target_stdlib(${func} ${ispc_name} ${target} 32 ${os} ${CPP_LIST} ${BC_LIST})
            endforeach()
            # Not all targets have 64bit
            disp_target_stdlib(${func} ${ispc_name} neon-i32x4 64 ${os} ${CPP_LIST} ${BC_LIST})
            disp_target_stdlib(${func} ${ispc_name} neon-i32x8 64 ${os} ${CPP_LIST} ${BC_LIST})
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
