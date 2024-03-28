#
#  Copyright (c) 2018-2024, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc Stdlib.cmake
#

if (ISPC_WINDOWS_TARGET)
    list(APPEND TARGET_OS_LIST_FOR_STDLIB "windows")
endif()
if (ISPC_MACOS_TARGET)
    list(APPEND TARGET_OS_LIST_FOR_STDLIB "macos")
endif()
if (ISPC_LINUX_TARGET OR ISPC_FREEBSD_TARGET OR ISPC_ANDROID_TARGET OR ISPC_PS_TARGET)
    list(APPEND TARGET_OS_LIST_FOR_STDLIB "linux")
endif()
if (WASM_ENABLED)
    list(APPEND TARGET_OS_LIST_FOR_STDLIB "web")
endif()

function(create_stdlib_commands stdlibInitStr)
    foreach (ispc_target ${ISPC_TARGETS})
        foreach (bit 32 64)
            foreach (target_os ${TARGET_OS_LIST_FOR_STDLIB})
                # Neon targets constrains: neon-i8x16 and neon-i16x8 are implemented only for 32 bit ARM.
                if ("${bit}" STREQUAL "64" AND
                    (${ispc_target} STREQUAL "neon-i8x16" OR ${ispc_target} STREQUAL "neon-i16x8"))
                    continue()
                endif()

                string(REGEX MATCH "^wasm" isWasm ${ispc_target})
                if (isWasm AND NOT ${target_os} STREQUAL "web")
                    continue()
                endif()
                if (${target_os} STREQUAL "web" AND NOT isWasm)
                    continue()
                endif()
                if (isWasm)
                    if (${bit} STREQUAL "64")
                        set(target_arch "wasm64")
                    else()
                        set(target_arch "wasm32")
                    endif()
                endif()

                string(REGEX MATCH "^(sse|avx)" isX86 ${ispc_target})
                if (isX86)
                    if (${bit} STREQUAL "64")
                        set(target_arch "x86_64")
                    else()
                        set(target_arch "x86")
                    endif()
                endif()

                string(REGEX MATCH "^(neon)" isArm ${ispc_target})
                if (isArm)
                    if (${bit} STREQUAL "64")
                        set(target_arch "aarch64")
                    else()
                        set(target_arch "arm")
                    endif()
                endif()

                # TODO! wasm
                string(REGEX MATCH "^(xe|gen9)" isXe "${ispc_target}")
                if (isXe)
                    set(target_arch "xe64")
                    if ("${bit}" STREQUAL "32")
                        continue()
                    endif()
                    if (WIN32)
                        if (${target_os} STREQUAL "macos" OR ${target_os} STREQUAL "linux")
                            continue()
                        endif()
                    elseif (APPLE)
                        continue()
                    elseif (UNIX)
                        if (${target_os} STREQUAL "macos" OR ${target_os} STREQUAL "windows")
                            continue()
                        endif()
                    endif()
                endif()

                if (${target_os} STREQUAL "windows" AND ${target_arch} STREQUAL "arm")
                    continue()
                endif()

                if (${target_os} STREQUAL "macos" AND ${target_arch} STREQUAL "arm")
                    continue()
                endif()

                if (${target_os} STREQUAL "macos" AND ${target_arch} STREQUAL "x86")
                    continue()
                endif()

                string(REGEX MATCH "^avx512spr" isSpr "${ispc_target}")
                if (isSpr AND ${target_os} STREQUAL "macos")
                    continue()
                endif()

                string(REPLACE "-" "_" ispc_target_u "${ispc_target}")
                set(filename stdlib-${ispc_target_u}_${target_arch}_${target_os}.bc)
                set(output ${BITCODE_FOLDER}/${filename})
                list(APPEND STDLIB_BC_FILES ${output})
                add_custom_command(
                    OUTPUT ${output}
                    COMMAND ${PROJECT_NAME} --nostdlib --gen-stdlib --target=${ispc_target} --arch=${target_arch} --target-os=${target_os} stdlib.ispc --emit-llvm -o ${output}
                    DEPENDS ${PROJECT_NAME} stdlib.ispc stdlib.isph target.isph
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                )

		        # macos is canonicalized to linux in target_registry.cpp:lGetTargetLib
		        if (${target_os} STREQUAL "macos")
		            set(target_os "linux")
		        endif()
                list(APPEND BITCODE_LIB_CONSTRUCTORS "BitcodeLib(\"${filename}\", ISPCTarget::${ispc_target_u}, TargetOS::${target_os}, Arch::${target_arch})")
            endforeach()
        endforeach()
    endforeach()

    add_custom_target(
        stdlib ALL
        DEPENDS ${PROJECT_NAME} builtins ${STDLIB_BC_FILES}
    )

    set_property(GLOBAL PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${STDLIB_BC_FILES})

    set(tmpStdlibInitStr "")
    foreach(elem IN LISTS BITCODE_LIB_CONSTRUCTORS)
        set(tmpStdlibInitStr "${tmpStdlibInitStr}\n    ${elem}, ")
    endforeach()

    set(${stdlibInitStr} ${tmpStdlibInitStr} PARENT_SCOPE)
endfunction()
