#
#  Copyright (c) 2018-2024, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc GenerateBuiltins.cmake
#
find_program(M4_EXECUTABLE m4)
    if (NOT M4_EXECUTABLE)
        message(FATAL_ERROR "Failed to find M4 macro processor" )
    endif()
    message(STATUS "M4 macro processor: " ${M4_EXECUTABLE})

if (WIN32)
    set(TARGET_OS_LIST_FOR_LL "windows" "linux")
elseif (UNIX)
    set(TARGET_OS_LIST_FOR_LL "linux")
endif()

# Explicitly enumerate .ll and .m4 files included by target .ll files.
# This is overly conservative, as they are added to every target .ll file.
# But m4 doesn't support building depfile, so explicit enumeration is the
# easiest workaround.
list(APPEND M4_IMPLICIT_DEPENDENCIES
    builtins/builtins-cm-64.ll
    builtins/svml.m4
    builtins/target-avx-utils.ll
    builtins/target-avx-common-8.ll
    builtins/target-avx-common-16.ll
    builtins/target-avx1-i64x4base.ll
    builtins/target-avx512-common-4.ll
    builtins/target-avx512-common-8.ll
    builtins/target-avx512-common-16.ll
    builtins/target-avx512-utils.ll
    builtins/target-neon-common.ll
    builtins/target-sse2-common.ll
    builtins/target-sse4-common.ll
    builtins/target-xe.ll
    builtins/util-xe.m4
    builtins/util.m4)

function(target_ll_to_cpp llFileName bit os_name filename output)
    set(inputFilePath builtins/${llFileName}.ll)
    set(includePath builtins)
    if (${os_name} STREQUAL "linux")
        set(os_name "unix")
    endif()
    string(TOUPPER ${os_name} os_name_macro)

    add_custom_command(
        OUTPUT ${output}
        COMMAND ${M4_EXECUTABLE} -I${includePath} -DBUILD_OS=${os_name_macro} -DRUNTIME=${bit} ${inputFilePath}
            | \"${LLVM_AS_EXECUTABLE}\" ${LLVM_TOOLS_OPAQUE_FLAGS} -o ${output}
        DEPENDS ${inputFilePath} ${M4_IMPLICIT_DEPENDENCIES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_source_files_properties(${output} PROPERTIES GENERATED true)
endfunction()

function(dispatch_ll_to_cpp llFileName filename output)
    set(inputFilePath builtins/${llFileName}.ll)
    add_custom_command(
        OUTPUT ${output}
        COMMAND ${M4_EXECUTABLE} ${inputFilePath} | \"${LLVM_AS_EXECUTABLE}\" ${LLVM_TOOLS_OPAQUE_FLAGS} -o ${output}
        DEPENDS ${inputFilePath}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_source_files_properties(${resultFileName} PROPERTIES GENERATED true)
endfunction()

function(builtin_to_cpp os_name target_arch filename output)
    set(inputFilePath builtins/builtins-c-cpu.cpp)
    set(includePath "")
    set(bit 32)

    if (${target_arch} STREQUAL "x86_64" OR ${target_arch} STREQUAL "aarch64" OR ${target_arch} STREQUAL "wasm64")
        set(bit 64)
    endif()

    # Do some arch renaming
    if (${target_arch} STREQUAL "x86")
        set(target_arch "i686")
    endif()

    if (${target_arch} STREQUAL "arm")
        set(target_arch "armv7")
    endif()

    if (${os_name} STREQUAL "ios" AND ${target_arch} STREQUAL "aarch64")
        set(target_arch "arm64")
    endif()

    # Determine triple
    set(fpic "")
    set(debian_triple)
    if (${os_name} STREQUAL "windows")
        set(triple ${target_arch}-pc-win32)
    elseif (${os_name} STREQUAL "linux")
        if (${target_arch} STREQUAL "i686" OR ${target_arch} STREQUAL "x86_64" OR ${target_arch} STREQUAL "aarch64")
            set(triple ${target_arch}-unknown-linux-gnu)
            set(debian_triple ${target_arch}-linux-gnu)
        elseif (${target_arch} STREQUAL "armv7")
            set(triple ${target_arch}-unknown-linux-gnueabihf)
            set(debian_triple arm-linux-gnueabihf)
        else()
            message(FATAL_ERROR "Error")
        endif()
        set(fpic -fPIC)
    elseif (${os_name} STREQUAL "freebsd")
        set(triple ${target_arch}-unknown-freebsd)
        set(fpic -fPIC)
    elseif (${os_name} STREQUAL "macos")
        set(triple ${target_arch}-apple-macosx)
    elseif (${os_name} STREQUAL "android")
        set(triple ${target_arch}-unknown-linux-android)
        set(fpic -fPIC)
    elseif (${os_name} STREQUAL "ios")
        set(triple ${target_arch}-apple-ios)
    elseif (${os_name} STREQUAL "ps4")
        set(triple ${target_arch}-scei-ps)
        set(fpic -fPIC)
    elseif (${os_name} STREQUAL "web")
        set(triple ${target_arch}-unknown-unknown)
        set(fpic -fPIC)
    else()
        message(FATAL_ERROR "Error")
    endif()

    # Determine include path
    if (WIN32)
        if (${os_name} STREQUAL "windows")
            set(includePath "")
        elseif(${os_name} STREQUAL "macos")
            # -isystemC:/iusers/MacOSX10.14.sdk.tar/MacOSX10.14.sdk/usr/include
            set(includePath -isystem${ISPC_MACOS_SDK_PATH}/usr/include)
        else()
            # -isystemC:/gnuwin32/include/glibc
            set(includePath -isystem${ISPC_GNUWIN32_PATH}/include/glibc)
        endif()
    elseif (APPLE)
        if (${os_name} STREQUAL "ios")
            # -isystem/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk/usr/include/
            set(includePath -isystem${ISPC_IOS_SDK_PATH}/usr/include)
        elseif (${os_name} STREQUAL "linux" OR ${os_name} STREQUAL "android" OR ${os_name} STREQUAL "freebsd")
            if (${target_arch} STREQUAL "armv7")
                # -isystem/Users/Shared/android-ndk-r20/sysroot/usr/include -isystem/Users/Shared/android-ndk-r20/sysroot/usr/include/arm-linux-androideabi
                set(includePath -isystem${ISPC_ANDROID_NDK_PATH}/sysroot/usr/include -isystem${ISPC_ANDROID_NDK_PATH}/sysroot/usr/include/arm-linux-androideabi)
            elseif (${target_arch} STREQUAL "aarch64")
                # -isystem/Users/Shared/android-ndk-r20/sysroot/usr/include -isystem/Users/Shared/android-ndk-r20/sysroot/usr/include/aarch64-linux-android
                set(includePath -isystem${ISPC_ANDROID_NDK_PATH}/sysroot/usr/include -isystem${ISPC_ANDROID_NDK_PATH}/sysroot/usr/include/aarch64-linux-android)
            elseif(${target_arch} STREQUAL "i686")
                # -isystem/Users/Shared/android-ndk-r20/sysroot/usr/include -isystem/Users/Shared/android-ndk-r20/sysroot/usr/include/i686-linux-android
                set(includePath -isystem${ISPC_ANDROID_NDK_PATH}/sysroot/usr/include -isystem${ISPC_ANDROID_NDK_PATH}/sysroot/usr/include/i686-linux-android)
            else()
                # -isystem/Users/Shared/android-ndk-r20/sysroot/usr/include -isystem/Users/Shared/android-ndk-r20/sysroot/usr/include/x86_64-linux-android
                set(includePath -isystem${ISPC_ANDROID_NDK_PATH}/sysroot/usr/include -isystem${ISPC_ANDROID_NDK_PATH}/sysroot/usr/include/x86_64-linux-android)
            endif()
        elseif (${os_name} STREQUAL "macos")
            set(includePath -isystem${ISPC_MACOS_SDK_PATH}/usr/include)
        endif()
    else()
        if (${os_name} STREQUAL "macos")
            # -isystem/iusers/MacOSX10.14.sdk.tar/MacOSX10.14.sdk/usr/include
            set(includePath -isystem${ISPC_MACOS_SDK_PATH}/usr/include)
        elseif(NOT ${debian_triple} STREQUAL "")
            # When compiling on Linux, there are two way to support cross targets:
            # - add "foreign" architecture to the set of supported architectures and install corresponding toolchain.
            #   For example on aarch64: "dpkg --add-architecture armhf" and "apt-get install libc6-dev:armhf".
            #   In this case the headers will be installed in /usr/include/arm-linux-gnueabihf and will be
            #   automatically picked up by clang.
            # - install cross library. For example: "apt-get install libc6-dev-armhf-cross".
            #   In this case headers will be installed in /usr/arm-linux-gnueabihf/include and will not be picked up
            #   by clang by default. So the following line adds such path explicitly. If this path doesn't exist and
            #   the headers can be found in other locations, this should not be a problem.
            set(includePath -isystem/usr/${debian_triple}/include)
        endif()
    endif()

    # Compose target flags
    set(target_flags --target=${triple} ${fpic} ${includePath})

    if (${os_name} STREQUAL "web")
        if(${target_arch} STREQUAL "wasm64")
            list(APPEND emcc_flags "-sMEMORY64")
        endif()
        # TODO!
        add_custom_command(
            OUTPUT ${output}
            COMMAND ${EMCC_EXECUTABLE} -DWASM -s WASM_OBJECT_FILES=0 ${emcc_flags} ${ISPC_OPAQUE_FLAGS} -I${CMAKE_SOURCE_DIR} -c ${inputFilePath} --std=gnu++17 -emit-llvm -c -o ${output}
            DEPENDS ${inputFilePath}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    else()
        add_custom_command(
            OUTPUT ${output}
            COMMAND ${CLANGPP_EXECUTABLE} ${target_flags} -I${CMAKE_SOURCE_DIR} -m${bit} -emit-llvm ${ISPC_OPAQUE_FLAGS} --std=gnu++17 -c ${inputFilePath} -o ${output}
            DEPENDS ${inputFilePath}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()

    set_source_files_properties(${resultFileName} PROPERTIES GENERATED true)
endfunction()

function(builtin_xe_to_cpp filename output)
    set(inputFilePath builtins/builtins-cm-64.ll)
    add_custom_command(
        OUTPUT ${output}
        COMMAND ${LLVM_AS_EXECUTABLE} ${LLVM_TOOLS_OPAQUE_FLAGS} ${inputFilePath} -o ${output}
        DEPENDS ${inputFilePath}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    set_source_files_properties(${resultFileName} PROPERTIES GENERATED true)
endfunction()

function (generate_target_builtins targetBuiltinsInitStr dispatchInitStr)
    # Dispatch module for macOS and all the rest of targets.
    if (${LLVM_VERSION_NUMBER} VERSION_GREATER_EQUAL "14.0.0")
        set(filename builtins-dispatch.bc)
        set(output_generic ${BITCODE_FOLDER}/${filename})
        dispatch_ll_to_cpp(dispatch ${filename} ${output_generic})
        list(APPEND DISPATCH_BITCODE_LIB_CONSTRUCTORS "BitcodeLib(\"${filename}\", TargetOS::linux)")
    else()
        set(filename builtins-dispatch-no-spr.bc)
        set(output_generic ${BITCODE_FOLDER}/${filename})
        dispatch_ll_to_cpp(dispatch-no-spr ${filename} ${output_generic})
        list(APPEND DISPATCH_BITCODE_LIB_CONSTRUCTORS "BitcodeLib(\"${filename}\", TargetOS::linux)")
    endif()

    set(filename builtins-macos.bc)
    set(output_macos ${BITCODE_FOLDER}/${filename})
    dispatch_ll_to_cpp(dispatch-macos ${filename} ${output_macos})
    list(APPEND tmpDispatchList ${output_generic} ${output_macos})
    list(APPEND DISPATCH_BITCODE_LIB_CONSTRUCTORS "BitcodeLib(\"${filename}\", TargetOS::macos)")

    # "Regular" targets, targeting specific real ISA: sse/avx/neon
    set(regular_targets ${ARGN})
    list(FILTER regular_targets EXCLUDE REGEX wasm)
    foreach (ispc_target ${regular_targets})
        foreach (bit 32 64)
            foreach (os_name ${TARGET_OS_LIST_FOR_LL})

                # Neon targets constrains: neon-i8x16 and neon-i16x8 are implemented only for 32 bit ARM.
                if ("${bit}" STREQUAL "64" AND
                    (${ispc_target} STREQUAL "neon-i8x16" OR ${ispc_target} STREQUAL "neon-i16x8"))
                    continue()
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

                # Xe targets are implemented only for 64 bit.
                string(REGEX MATCH "^(xe|gen9)" isXe "${ispc_target}")
                if ("${bit}" STREQUAL "32" AND isXe)
                    continue()
                elseif (isXe)
                    set(target_arch "xe64")
                endif()

                # TODO! wasm
                string(REGEX MATCH "^(xe|gen9)" isXe "${ispc_target}")
                if (isXe)
                    set(target_arch "xe64")
                    if ("${bit}" STREQUAL "32")
                        continue()
                    endif()
                endif()

                if (${os_name} STREQUAL "windows" AND ${target_arch} STREQUAL "arm")
                    continue()
                endif()

                string(REPLACE "-" "_" ispc_target_u "${ispc_target}")
                set(filename builtins-target-${ispc_target_u}_${target_arch}_${os_name}.bc)
                set(fullpath ${BITCODE_FOLDER}/${filename})
                target_ll_to_cpp(target-${ispc_target} ${bit} ${os_name} ${filename} ${fullpath})
                list(APPEND tmpBitcodeList ${fullpath})
                list(APPEND BITCODE_LIB_CONSTRUCTORS "BitcodeLib(\"${filename}\", ISPCTarget::${ispc_target_u}, TargetOS::${os_name}, Arch::${target_arch})")
            endforeach()
        endforeach()
    endforeach()

    # WASM targets.
    if (WASM_ENABLED)
        set(wasm_targets ${ARGN})
        list(FILTER wasm_targets INCLUDE REGEX wasm)
        foreach (wasm_target ${wasm_targets})
            foreach (bit 32 64)
                string(REPLACE "-" "_" wasm_target_u "${wasm_target}")
                set(filename builtins-target-${wasm_target_u}_${bit}_web.bc)
                set(fullpath ${BITCODE_FOLDER}/${filename})
                target_ll_to_cpp(target-${wasm_target} ${bit} web ${filename} ${fullpath})
                list(APPEND tmpBitcodeList ${fullpath})
                list(APPEND BITCODE_LIB_CONSTRUCTORS "BitcodeLib(\"${filename}\", ISPCTarget::${wasm_target_u}, TargetOS::web, Arch::wasm${bit})")
            endforeach()
        endforeach()
    endif()

    set(tmpTargetBuiltinsInitStr "")
    foreach(elem IN LISTS BITCODE_LIB_CONSTRUCTORS)
        set(tmpTargetBuiltinsInitStr "${tmpTargetBuiltinsInitStr}\n    ${elem}, ")
    endforeach()

    set(tmpDispatchBuiltinsInitStr "")
    foreach(elem IN LISTS DISPATCH_BITCODE_LIB_CONSTRUCTORS)
        set(tmpDispatchBuiltinsInitStr "${tmpDispatchBuiltinsInitStr}\n    ${elem}, ")
    endforeach()

    add_custom_target(
        target-builtins ALL
        DEPENDS ${tmpBitcodeList}
    )

    add_custom_target(
        dispatch-builtins ALL
        DEPENDS ${tmpDispatchList}
    )

    set_property(
        GLOBAL PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
        ${tmpBitcodeList} ${tmpDispatchList}
    )

    # Return the list
    set(${targetBuiltinsInitStr} ${tmpTargetBuiltinsInitStr} PARENT_SCOPE)
    set(${dispatchInitStr} ${tmpDispatchBuiltinsInitStr} PARENT_SCOPE)
endfunction()

function (generate_common_builtins commonBuiltinsInitStr)
    # Supported architectures, names have to correspond to Arch enum in src/target_enums.h
    if (X86_ENABLED)
        list (APPEND supported_archs "x86")
        list (APPEND supported_archs "x86_64")
    endif()
    if (ARM_ENABLED)
        list (APPEND supported_archs "arm")
        list (APPEND supported_archs "aarch64")
    endif()
    if (WASM_ENABLED)
        list (APPEND supported_archs "wasm32")
        list (APPEND supported_archs "wasm64")
        list (APPEND supported_oses "web")
    endif()

    # Supported OSes, names have to corresponding to TargetOS enum in src/target_enums.h
    if (ISPC_WINDOWS_TARGET)
        list (APPEND supported_oses "windows")
    endif()
    if (ISPC_LINUX_TARGET)
        list (APPEND supported_oses "linux")
    endif()
    if (ISPC_FREEBSD_TARGET)
        list (APPEND supported_oses "freebsd")
    endif()
    if (ISPC_MACOS_TARGET)
        list (APPEND supported_oses "macos")
    endif()
    if (ISPC_ANDROID_TARGET)
        list (APPEND supported_oses "android")
    endif()
    if (ISPC_IOS_TARGET)
        list (APPEND supported_oses "ios")
    endif()
    if (ISPC_PS_TARGET)
        list (APPEND supported_oses "ps4")
        # TODO! what about ps5?
    endif()

    foreach (os_name ${supported_oses})
        foreach (arch ${supported_archs})

            string(REGEX MATCH "^wasm" isWasm "${arch}")
            if ((    ${os_name} STREQUAL "web" AND NOT isWasm) OR
                (NOT ${os_name} STREQUAL "web" AND     isWasm))
                continue()
            endif()

            # Host to target OS constrains
            if (WIN32)
                if (${os_name} STREQUAL "ios")
                    continue()
                endif()
            elseif (APPLE)
                if (${os_name} STREQUAL "windows")
                    continue()
                elseif (${os_name} STREQUAL "ps4")
                    continue()
                endif()
            else()
                if (${os_name} STREQUAL "windows")
                    continue()
                elseif (${os_name} STREQUAL "ps4")
                    continue()
                elseif (${os_name} STREQUAL "ios")
                    continue()
                endif()
            endif()

            # OS to arch constrains
            if (${os_name} STREQUAL "windows" AND "${arch}" STREQUAL "arm")
                continue()
            endif()
            if (${os_name} STREQUAL "macos")
                if (NOT (${arch} STREQUAL "x86_64" OR
                         (${arch} STREQUAL "aarch64" AND ISPC_MACOS_ARM_TARGET)))
                    continue()
                endif()
            endif()
            if (${os_name} STREQUAL "ps4" AND NOT ${arch} STREQUAL "x86_64")
                continue()
            endif()
            if (${os_name} STREQUAL "ios" AND NOT ${arch} STREQUAL "aarch64")
                continue()
            endif()

            message (STATUS "Enabling builtins-cpp for ${os_name} / ${arch}")

            set(filename builtins-cpp-${os_name}-${arch}.bc)
            set(fullpath ${BITCODE_FOLDER}/${filename})
            builtin_to_cpp(${os_name} ${arch} ${filename} ${fullpath})
            list(APPEND tmpBitcodeList ${fullpath} )
            list(APPEND BITCODE_LIB_CONSTRUCTORS "BitcodeLib(\"${filename}\", TargetOS::${os_name}, Arch::${arch})")
        endforeach()
    endforeach()

    if (XE_ENABLED AND NOT APPLE)
        set(target_arch "xe64")
        set(SKIP OFF)
        if (WIN32)
            set(os_name "windows")
        else ()
            set(os_name "linux")
        endif()

        message (STATUS "Enabling builtins-cpp for ${os_name} / ${target_arch}")

        set(filename builtins-cpp-${os_name}-${target_arch}.bc)
        set(fullpath ${BITCODE_FOLDER}/${filename})
        builtin_xe_to_cpp(${filename} ${fullpath})
        list(APPEND tmpBitcodeList ${fullpath} )
        list(APPEND BITCODE_LIB_CONSTRUCTORS "BitcodeLib(\"${filename}\", TargetOS::${os_name}, Arch::${target_arch})")
    endif()

    set(tmpCommonBuiltinsInitStr "")
    foreach(elem IN LISTS BITCODE_LIB_CONSTRUCTORS)
        set(tmpCommonBuiltinsInitStr "${tmpCommonBuiltinsInitStr}\n    ${elem}, ")
    endforeach()

    add_custom_target(
        common-builtins ALL
        DEPENDS ${tmpBitcodeList}
    )

    set_property(GLOBAL PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${tmpBitcodeList})

    set(${commonBuiltinsInitStr} ${tmpCommonBuiltinsInitStr} PARENT_SCOPE)
endfunction()
