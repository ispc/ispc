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

# Explicitly enumerate .ll and .m4 files included by target .ll files.
# This is overly conservative, as they are added to every target .ll file.
# But m4 doesn't support building depfile, so explicit enumeration is the
# easiest workaround.
list(APPEND M4_IMPLICIT_DEPENDENCIES
    builtins/builtins-cm-32.ll
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

function(target_ll_to_cpp target bit os)
    set(input builtins/target-${target}.ll)
    set(include builtins)
    string(TOUPPER ${os} OS_UP)

    set(name builtins-target-${target}-${bit}bit-${os})
    string(REPLACE "-" "_" name ${name})
    set(cpp ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${name}.cpp)
    set(bc ${BITCODE_FOLDER}/${name}.bc)

    add_custom_command(
        OUTPUT ${bc}
        COMMAND ${M4_EXECUTABLE} -I${include} -DBUILD_OS=${OS_UP} -DRUNTIME=${bit} ${input}
            | \"${LLVM_AS_EXECUTABLE}\" ${LLVM_TOOLS_OPAQUE_FLAGS} -o ${bc}
        DEPENDS ${input} ${M4_IMPLICIT_DEPENDENCIES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_custom_command(
        OUTPUT ${cpp}
        COMMAND ${Python3_EXECUTABLE} bitcode2cpp.py ${bc} --type=ispc-target --runtime=${bit} --os=${OS_UP}
            > ${cpp}
        DEPENDS ${bc} bitcode2cpp.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set(tmp_list_cpp ${TARGET_BUILTIN_CPP_FILES})
    list(APPEND tmp_list_cpp ${cpp})
    set(TARGET_BUILTIN_CPP_FILES ${tmp_list_cpp} PARENT_SCOPE)

    set(tmp_list_bc ${TARGET_BUILTIN_BC_FILES})
    list(APPEND tmp_list_bc ${bc})
    set(TARGET_BUILTIN_BC_FILES ${tmp_list_bc} PARENT_SCOPE)
endfunction()

function(generate_dispatcher os)
    set(input builtins/dispatch.c)
    set(DISP_TYPE -DREGULAR)
    set(name "builtins-dispatch")
    if (${os} STREQUAL "macos")
        set(DISP_TYPE -DMACOS)
        set(name "builtins-dispatch-macos")
    endif()
    string(REPLACE "-" "_" name ${name})
    set(cpp ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${name}.cpp)
    set(bc ${BITCODE_FOLDER}/${name}.bc)

    add_custom_command(
        OUTPUT ${bc}
        COMMAND ${CLANGPP_EXECUTABLE} -x c ${ISPC_OPAQUE_FLAGS} ${DISP_TYPE} -O2 -emit-llvm -c ${input} -o ${bc}
        DEPENDS ${input}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_custom_command(
        OUTPUT ${cpp}
        COMMAND ${Python3_EXECUTABLE} bitcode2cpp.py ${bc} --type=dispatch --os=${os} > ${cpp}
        DEPENDS ${bc} bitcode2cpp.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set(tmp_list_cpp ${DISPATCH_BUILTIN_CPP_FILES})
    list(APPEND tmp_list_cpp ${cpp})
    set(DISPATCH_BUILTIN_CPP_FILES ${tmp_list_cpp} PARENT_SCOPE)

    set(tmp_list_bc ${DISPATCH_BUILTIN_BC_FILES})
    list(APPEND tmp_list_bc ${bc})
    set(DISPATCH_BUILTIN_BC_FILES ${tmp_list_bc} PARENT_SCOPE)
endfunction()

function(builtin_wasm_to_cpp bit os arch)
    set(input builtins/builtins-c-cpu.cpp)
    set(name builtins-cpp-${bit}-${os}-${arch})
    string(REPLACE "-" "_" name ${name})

    # Report supported targets.
    message (STATUS "Enabling target: ${os} / ${arch}")

    set(cpp ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${name}.cpp)
    set(bc ${BITCODE_FOLDER}/${name}.bc)

    list(APPEND flags
        -DWASM -s WASM_OBJECT_FILES=0 ${ISPC_OPAQUE_FLAGS} -I${CMAKE_SOURCE_DIR} --std=gnu++17 -S -emit-llvm -c)
    if("${bit}" STREQUAL "64")
        list(APPEND flags "-sMEMORY64")
    endif()

    add_custom_command(
        OUTPUT ${bc}
        COMMAND ${EMCC_EXECUTABLE} ${flags} ${input} -o -
            | \"${LLVM_AS_EXECUTABLE}\" ${LLVM_TOOLS_OPAQUE_FLAGS} -o ${bc}
        DEPENDS ${input}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_custom_command(
        OUTPUT ${cpp}
        COMMAND ${Python3_EXECUTABLE} bitcode2cpp.py ${bc} --type=builtins-c --runtime=${bit} --os=${os} --arch=${arch}
            > ${cpp}
        DEPENDS ${bc} bitcode2cpp.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set(tmp_list_cpp ${COMMON_BUILTIN_CPP_FILES})
    list(APPEND tmp_list_cpp ${cpp})
    set(COMMON_BUILTIN_CPP_FILES ${tmp_list_cpp} PARENT_SCOPE)

    set(tmp_list_bc ${COMMON_BUILTIN_BC_FILES})
    list(APPEND tmp_list_bc ${bc})
    set(COMMON_BUILTIN_BC_FILES ${tmp_list_bc} PARENT_SCOPE)
endfunction()

function (get_target_flags os arch out)
    # Determine triple
    set(fpic "")
    set(debian_triple)
    if (${os} STREQUAL "windows")
        set(triple ${arch}-pc-win32)
    elseif (${os} STREQUAL "linux")
        if (${arch} STREQUAL "i686" OR ${arch} STREQUAL "x86_64" OR ${arch} STREQUAL "aarch64")
            set(triple ${arch}-unknown-linux-gnu)
            set(debian_triple ${arch}-linux-gnu)
        elseif (${arch} STREQUAL "armv7")
            set(triple ${arch}-unknown-linux-gnueabihf)
            set(debian_triple arm-linux-gnueabihf)
        else()
            message(FATAL_ERROR "Error")
        endif()
        set(fpic -fPIC)
    elseif (${os} STREQUAL "freebsd")
        set(triple ${arch}-unknown-freebsd)
        set(fpic -fPIC)
    elseif (${os} STREQUAL "macos")
        set(triple ${arch}-apple-macosx)
    elseif (${os} STREQUAL "android")
        set(triple ${arch}-unknown-linux-android)
        set(fpic -fPIC)
    elseif (${os} STREQUAL "ios")
        set(triple ${arch}-apple-ios)
    elseif (${os} STREQUAL "ps4")
        set(triple ${arch}-scei-ps)
        set(fpic -fPIC)
    elseif (${os} STREQUAL "web")
        set(triple ${arch}-unknown-unknown)
        set(fpic -fPIC)
    else()
        message(FATAL_ERROR "Error")
    endif()

    # Determine include path
    if (WIN32)
        if (${os} STREQUAL "windows")
            set(include "")
        elseif(${os} STREQUAL "macos")
            # -isystemC:/iusers/MacOSX10.14.sdk.tar/MacOSX10.14.sdk/usr/include
            set(include -isystem${ISPC_MACOS_SDK_PATH}/usr/include)
        else()
            # -isystemC:/gnuwin32/include/glibc
            set(include -isystem${ISPC_GNUWIN32_PATH}/include/glibc)
        endif()
    elseif (APPLE)
        if (${os} STREQUAL "ios")
            # -isystem/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk/usr/include/
            set(include -isystem${ISPC_IOS_SDK_PATH}/usr/include)
        elseif (${os} STREQUAL "linux" OR ${os} STREQUAL "android" OR ${os} STREQUAL "freebsd")
            if (${arch} STREQUAL "armv7")
                # -isystem/Users/Shared/android-ndk-r20/sysroot/usr/include -isystem/Users/Shared/android-ndk-r20/sysroot/usr/include/arm-linux-androideabi
                set(include -isystem${ISPC_ANDROID_NDK_PATH}/sysroot/usr/include -isystem${ISPC_ANDROID_NDK_PATH}/sysroot/usr/include/arm-linux-androideabi)
            elseif (${arch} STREQUAL "aarch64")
                # -isystem/Users/Shared/android-ndk-r20/sysroot/usr/include -isystem/Users/Shared/android-ndk-r20/sysroot/usr/include/aarch64-linux-android
                set(include -isystem${ISPC_ANDROID_NDK_PATH}/sysroot/usr/include -isystem${ISPC_ANDROID_NDK_PATH}/sysroot/usr/include/aarch64-linux-android)
            elseif(${arch} STREQUAL "i686")
                # -isystem/Users/Shared/android-ndk-r20/sysroot/usr/include -isystem/Users/Shared/android-ndk-r20/sysroot/usr/include/i686-linux-android
                set(include -isystem${ISPC_ANDROID_NDK_PATH}/sysroot/usr/include -isystem${ISPC_ANDROID_NDK_PATH}/sysroot/usr/include/i686-linux-android)
            else()
                # -isystem/Users/Shared/android-ndk-r20/sysroot/usr/include -isystem/Users/Shared/android-ndk-r20/sysroot/usr/include/x86_64-linux-android
                set(include -isystem${ISPC_ANDROID_NDK_PATH}/sysroot/usr/include -isystem${ISPC_ANDROID_NDK_PATH}/sysroot/usr/include/x86_64-linux-android)
            endif()
        elseif (${os} STREQUAL "macos")
            set(include -isystem${ISPC_MACOS_SDK_PATH}/usr/include)
        endif()
    else()
        if (${os} STREQUAL "macos")
            # -isystem/iusers/MacOSX10.14.sdk.tar/MacOSX10.14.sdk/usr/include
            set(include -isystem${ISPC_MACOS_SDK_PATH}/usr/include)
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
            set(include -isystem/usr/${debian_triple}/include)
        endif()
    endif()

    # Compose target flags
    set(${out} --target=${triple} ${fpic} ${include} PARENT_SCOPE)
endfunction()

function(builtin_to_cpp bit os generic_arch)
    set(input builtins/builtins-c-cpu.cpp)
    set(include "")

    if ("${bit}" STREQUAL "32" AND ${generic_arch} STREQUAL "x86")
        set(arch "i686")
    elseif ("${bit}" STREQUAL "64" AND ${generic_arch} STREQUAL "x86")
        set(arch "x86_64")
    elseif ("${bit}" STREQUAL "32" AND ${generic_arch} STREQUAL "arm")
        set(arch "armv7")
    elseif ("${bit}" STREQUAL "64" AND ${generic_arch} STREQUAL "arm")
        set(arch "aarch64")
    else()
        message(FATAL_ERROR "Error")
    endif()

    # Report supported targets.
    message (STATUS "Enabling target: ${os} / ${arch}")

    get_target_flags(${os} ${arch} target_flags)
    list(APPEND flags ${target_flags}
        -I${CMAKE_SOURCE_DIR} -m${bit} -S -emit-llvm ${ISPC_OPAQUE_FLAGS} --std=gnu++17 -c
    )

    set(name builtins-cpp-${bit}-${os}-${arch})
    string(REPLACE "-" "_" name ${name})
    set(cpp ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${name}.cpp)
    set(bc ${BITCODE_FOLDER}/${name}.bc)

    add_custom_command(
        OUTPUT ${bc}
        COMMAND ${CLANGPP_EXECUTABLE} ${flags} ${input} -o -
            | \"${LLVM_AS_EXECUTABLE}\" ${LLVM_TOOLS_OPAQUE_FLAGS} -o ${bc}
        DEPENDS ${input}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_custom_command(
        OUTPUT ${cpp}
        COMMAND ${Python3_EXECUTABLE} bitcode2cpp.py ${bc} --type=builtins-c --runtime=${bit} --os=${os} --arch=${arch}
            > ${cpp}
        DEPENDS ${bc} bitcode2cpp.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set(tmp_list_cpp ${COMMON_BUILTIN_CPP_FILES})
    list(APPEND tmp_list_cpp ${cpp})
    set(COMMON_BUILTIN_CPP_FILES ${tmp_list_cpp} PARENT_SCOPE)

    set(tmp_list_bc ${COMMON_BUILTIN_BC_FILES})
    list(APPEND tmp_list_bc ${bc})
    set(COMMON_BUILTIN_BC_FILES ${tmp_list_bc} PARENT_SCOPE)
endfunction()

function(builtin_xe_to_cpp os)
    set(bit 64)
    set(arch xe64)
    set(name builtins-cm-${bit})
    set(input builtins/${name}.ll)

    string(REPLACE "-" "_" name ${name})
    set(cpp ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${name}.cpp)
    set(bc ${BITCODE_FOLDER}/${name}.bc)

    add_custom_command(
        OUTPUT ${bc}
        COMMAND cat ${input} | \"${LLVM_AS_EXECUTABLE}\" ${LLVM_TOOLS_OPAQUE_FLAGS} -o ${bc}
        DEPENDS ${input}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )

    add_custom_command(
        OUTPUT ${cpp}
        COMMAND ${Python3_EXECUTABLE} bitcode2cpp.py ${bc} --type=builtins-c --runtime=${bit} --os=${os} --arch=${arch}
            > ${cpp}
        DEPENDS ${bc} bitcode2cpp.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )

    set(tmp_list_cpp ${COMMON_BUILTIN_CPP_FILES})
    list(APPEND tmp_list_cpp ${cpp})
    set(COMMON_BUILTIN_CPP_FILES ${tmp_list_cpp} PARENT_SCOPE)

    set(tmp_list_bc ${COMMON_BUILTIN_BC_FILES})
    list(APPEND tmp_list_bc ${bc})
    set(COMMON_BUILTIN_BC_FILES ${tmp_list_bc} PARENT_SCOPE)
endfunction()

function (generate_dispatch_builtins)
    if (X86_ENABLED)
        # If we build ISPC without X86 support, we don't need to generate x86
        # specific dispatch code.
        generate_dispatcher("linux")
        generate_dispatcher("macos")
    endif()

    set(DISPATCH_BUILTIN_CPP_FILES ${DISPATCH_BUILTIN_CPP_FILES} PARENT_SCOPE)
    set(DISPATCH_BUILTIN_BC_FILES ${DISPATCH_BUILTIN_BC_FILES} PARENT_SCOPE)
endfunction()

function (generate_target_builtins)
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
                    target_ll_to_cpp(${target} ${bit} ${os})
                endforeach()
            endforeach()
        endforeach()
    endif()

    # XE targets
    if (XE_ENABLED)
        foreach (target ${XE_TARGETS})
            foreach (os ${os_list})
                target_ll_to_cpp(${target} 64 ${os})
            endforeach()
        endforeach()
    endif()

    # ARM targets
    if (ARM_ENABLED)
        foreach (os ${os_list})
            foreach (target ${ARM_TARGETS})
                target_ll_to_cpp(${target} 32 ${os})
            endforeach()
            # Not all targets have 64bit
            target_ll_to_cpp(neon-i32x4 64 ${os})
            target_ll_to_cpp(neon-i32x8 64 ${os})
        endforeach()
    endif()

    # WASM targets.
    if (WASM_ENABLED)
        foreach (target ${WASM_TARGETS})
            foreach (bit 32 64)
                target_ll_to_cpp(${target} ${bit} web)
            endforeach()
        endforeach()
    endif()

    set(TARGET_BUILTIN_CPP_FILES ${TARGET_BUILTIN_CPP_FILES} PARENT_SCOPE)
    set(TARGET_BUILTIN_BC_FILES ${TARGET_BUILTIN_BC_FILES} PARENT_SCOPE)
endfunction()

function (generate_common_builtins)
    if (ISPC_LINUX_TARGET AND ARM_ENABLED)
        builtin_to_cpp(32 linux arm)
        builtin_to_cpp(64 linux arm)
    endif()

    if (ISPC_LINUX_TARGET AND X86_ENABLED)
        builtin_to_cpp(32 linux x86)
        builtin_to_cpp(64 linux x86)
    endif()

    if (ISPC_ANDROID_TARGET AND ARM_ENABLED)
        builtin_to_cpp(32 android arm)
        builtin_to_cpp(64 android arm)
    endif()

    if (ISPC_ANDROID_TARGET AND X86_ENABLED)
        builtin_to_cpp(32 android x86)
        builtin_to_cpp(64 android x86)
    endif()

    if (ISPC_FREEBSD_TARGET AND ARM_ENABLED)
        builtin_to_cpp(32 freebsd arm)
        builtin_to_cpp(64 freebsd arm)
    endif()

    if (ISPC_FREEBSD_TARGET AND X86_ENABLED)
        builtin_to_cpp(32 freebsd x86)
        builtin_to_cpp(64 freebsd x86)
    endif()

    if (ISPC_WINDOWS_TARGET AND ARM_ENABLED)
        builtin_to_cpp(64 windows arm)
    endif()

    if (ISPC_WINDOWS_TARGET AND X86_ENABLED)
        builtin_to_cpp(32 windows x86)
        builtin_to_cpp(64 windows x86)
    endif()

    if (ISPC_MACOS_TARGET AND ARM_ENABLED)
        builtin_to_cpp(64 macos arm)
    endif()

    if (ISPC_MACOS_TARGET AND X86_ENABLED)
        builtin_to_cpp(64 macos x86)
    endif()

    if (ISPC_IOS_TARGET AND ARM_ENABLED)
        builtin_to_cpp(64 ios arm)
    endif()

    if (ISPC_PS_TARGET AND X86_ENABLED)
        builtin_to_cpp(64 ps4 x86)
    endif()

    if (WIN32)
        builtin_xe_to_cpp(windows)
    elseif (APPLE)
        # no xe support
    else()
        builtin_xe_to_cpp(linux)
    endif()

    if (WASM_ENABLED)
        builtin_wasm_to_cpp(32 web wasm32)
        builtin_wasm_to_cpp(64 web wasm64)
    endif()

    set(COMMON_BUILTIN_CPP_FILES ${COMMON_BUILTIN_CPP_FILES} PARENT_SCOPE)
    set(COMMON_BUILTIN_BC_FILES ${COMMON_BUILTIN_BC_FILES} PARENT_SCOPE)
endfunction()
