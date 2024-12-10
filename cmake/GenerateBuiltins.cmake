#
#  Copyright (c) 2018-2025, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc GenerateBuiltins.cmake
#

function(write_target_bitcode_lib name target os bit)
    determine_arch_and_os(${target} ${bit} ${os} fixed_arch fixed_os)
    string(REPLACE "-" "_" target ${target})
    file(APPEND ${CMAKE_BINARY_DIR}/bitcode_libs_generated.cpp
      "static BitcodeLib ${name}(\"${name}.bc\", ISPCTarget::${target}, TargetOS::${fixed_os}, Arch::${fixed_arch});\n")
endfunction()

function(write_common_bitcode_lib name os arch)
    if ("${arch}" STREQUAL "i686")
        set(arch "x86")
    elseif ("${arch}" STREQUAL "armv7")
        set(arch "arm")
    endif()

    file(APPEND ${CMAKE_BINARY_DIR}/bitcode_libs_generated.cpp
      "static BitcodeLib ${name}(\"${name}.bc\", TargetOS::${os}, Arch::${arch});\n")
endfunction()

function(write_dispatch_bitcode_lib name os)
    file(APPEND ${CMAKE_BINARY_DIR}/bitcode_libs_generated.cpp
      "static BitcodeLib ${name}(\"${name}.bc\", TargetOS::${os});\n")
endfunction()

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

function(target_ll_to_cpp target bit os CPP_LIST BC_LIST)
    set(input builtins/target-${target}.ll)
    set(include builtins)
    string(TOUPPER ${os} OS_UP)

    set(name builtins-target-${target}-${bit}bit-${os})
    string(REPLACE "-" "_" name ${name})
    set(cpp ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${name}.cpp)
    set(bc ${BITCODE_FOLDER}/${name}.bc)

    write_target_bitcode_lib(${name} ${target} ${os} ${bit})

    add_custom_command(
        OUTPUT ${bc}
        COMMAND ${M4_EXECUTABLE} -I${include} -DBUILD_OS=${OS_UP} -DRUNTIME=${bit} ${input}
            | \"${LLVM_AS_EXECUTABLE}\" -o ${bc}
        DEPENDS ${input} ${M4_IMPLICIT_DEPENDENCIES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_custom_command(
        OUTPUT ${cpp}
        COMMAND ${Python3_EXECUTABLE} ${BITCODE2CPP} ${bc} --type=ispc-target --runtime=${bit} --os=${OS_UP} ${cpp}
        DEPENDS ${bc} ${BITCODE2CPP}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set(tmp_list_cpp ${${CPP_LIST}})
    list(APPEND tmp_list_cpp ${cpp})
    set(${CPP_LIST} ${tmp_list_cpp} PARENT_SCOPE)

    set(tmp_list_bc ${${BC_LIST}})
    list(APPEND tmp_list_bc ${bc})
    set(${BC_LIST} ${tmp_list_bc} PARENT_SCOPE)
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

    write_dispatch_bitcode_lib(${name} ${os})

    set(EXTRA_OPTS "")
    if (NOT WIN32)
        set(EXTRA_OPTS "-fPIC")
    endif()

    add_custom_command(
        OUTPUT ${bc}
        COMMAND ${CLANGPP_EXECUTABLE} -x c ${DISP_TYPE} ${EXTRA_OPTS} --target=x86_64-unknown-unknown -march=core2 -mtune=generic -O2 -emit-llvm ${input} -c -o ${bc}
        DEPENDS ${input}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_custom_command(
        OUTPUT ${cpp}
        COMMAND ${Python3_EXECUTABLE} ${BITCODE2CPP} ${bc} --type=dispatch --os=${os} ${cpp}
        DEPENDS ${bc} ${BITCODE2CPP}
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

    write_common_bitcode_lib(${name} ${os} ${arch})

    list(APPEND flags
        -DWASM -s WASM_OBJECT_FILES=0 -I${CMAKE_SOURCE_DIR} --std=gnu++17 -S -emit-llvm)
    if("${bit}" STREQUAL "64")
        list(APPEND flags "-sMEMORY64")
    endif()

    add_custom_command(
        OUTPUT ${bc}
        COMMAND ${EMCC_EXECUTABLE} ${flags} ${input} -o -
            | \"${LLVM_AS_EXECUTABLE}\" -o ${bc}
        DEPENDS ${input}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_custom_command(
        OUTPUT ${cpp}
        COMMAND ${Python3_EXECUTABLE} ${BITCODE2CPP} ${bc} --type=builtins-c --runtime=${bit} --os=${os} --arch=${arch} ${cpp}
        DEPENDS ${bc} ${BITCODE2CPP}
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
        -I${CMAKE_SOURCE_DIR} -m${bit} -S -emit-llvm --std=gnu++17
    )

    set(name builtins-cpp-${bit}-${os}-${arch})
    string(REPLACE "-" "_" name ${name})
    set(cpp ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${name}.cpp)
    set(bc ${BITCODE_FOLDER}/${name}.bc)

    write_common_bitcode_lib(${name} ${os} ${arch})

    add_custom_command(
        OUTPUT ${bc}
        COMMAND ${CLANGPP_EXECUTABLE} ${flags} ${input} -o -
            | \"${LLVM_AS_EXECUTABLE}\" -o ${bc}
        DEPENDS ${input}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_custom_command(
        OUTPUT ${cpp}
        COMMAND ${Python3_EXECUTABLE} ${BITCODE2CPP} ${bc} --type=builtins-c --runtime=${bit} --os=${os} --arch=${arch} ${cpp}
        DEPENDS ${bc} ${BITCODE2CPP}
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

    write_common_bitcode_lib(${name} ${os} ${arch})

    add_custom_command(
        OUTPUT ${bc}
        COMMAND cat ${input} | \"${LLVM_AS_EXECUTABLE}\" -o ${bc}
        DEPENDS ${input}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )

    add_custom_command(
        OUTPUT ${cpp}
        COMMAND ${Python3_EXECUTABLE} ${BITCODE2CPP} ${bc} --type=builtins-c --runtime=${bit} --os=${os} --arch=${arch} ${cpp}
        DEPENDS ${bc} ${BITCODE2CPP}
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

    if (MSVC)
        # Group generated files inside Visual Studio
        source_group("Generated Dispatch Builtins" FILES ${DISPATCH_BUILTIN_CPP_FILES})
    endif()
    set_source_files_properties(${DISPATCH_BUILTIN_CPP_FILES} PROPERTIES GENERATED true)

    add_custom_target(dispatch-builtins-bc DEPENDS ${DISPATCH_BUILTIN_BC_FILES})
    add_custom_target(dispatch-builtins-cpp DEPENDS dispatch-builtins-bc)
    set_target_properties(dispatch-builtins-cpp PROPERTIES SOURCES "${DISPATCH_BUILTIN_CPP_FILES}")
    add_dependencies(builtins-cpp dispatch-builtins-cpp)
    add_dependencies(builtins-bc dispatch-builtins-bc)

    set(DISPATCH_BUILTIN_CPP_FILES ${DISPATCH_BUILTIN_CPP_FILES} PARENT_SCOPE)
    set(DISPATCH_BUILTIN_BC_FILES ${DISPATCH_BUILTIN_BC_FILES} PARENT_SCOPE)
endfunction()

function (generate_target_builtins)
    generate_stdlib_or_target_builtins(target_ll_to_cpp dummy TARGET_BUILTIN_CPP_FILES TARGET_BUILTIN_BC_FILES)

    if (MSVC)
        # Group generated files inside Visual Studio
        source_group("Generated Target Builtins" FILES ${TARGET_BUILTIN_CPP_FILES})
    endif()
    set_source_files_properties(${TARGET_BUILTIN_CPP_FILES} PROPERTIES GENERATED true)

    add_custom_target(target-builtins-bc DEPENDS ${TARGET_BUILTIN_BC_FILES})
    add_custom_target(target-builtins-cpp DEPENDS target-builtins-bc)
    set_target_properties(target-builtins-cpp PROPERTIES SOURCES "${TARGET_BUILTIN_CPP_FILES}")
    add_dependencies(builtins-cpp target-builtins-cpp)
    add_dependencies(builtins-bc target-builtins-bc)

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

    if (MSVC)
        # Group generated files inside Visual Studio
        source_group("Generated Common Builtins" FILES ${COMMON_BUILTIN_CPP_FILES})
    endif()
    set_source_files_properties(${COMMON_BUILTIN_CPP_FILES} PROPERTIES GENERATED true)

    add_custom_target(common-builtins-bc DEPENDS ${COMMON_BUILTIN_BC_FILES})
    add_custom_target(common-builtins-cpp DEPENDS common-builtins-bc)
    set_target_properties(common-builtins-cpp PROPERTIES SOURCES "${COMMON_BUILTIN_CPP_FILES}")
    add_dependencies(builtins-cpp common-builtins-cpp)
    add_dependencies(builtins-bc common-builtins-bc)

    set(COMMON_BUILTIN_CPP_FILES ${COMMON_BUILTIN_CPP_FILES} PARENT_SCOPE)
    set(COMMON_BUILTIN_BC_FILES ${COMMON_BUILTIN_BC_FILES} PARENT_SCOPE)
endfunction()

function (generate_builtins)
    file(WRITE ${CMAKE_BINARY_DIR}/bitcode_libs_generated.cpp)

    add_custom_target(builtins-bc)
    add_custom_target(builtins-cpp)

    generate_dispatch_builtins()
    generate_target_builtins()
    generate_common_builtins()

    add_library(builtin OBJECT EXCLUDE_FROM_ALL
        ${DISPATCH_BUILTIN_CPP_FILES} ${COMMON_BUILTIN_CPP_FILES} ${TARGET_BUILTIN_CPP_FILES})
    add_dependencies(builtin builtins-cpp)
endfunction()

# Define custom commands to generate bitcode for a specific generic target and
# a corresponding cpp wrapper for bitcode. All they are generated from
# builtins/generic.ispc file. For slim binary, an entry to the
# bitcode_libs_generated.cpp file is added.
function (generate_generic_target_builtin ispc_name target arch bit os)
    set(include ${CMAKE_CURRENT_SOURCE_DIR}/stdlib/include)

    set(name "builtins_target_generic_${target}_${arch}_${os}")
    set(input_ispc builtins/generic.ispc)

    if ("${os}" STREQUAL "unix")
        set(fixed_os "linux")
        set(OS_UP UNIX)
    elseif ("${os}" STREQUAL "windows")
        set(fixed_os "windows")
        set(OS_UP WINDOWS)
    else()
        message(FATAL_ERROR "Error: unknown OS for target ${os}")
    endif()

    file(APPEND ${CMAKE_BINARY_DIR}/bitcode_libs_generated.cpp
      "static BitcodeLib ${name}(\"${name}.bc\", ISPCTarget::generic_${target}, TargetOS::${fixed_os}, Arch::${arch});\n")
    set(target "generic-${target}")

    set(cpp ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${name}.cpp)
    set(bc ${BITCODE_FOLDER}/${name}.bc)

    if ("${fixed_os}" STREQUAL "linux" AND NOT ISPC_LINUX_TARGET)
        # If ISPC_LINUX_TARGET is disabled then we can't run ispc-slim with
        # --target-os=linux to generate generic target bitcode. So we need pass
        # the target-os that is supported by ispc-slim.
        if (APPLE)
            set(fixed_os "macos")
        elseif (CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
            set(fixed_os "freebsd")
        else ()
            message(FATAL_ERROR "Error: unknown OS for target ${fixed_os}")
        endif()
    endif()

    add_custom_command(
        OUTPUT ${bc}
        COMMAND ${ispc_name} -I ${include} --enable-llvm-intrinsics --nostdlib --gen-stdlib --target=${target} --arch=${arch} --target-os=${fixed_os}  --emit-llvm -o ${bc} ${input_ispc} -DBUILD_OS=${OS_UP} -DRUNTIME=${bit}
        DEPENDS ${ispc_name} ${input_ispc}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_custom_command(
        OUTPUT ${cpp}
        COMMAND ${Python3_EXECUTABLE} ${BITCODE2CPP} ${bc} --type=ispc-target --runtime=${bit} --os=${OS_UP} --arch=${arch} ${cpp}
        DEPENDS ${bc} ${BITCODE2CPP}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    list(APPEND GENERIC_TARGET_BC_FILE ${bc})
    set(GENERIC_TARGET_BC_FILE ${GENERIC_TARGET_BC_FILE} PARENT_SCOPE)

    list(APPEND GENERIC_TARGET_CPP_FILE ${cpp})
    set(GENERIC_TARGET_CPP_FILE ${GENERIC_TARGET_CPP_FILE} PARENT_SCOPE)
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

    # "generic-i1x64" is not supported due to #3123
    list(APPEND TARGET_LIST
        "i1x4"
        "i1x8"
        "i1x16"
        "i1x32"
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
        if (NOT APPLE)
            list(APPEND ARCH_LIST
                "x86,32"
            )
        endif()
    endif()

    if (ARM_ENABLED)
        list(APPEND ARCH_LIST
            "aarch64,64"
        )
        if (NOT APPLE AND NOT WIN32)
            list(APPEND ARCH_LIST
                "arm,32"
            )
        endif()
    endif()

    foreach(os ${os_list})
        foreach(target ${TARGET_LIST})
            foreach(pair ${ARCH_LIST})
                string(REGEX REPLACE "," ";" pair_split ${pair})
                list(GET pair_split 0 arch)
                list(GET pair_split 1 bit)
                generate_generic_target_builtin(${ispc_name} ${target} ${arch} ${bit} ${os})
            endforeach()
        endforeach()
    endforeach()

    add_custom_target(generic-target-bc DEPENDS ${GENERIC_TARGET_BC_FILE})
    add_custom_target(generic-target-cpp DEPENDS ${GENERIC_TARGET_CPP_FILE})

    add_library(generic-target OBJECT EXCLUDE_FROM_ALL ${GENERIC_TARGET_CPP_FILE})
endfunction()
