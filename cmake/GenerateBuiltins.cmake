#
#  Copyright (c) 2018-2020, Intel Corporation
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
#   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
# ispc GenerateBuiltins.cmake
#
find_program(M4_EXECUTABLE m4)
    if (NOT M4_EXECUTABLE)
        message(FATAL_ERROR "Failed to find M4 macro processor" )
    endif()
    message(STATUS "M4 macro processor: " ${M4_EXECUTABLE})

if (WIN32)
    set(TARGET_OS_LIST_FOR_LL "windows" "unix")
elseif (UNIX)
    set(TARGET_OS_LIST_FOR_LL "unix")
endif()

function(ll_to_cpp llFileName bit os_name resultFileName)
    set(inputFilePath builtins/${llFileName}.ll)
    set(includePath builtins)
    if ("${llFileName}" STREQUAL "dispatch")
        set(type dispatch)
    else ()
        set(type ispc-target)
    endif()
    string(TOUPPER ${os_name} os_name_macro)
    if ("${bit}" STREQUAL "")
        set(output ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/builtins-${llFileName}-${os_name}.cpp)
        add_custom_command(
            OUTPUT ${output}
            COMMAND ${M4_EXECUTABLE} -I${includePath}
                -DLLVM_VERSION=${LLVM_VERSION} -DBUILD_OS=${os_name_macro} ${inputFilePath}
                | \"${Python3_EXECUTABLE}\" bitcode2cpp.py ${inputFilePath} --type=${type} --os=${os_name_macro} --llvm_as ${LLVM_AS_EXECUTABLE}
                > ${output}
            DEPENDS ${inputFilePath} bitcode2cpp.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    else ()
        set(output ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/builtins-${llFileName}-${bit}bit-${os_name}.cpp)
        add_custom_command(
            OUTPUT ${output}
            COMMAND ${M4_EXECUTABLE} -I${includePath}
                -DLLVM_VERSION=${LLVM_VERSION} -DBUILD_OS=${os_name_macro} -DRUNTIME=${bit} ${inputFilePath}
                | \"${Python3_EXECUTABLE}\" bitcode2cpp.py ${inputFilePath} --type=${type} --runtime=${bit} --os=${os_name_macro} --llvm_as ${LLVM_AS_EXECUTABLE}
                > ${output}
            DEPENDS ${inputFilePath} bitcode2cpp.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
    set(${resultFileName} ${output} PARENT_SCOPE)
    set_source_files_properties(${resultFileName} PROPERTIES GENERATED true)
endfunction()

function(builtin_to_cpp bit os_name arch supported_archs supported_oses resultFileName)
    set(inputFilePath builtins/builtins.c)
    set(includePath "")
    set(SKIP OFF)
    if (NOT ${arch} IN_LIST supported_archs OR NOT ${os_name} IN_LIST supported_oses)
        set(SKIP ON)
    endif()

    if ((    ${os_name} STREQUAL "web" AND NOT ${arch} STREQUAL "wasm32") OR
        (NOT ${os_name} STREQUAL "web" AND     ${arch} STREQUAL "wasm32") OR
        (    ${os_name} STREQUAL "web" AND     ${arch} STREQUAL "wasm32" AND NOT "${bit}" STREQUAL "32"))
        return()
    endif()

    if ("${bit}" STREQUAL "32" AND ${arch} STREQUAL "x86")
        set(target_arch "i386")
    elseif ("${bit}" STREQUAL "64" AND ${arch} STREQUAL "x86")
        set(target_arch "x86_64")
    elseif ("${bit}" STREQUAL "32" AND ${arch} STREQUAL "arm")
        set(target_arch "armv7")
    elseif ("${bit}" STREQUAL "64" AND ${arch} STREQUAL "arm")
        set(target_arch "aarch64")
    elseif ("${bit}" STREQUAL "32" AND ${arch} STREQUAL "wasm32")
        set(target_arch "wasm32")
    else()
        message(FATAL_ERROR "Error")
    endif()

    # Special case when ISPC is built on ARM system:
    string(FIND ${CMAKE_HOST_SYSTEM_PROCESSOR} "armv" IS_ARM_SYSTEM)
    string(FIND ${CMAKE_HOST_SYSTEM_PROCESSOR} "aarch64" IS_AARCH64_SYSTEM)
    if ((((${IS_ARM_SYSTEM} GREATER -1) AND NOT (${target_arch} STREQUAL "armv7"))) OR
        ((${IS_AARCH64_SYSTEM} GREATER -1) AND NOT (${target_arch} STREQUAL "aarch64")))
            set(SKIP ON)
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
            elseif(${target_arch} STREQUAL "i386")
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
        endif()
    endif()

    # Host to target OS constrains
    if (WIN32)
        if (${os_name} STREQUAL "ios")
            set(SKIP ON)
        endif()
    elseif (APPLE)
        if (${os_name} STREQUAL "windows")
            set(SKIP ON)
        elseif (${os_name} STREQUAL "ps4")
            set(SKIP ON)
        endif()
    else()
        if (${os_name} STREQUAL "windows")
            set(SKIP ON)
        elseif (${os_name} STREQUAL "ps4")
            set(SKIP ON)
        elseif (${os_name} STREQUAL "ios")
            set(SKIP ON)
        endif()
    endif()

    # OS to arch constrains
    if (${os_name} STREQUAL "windows" AND ${arch} STREQUAL "arm")
        set(SKIP ON)
    endif()
    if (${os_name} STREQUAL "macos" AND NOT ${target_arch} STREQUAL "x86_64")
        set(SKIP ON)
    endif()
    if (${os_name} STREQUAL "ps4" AND NOT ${target_arch} STREQUAL "x86_64")
        set(SKIP ON)
    endif()
    if (${os_name} STREQUAL "ios")
        if (${target_arch} STREQUAL "aarch64")
            set(target_arch "arm64")
        else()
            set(SKIP ON)
        endif()
    endif()
    # FreeBSD seems to prefer using amd64 over x86_64 naming, but it's nothing more than an alias.

    # Determine triple
    if (${os_name} STREQUAL "windows")
        set(target_flags --target=${target_arch}-pc-win32 ${includePath})
    elseif (${os_name} STREQUAL "linux")
        if (${target_arch} STREQUAL "i386" OR ${target_arch} STREQUAL "x86_64" OR ${target_arch} STREQUAL "aarch64")
            set(target_flags --target=${target_arch}-unknown-linux-gnu -fPIC ${includePath})
        elseif (${target_arch} STREQUAL "armv7")
            set(target_flags --target=${target_arch}-unknown-linux-gnueabihf -fPIC ${includePath})
        else()
            message(FATAL_ERROR "Error")
        endif()
    elseif (${os_name} STREQUAL "freebsd")
        set(target_flags --target=${target_arch}-unknown-freebsd -fPIC ${includePath})
    elseif (${os_name} STREQUAL "macos")
        set(target_flags --target=${target_arch}-apple-macosx ${includePath})
    elseif (${os_name} STREQUAL "android")
        set(target_flags --target=${target_arch}-unknown-linux-android -fPIC ${includePath})
    elseif (${os_name} STREQUAL "ios")
        set(target_flags --target=${target_arch}-apple-ios ${includePath})
    elseif (${os_name} STREQUAL "ps4")
        set(target_flags --target=${target_arch}-scei-ps4 -fPIC ${includePath})
    elseif (${os_name} STREQUAL "web")
        set(target_flags --target=${target_arch}-unknown-unknown -fPIC ${includePath})
    else()
        message(FATAL_ERROR "Error")
    endif()

    # Just invert SKIP to print supported targets
    if (NOT ${SKIP})
        set (SUPPORT ON)
    else()
        set (SUPPORT OFF)
    endif()
    message (STATUS "${os_name} for ${target_arch} is ${SUPPORT}")

    if (NOT SKIP)
        set(output ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/builtins-c-${bit}-${os_name}-${target_arch}.cpp)
        if (${os_name} STREQUAL "web")
            add_custom_command(
                OUTPUT ${output}
                COMMAND ${EMCC_EXECUTABLE} -DWASM -s WASM_OBJECT_FILES=0 -c ${inputFilePath} -emit-llvm -c -o -
                    | (\"${LLVM_DIS_EXECUTABLE}\" - || echo "builtins.c compile error")
                    | \"${Python3_EXECUTABLE}\" bitcode2cpp.py c --type=builtins-c --runtime=${bit} --os=${os_name} --arch=${target_arch} --llvm_as ${LLVM_AS_EXECUTABLE}
                    > ${output}
                DEPENDS ${inputFilePath} bitcode2cpp.py
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )
        else()
            add_custom_command(
                OUTPUT ${output}
                COMMAND ${CLANG_EXECUTABLE} ${target_flags} -w -m${bit} -emit-llvm -c ${inputFilePath} -o - | (\"${LLVM_DIS_EXECUTABLE}\" - || echo "builtins.c compile error")
                    | \"${Python3_EXECUTABLE}\" bitcode2cpp.py c --type=builtins-c --runtime=${bit} --os=${os_name} --arch=${target_arch} --llvm_as ${LLVM_AS_EXECUTABLE}
                    > ${output}
                DEPENDS ${inputFilePath} bitcode2cpp.py
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )
        endif()
        
        set(${resultFileName} ${output} PARENT_SCOPE)
        set_source_files_properties(${resultFileName} PROPERTIES GENERATED true)
    endif()
endfunction()

function (generate_target_builtins resultList)
    # Dispatch module for Windows and/or Unix targets.
    foreach (os_name ${TARGET_OS_LIST_FOR_LL})
        ll_to_cpp(dispatch "" ${os_name} output${os_name})
        list(APPEND tmpList ${output${os_name}})
        if(MSVC)
            # Group generated files inside Visual Studio
            source_group("Generated Builtins" FILES ${output}${os_name})
        endif()
    endforeach()
    # "Regular" targets, targeting specific real ISA: sse/avx/neon
    set(regular_targets ${ARGN})
    list(FILTER regular_targets EXCLUDE REGEX wasm)
    foreach (ispc_target ${regular_targets})
        foreach (bit 32 64)
            foreach (os_name ${TARGET_OS_LIST_FOR_LL})
                ll_to_cpp(target-${ispc_target} ${bit} ${os_name} output${os_name}${bit})
                list(APPEND tmpList ${output${os_name}${bit}})
                if(MSVC)
                    # Group generated files inside Visual Studio
                    source_group("Generated Builtins" FILES ${output${os_name}${bit}})
                endif()
            endforeach()
        endforeach()
    endforeach()
    # WASM targets.
    if (WASM_ENABLED)
        set(wasm_targets ${ARGN})
        list(FILTER wasm_targets INCLUDE REGEX wasm)
        foreach (wasm_target ${wasm_targets})
            ll_to_cpp(target-${wasm_target} 32 web outputweb32)
            list(APPEND tmpList ${outputweb32})
        endforeach()
    endif()
    # Return the list
    set(${resultList} ${tmpList} PARENT_SCOPE)
endfunction()

function (generate_common_builtins resultList)
    list (APPEND supported_archs "x86")
    if (ARM_ENABLED)
        list (APPEND supported_archs "arm")
    endif()
    if (WASM_ENABLED)
        list (APPEND supported_archs "wasm32")
        list (APPEND supported_oses "web")
    endif()
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
    if (ISPC_PS4_TARGET)
        list (APPEND supported_oses "ps4")
    endif()
    message (STATUS "ISPC will be built with support of ${supported_oses} for ${supported_archs}")
    foreach (bit 32 64)
        foreach (os_name "windows" "linux" "freebsd" "macos" "android" "ios" "ps4" "web")
            foreach (arch "x86" "arm" "wasm32")
                builtin_to_cpp(${bit} ${os_name} ${arch} "${supported_archs}" "${supported_oses}" res${bit}${os_name}${arch})
                list(APPEND tmpList ${res${bit}${os_name}${arch}} )
                if(MSVC)
                    # Group generated files inside Visual Studio
                    source_group("Generated Builtins" FILES ${res${bit}${os_name}${arch}})
                endif()
            endforeach()
        endforeach()
    endforeach()
    set(${resultList} ${tmpList} PARENT_SCOPE)
endfunction()
