#
#  Copyright (c) 2018-2024, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc Stdlib.cmake
#

function(write_stdlib_bitcode_lib name target os bit out_arch out_os)
    determine_arch_and_os(${target} ${bit} ${os} fixed_arch fixed_os)
    string(REPLACE "-" "_" target ${target})
    file(APPEND ${CMAKE_BINARY_DIR}/bitcode_libs_generated.cpp
      "static BitcodeLib ${name}(BitcodeLib::BitcodeLibType::Stdlib, \"${name}.bc\", ISPCTarget::${target}, TargetOS::${fixed_os}, Arch::${fixed_arch});\n")
    if ("${fixed_os}" STREQUAL "linux" AND NOT ISPC_LINUX_TARGET)
        # If ISPC_LINUX_TARGET is disabled then we can't run ispc-slim with
        # --target-os=linux to generate stdlib bitcode. So we need pass the
        # target-os that is supported by ispc-slim.
        if (APPLE)
            set(fixed_os "macos")
        elseif (CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
            set(fixed_os "freebsd")
        endif()
    endif()
    set(${out_os} ${fixed_os} PARENT_SCOPE)
    set(${out_arch} ${fixed_arch} PARENT_SCOPE)
endfunction()

function (stdlib_to_cpp ispc_name target bit os CPP_LIST BC_LIST)
    set(name stdlib-${target}-${bit}bit-${os})
    string(REPLACE "-" "_" name "${name}")
    set(cpp ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${name}.cpp)
    set(bc ${BITCODE_FOLDER}/${name}.bc)

    if ("${os}" STREQUAL "unix" AND APPLE AND NOT ISPC_LINUX_TARGET)
        # macOS target supports only x86_64 and aarch64
        if ("${bit}" STREQUAL "32")
            return()
        endif()
        # ISPC doesn't support avx512spr targets on macOS
        if ("${target}" MATCHES "avx512spr")
            return()
        endif()
    endif()

    # define canon_os and arch
    write_stdlib_bitcode_lib(${name} ${target} ${os} ${bit} canon_arch canon_os)

    set(INCLUDE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/stdlib/include)

    add_custom_command(
        OUTPUT ${bc}
        COMMAND ${ispc_name} -I ${INCLUDE_FOLDER} --nostdlib --gen-stdlib --target=${target} --arch=${canon_arch} --target-os=${canon_os} stdlib/stdlib.ispc --emit-llvm -o ${bc}
        DEPENDS ${ispc_name} ${STDLIB_ISPC_FILES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    string(TOUPPER ${os} OS_UP)
    add_custom_command(
        OUTPUT ${cpp}
        COMMAND ${Python3_EXECUTABLE} ${BITCODE2CPP} ${bc} --type=stdlib --runtime=${bit} --os=${OS_UP} ${cpp}
        DEPENDS ${BITCODE2CPP} ${bc}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set(tmp_list_bc ${${BC_LIST}})
    list(APPEND tmp_list_bc ${bc})
    set(${BC_LIST} ${tmp_list_bc} PARENT_SCOPE)

    set(tmp_list_cpp ${${CPP_LIST}})
    list(APPEND tmp_list_cpp ${cpp})
    set(${CPP_LIST} ${tmp_list_cpp} PARENT_SCOPE)
endfunction()

function (generate_stdlibs_1 ispc_name)
    generate_stdlib_or_target_builtins(stdlib_to_cpp ${ispc_name} STDLIB_CPP_FILES STDLIB_BC_FILES)

    set(STDLIB_BC_FILES ${STDLIB_BC_FILES} PARENT_SCOPE)
    set(STDLIB_CPP_FILES ${STDLIB_CPP_FILES} PARENT_SCOPE)
endfunction()

function (stdlib_header_cpp name)
    set(src ${INCLUDE_FOLDER}/${header})
    string(REPLACE "." "_" header ${header})
    set(cpp ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${header}.cpp)

    add_custom_command(
        OUTPUT ${cpp}
        COMMAND ${Python3_EXECUTABLE} ${BITCODE2CPP} ${src} --type=header ${cpp}
        DEPENDS ${BITCODE2CPP} ${src}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set(tmp_list ${STDLIB_HEADERS_CPP})
    list(APPEND tmp_list ${cpp})
    set(STDLIB_HEADERS_CPP ${tmp_list} PARENT_SCOPE)
endfunction()

function (stdlib_headers)
    foreach (header ${STDLIB_HEADERS})
        set(target_name stdlib-${header})
        set(src ${CMAKE_SOURCE_DIR}/stdlib/include/${header})
        set(dest ${INCLUDE_FOLDER}/${header})
        list(APPEND stdlib_headers_list ${dest})

        add_custom_command(
            OUTPUT ${dest}
            DEPENDS ${src}
            COMMAND ${CMAKE_COMMAND} -E copy ${src} ${dest})

        stdlib_header_cpp(${header})
    endforeach()

    add_custom_target(stdlib-headers ALL DEPENDS ${stdlib_headers_list})

    set(STDLIB_HEADERS_CPP ${STDLIB_HEADERS_CPP} PARENT_SCOPE)
endfunction()

function (generate_stdlibs ispc_name)
    stdlib_headers()

    add_custom_target(stdlib-headers-cpp DEPENDS ${STDLIB_HEADERS_CPP})
    set_target_properties(stdlib-headers-cpp PROPERTIES SOURCE "${STDLIB_HEADERS_CPP}")

    generate_stdlibs_1(${ispc_name})

    add_custom_target(stdlibs-bc DEPENDS ${STDLIB_BC_FILES})
    # TODO! stdlibs-cpp is kind of empty
    add_custom_target(stdlibs-cpp DEPENDS stdlibs-bc)
    set_target_properties(stdlibs-cpp PROPERTIES SOURCE "${STDLIB_CPP_FILES}")

    add_library(stdlib OBJECT EXCLUDE_FROM_ALL ${STDLIB_CPP_FILES} ${STDLIB_HEADERS_CPP})
    add_dependencies(stdlib stdlibs-cpp)
    add_dependencies(stdlib stdlib-headers-cpp)

    if (MSVC)
        source_group("Generated Include Files" FILES ${STDLIB_HEADERS_CPP})
        source_group("Generated Stdlib Files" FILES ${STDLIB_CPP_FILES})
    endif()
    set_source_files_properties(${STDLIB_HEADERS_CPP} PROPERTIES GENERATED true)
    set_source_files_properties(${STDLIB_CPP_FILES} PROPERTIES GENERATED true)
endfunction()
