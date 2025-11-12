#
#  Copyright (c) 2018-2025, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc Stdlib.cmake
#

function(write_stdlib_bitcode_lib name target os bit out_arch out_os bitcode_filename)
    # If bitcode_filename is empty, default to ${name}.bc
    if("${bitcode_filename}" STREQUAL "")
        set(bitcode_filename "${name}.bc")
    endif()

    determine_arch_and_os(${target} ${bit} ${os} fixed_arch fixed_os)
    string(REPLACE "-" "_" target_enum ${target})
    file(APPEND ${CMAKE_BINARY_DIR}/bitcode_libs_generated.cpp
      "static BitcodeLib ${name}(BitcodeLib::BitcodeLibType::Stdlib, \"${bitcode_filename}\", ISPCTarget::${target_enum}, TargetOS::${fixed_os}, Arch::${fixed_arch});\n")
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

# Process a stdlib family: register all targets in family to use base stdlib
# Uses unified families from cmake/StdlibFamilies.cmake
function (process_stdlib_family width_family ispc_name bit os CPP_LIST BC_LIST)
    # Get family members and base target from StdlibFamilies.cmake
    set(family_targets ${STDLIB_FAMILY_${width_family}})
    set(base ${STDLIB_FAMILY_BASE_${width_family}})

    if(NOT family_targets)
        message(WARNING "No targets found for width family ${width_family}")
        return()
    endif()

    # Determine bitcode filename based on family type
    # Generic families: stdlib_generic_<width>_<arch>_<os>.bc (uses architecture)
    # Inheritance families: stdlib_<isa>_<width>_<bit>bit_<os>.bc (uses bit, matches stdlib_to_cpp)
    string(REPLACE "-" "_" base_safe "${base}")

    if(base MATCHES "^generic-")
        # Generic family: base is like "generic-i1x16"
        # Files are named: stdlib_generic_i1x16_x86_64_unix.bc
        # Need to determine architecture for this family
        list(GET family_targets 0 first_target)
        determine_arch_and_os(${first_target} ${bit} ${os} family_arch family_os)
        # Use original 'os' parameter (unix/windows), not canonicalized family_os (linux/windows)
        set(bitcode_filename "stdlib_${base_safe}_${family_arch}_${os}.bc")
    else()
        # Inheritance family: base is a real ISA like "avx1-i32x8"
        # Files are named: stdlib_avx1_i32x8_64bit_unix.bc (matches stdlib_to_cpp output)
        set(bitcode_filename "stdlib_${base_safe}_${bit}bit_${os}.bc")
    endif()

    # Register all family members to use this stdlib
    foreach(target ${family_targets})
        should_skip_target_for_os(${target} ${os} ${bit} skip_target)
        if(skip_target)
            continue()
        endif()

        set(target_name stdlib-${target}-${bit}bit-${os})
        string(REPLACE "-" "_" target_name "${target_name}")
        write_stdlib_bitcode_lib(${target_name} ${target} ${os} ${bit} dummy_arch dummy_os ${bitcode_filename})
    endforeach()

    # No need to propagate BC/CPP files - base stdlib is already built separately
    set(${CPP_LIST} ${${CPP_LIST}} PARENT_SCOPE)
    set(${BC_LIST} ${${BC_LIST}} PARENT_SCOPE)
endfunction()

function (stdlib_to_cpp ispc_name target bit os CPP_LIST BC_LIST)
    set(name stdlib-${target}-${bit}bit-${os})
    string(REPLACE "-" "_" name "${name}")
    set(cpp ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${name}.cpp)
    set(bc ${BITCODE_FOLDER}/${name}.bc)

    should_skip_target_for_os(${target} ${os} ${bit} skip_target)
    if(skip_target)
        return()
    endif()
    if(target MATCHES "^avx10_[0-9]+")
        # Replace the first underscore after "avx10" with a dot for file lookup
        string(REGEX REPLACE "^(avx10)_([0-9]+)" "\\1.\\2" target_for_file "${target}")
    else()
        set(target_for_file "${target}")
    endif()
    # define canon_os and arch
    write_stdlib_bitcode_lib(${name} ${target} ${os} ${bit} canon_arch canon_os "")

    set(SRC_INCLUDE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/stdlib/include)

    add_custom_command(
        OUTPUT ${bc}
        COMMAND ${ispc_name} -I ${SRC_INCLUDE_FOLDER} --nostdlib --gen-stdlib --target=${target_for_file} --arch=${canon_arch} --target-os=${canon_os} stdlib/stdlib.ispc --emit-llvm -o ${bc}
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
    set(src ${INCLUDE_FOLDER}/stdlib/${header})
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
        set(dest ${INCLUDE_FOLDER}/stdlib/${header})
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
