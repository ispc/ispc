#
#  Copyright (c) 2019-2020, Intel Corporation
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

function(add_perf_example)
    set(options CM_TEST)
    set(oneValueArgs NAME ISPC_SRC_NAME ISPC_TARGET CM_SRC_NAME ISPC_OBJ_NAME HOST_NAME CM_HOST_NAME CM_OBJ_NAME TEST_NAME CM_TEST_NAME)
    set(multiValueArgs ISPC_FLAGS HOST_SOURCES CM_HOST_SOURCES CM_HOST_FLAGS)
    cmake_parse_arguments("parsed" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # Compile ISPC kernel
    list(APPEND ISPC_BUILD_OUTPUT ${parsed_ISPC_OBJ_NAME})
    add_custom_command(OUTPUT ${ISPC_BUILD_OUTPUT}
                       COMMAND ${ISPC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${parsed_ISPC_SRC_NAME}.ispc ${parsed_ISPC_FLAGS} --target=${parsed_ISPC_TARGET} -o ${parsed_ISPC_OBJ_NAME}
                       VERBATIM
                       DEPENDS ${ISPC_EXECUTABLE}
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${parsed_ISPC_SRC_NAME}.ispc)
    set_source_files_properties(${ISPC_BUILD_OUTPUT} PROPERTIES GENERATED true)
    # To show ispc source in VS solution:
    if (WIN32)
        set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/${parsed_ISPC_SRC_NAME}.ispc" PROPERTIES HEADER_FILE_ONLY TRUE)
    endif()
    # Compile host code
    if (NOT HOST_SOURCES)
        return()
    endif()
    set(HOST_EXECUTABLE "host_${parsed_TEST_NAME}")
    add_executable(${HOST_EXECUTABLE} ${parsed_HOST_SOURCES} ${ISPC_BUILD_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/${parsed_ISPC_SRC_NAME}.ispc)
    if (WIN32)
        target_compile_options(${HOST_EXECUTABLE} PRIVATE /nologo /DCM_DX11 /EHsc /D_CRT_SECURE_NO_WARNINGS /Zi /DWIN32)
        target_include_directories(${HOST_EXECUTABLE} PRIVATE "${COMMON_PATH}"
                                   "${CMC_INCLUDE_PATH}"
                                   "${MDF_ROOT}/runtime/include"
                                   "${MDF_ROOT}/examples/helper")
        target_link_libraries(${HOST_EXECUTABLE} "${MDF_ROOT}/runtime/lib/x64/igfx11cmrt64.lib" AdvAPI32 Ole32)
        file(COPY "${MDF_ROOT}/runtime/lib/x64/igfx11cmrt64.dll" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        set_target_properties(${HOST_EXECUTABLE} PROPERTIES FOLDER "GEN_Examples")
    else()
        # L0 build
        target_compile_definitions(${HOST_EXECUTABLE} PRIVATE LZERO)
        target_include_directories(${HOST_EXECUTABLE} PRIVATE "${COMMON_PATH}"
                                   ${NEO_INSTALL_PATH}/neo/usr/local/include)
        if (NEO_LOCAL_BUILD)
            target_link_libraries(${HOST_EXECUTABLE} igdgmm -L${NEO_INSTALL_PATH}/neo/lib/extra)
        endif()
        target_link_libraries(${HOST_EXECUTABLE} rt m dl tbb ze_loader -L${NEO_INSTALL_PATH}/neo/usr/local/lib)
        set_target_properties(${HOST_EXECUTABLE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()
    install(TARGETS "${HOST_EXECUTABLE}" RUNTIME DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

    # Compile CM kernel if present
    if (parsed_CM_TEST AND CMC_EXECUTABLE)
        set(parsed_TEST_NAME "${parsed_TEST_NAME}_cm")
        set(CM_TEST_NAME "${parsed_TEST_NAME}")
        list(APPEND CM_BUILD_OUTPUT ${parsed_CM_OBJ_NAME})
        set(CM_HOST_BINARY "host_${parsed_TEST_NAME}")
        add_executable(${CM_HOST_BINARY} ${parsed_CM_HOST_SOURCES} ${parsed_CM_OBJ_NAME})
        if (WIN32)
            add_custom_command(OUTPUT ${parsed_CM_OBJ_NAME}
                COMMAND ${CMC_EXECUTABLE} -isystem ${CMC_INCLUDE_PATH} -march=SKL "/DCM_PTRSIZE=64" ${CMAKE_CURRENT_SOURCE_DIR}/${parsed_CM_SRC_NAME}.cpp -o ${parsed_CM_OBJ_NAME}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                VERBATIM
                DEPENDS ${CMC_EXECUTABLE}
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${parsed_CM_SRC_NAME}.cpp
            )

            target_compile_options(${CM_HOST_BINARY} PRIVATE /nologo /DCM_DX11 /EHsc /D_CRT_SECURE_NO_WARNINGS /Zi /DWIN32 ${parsed_CM_HOST_FLAGS})
            target_include_directories(${CM_HOST_BINARY} PRIVATE "${COMMON_PATH}"
                                       "${CMC_INCLUDE_PATH}"
                                       "${MDF_ROOT}/runtime/include"
                                       "${MDF_ROOT}/examples/helper")
            target_link_libraries(${CM_HOST_BINARY} "${MDF_ROOT}/runtime/lib/x64/igfx11cmrt64.lib" AdvAPI32 Ole32)
            file(COPY "${MDF_ROOT}/runtime/lib/x64/igfx11cmrt64.dll" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
            set_target_properties(${CM_HOST_BINARY} PROPERTIES FOLDER "GEN_Examples")
        else()
            add_custom_command(
               OUTPUT ${parsed_CM_OBJ_NAME}
               COMMAND ${CMC_EXECUTABLE} -march=SKL -fcmocl "/DCM_PTRSIZE=64" -emit-spirv -o ${parsed_CM_OBJ_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${parsed_CM_SRC_NAME}.cpp
               VERBATIM
               DEPENDS ${CMC_EXECUTABLE}
            )
            # L0 build
            target_compile_definitions(${CM_HOST_BINARY} PRIVATE LZERO CMKERNEL)
            target_include_directories(${CM_HOST_BINARY} PRIVATE "${COMMON_PATH}"
                                   ${NEO_INSTALL_PATH}/neo/usr/local/include)
            if (NEO_LOCAL_BUILD)
                target_link_libraries(${CM_HOST_BINARY} igdgmm -L${NEO_INSTALL_PATH}/neo/lib/extra)
            endif()
            target_link_libraries(${CM_HOST_BINARY} rt m dl tbb ze_loader -L${NEO_INSTALL_PATH}/neo/usr/local/lib)
            set_target_properties(${CM_HOST_BINARY} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        endif()
        install(TARGETS "${CM_HOST_BINARY}" RUNTIME DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    endif()
endfunction()
