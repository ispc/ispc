#
#  Copyright (c) 2019, Intel Corporation
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

message(STATUS "Processing cmake/AddPerfExample.cmake")
function(add_perf_example)
    set(options CM_TEST)
    set(oneValueArgs NAME ISPC_SRC_NAME CM_SRC_NAME ISPC_OBJ_NAME HOST_NAME CM_HOST_NAME CM_OBJ_NAME TEST_NAME)
    set(multiValueArgs ISPC_FLAGS HOST_SOURCES CM_HOST_SOURCES CM_HOST_FLAGS)
    cmake_parse_arguments("parsed" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
# compile ispc kernel    
    list(APPEND ISPC_BUILD_OUTPUT ${parsed_ISPC_OBJ_NAME})
    add_custom_command(
                       OUTPUT ${ISPC_BUILD_OUTPUT}
                       COMMAND ${ISPC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${parsed_ISPC_SRC_NAME}.ispc ${parsed_ISPC_FLAGS} --target=genx -o ${parsed_ISPC_OBJ_NAME}
                       VERBATIM
                       DEPENDS ${ISPC_EXECUTABLE}
                      )
    add_custom_target(${parsed_TEST_NAME} ALL DEPENDS ${ISPC_BUILD_OUTPUT})

    if (NOT HOST_SOURCES)
        return()
    endif()

    if (parsed_CM_TEST)
        set(parsed_TEST_NAME "${parsed_TEST_NAME}_cm")
    endif()

# compile host code
    set(HOST_EXECUTABLE "host_${parsed_TEST_NAME}")
    if (WIN32)
        add_executable(${HOST_EXECUTABLE} ${parsed_HOST_SOURCES})
        target_include_directories(${HOST_EXECUTABLE} PRIVATE "${COMMON_PATH}"
                                   "${MDF}/compiler/include_icl"
                                   "${MDF}/compiler/include_icl/cm"
                                   "${MDF}/runtime/include"
                                   "${MDF}/examples/helper")

        target_link_libraries(${HOST_EXECUTABLE} "${MDF}/runtime/lib/x86/igfx11cmrt32.lib"
                              "${MDF}/compiler/lib/x86/libcm.lib" AdvAPI32 Ole32)
        install(TARGETS "${HOST_EXECUTABLE}" RUNTIME DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    else()
# L0 build
        # TODO: check exists
        set(SYCL_CLANG_EXECUTABLE "${SYCL}/bin/clang++")
        add_custom_target(${HOST_EXECUTABLE} ALL
            COMMAND ${SYCL_CLANG_EXECUTABLE} -fsycl -isystem ${COMMON_PATH} ${parsed_HOST_SOURCES} -o ${HOST_EXECUTABLE} -L${SYCL}/lib -llevel_zero 
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            VERBATIM
        )
    endif()
    # compile cm kernel if present
    if (parsed_CM_TEST AND MDF)
        set(CM_TEST_NAME "${parsed_TEST_NAME}")
        list(APPEND CM_BUILD_OUTPUT ${parsed_CM_OBJ_NAME})

        if (WIN32)
            set(CMC_SUPPORT_INCLUDE ${CMC_INSTALL_PATH}/include)
        endif()

        add_custom_target(${parsed_CM_SRC_NAME} ALL
            COMMAND ${CM_EXECUTABLE} -isystem ${CMC_SUPPORT_INCLUDE} -march=SKL "/DCM_PTRSIZE=32" ${CMAKE_CURRENT_SOURCE_DIR}/${parsed_CM_SRC_NAME}.cpp -o ${parsed_CM_OBJ_NAME}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            VERBATIM
            DEPENDS ${CM_EXECUTABLE}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${parsed_CM_SRC_NAME}.cpp
        )

        set(CM_HOST_EXECUTABLE ${parsed_CM_HOST_NAME}_${parsed_TEST_NAME})
        add_executable(${CM_HOST_EXECUTABLE} ${parsed_CM_HOST_SOURCES})
        if (WIN32)
            target_include_directories(${CM_HOST_EXECUTABLE} PRIVATE "${COMMON_PATH}"
                                       "${MDF}/compiler/include_icl"
                                       "${MDF}/compiler/include_icl/cm"
                                       "${MDF}/runtime/include"
                                       "${MDF}/examples/helper")

            target_link_libraries(${CM_HOST_EXECUTABLE} "${MDF}/runtime/lib/x86/igfx11cmrt32.lib"
                                  "${MDF}/compiler/lib/x86/libcm.lib" AdvAPI32 Ole32)
        endif()
        target_compile_options(${CM_HOST_EXECUTABLE} PRIVATE ${parsed_CM_HOST_FLAGS})
        install(TARGETS "${CM_HOST_EXECUTABLE}" RUNTIME DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

    endif()

endfunction()
