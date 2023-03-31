#
#  Copyright (c) 2019-2023, Intel Corporation
#  All rights reserved.
#
#  SPDX-License-Identifier: BSD-3-Clause

enable_testing()
# Default test runner
configure_file(runner.py.in runner.py)
set(TEST_RUNNER "${CMAKE_CURRENT_BINARY_DIR}/runner.py")

function(test_add)
    set(options TEST_IS_CM TEST_IS_ISPC TEST_IS_CM_RUNTIME TEST_IS_ISPCRT_RUNTIME TEST_IS_DPCPP)
    set(oneValueArgs NAME RES_IMAGE REF_IMAGE IMAGE_CMP_TH DPCPP_SPV)
    cmake_parse_arguments("PARSED_ARGS" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # If test is written with TEST_IS_CM_RUNTIME it is supported on Windows only
    set(SUPPORTED 1)
    if (PARSED_ARGS_TEST_IS_CM_RUNTIME AND NOT WIN32)
        set(SUPPORTED 0)
    endif()
    if (PARSED_ARGS_TEST_IS_CM AND NOT CMC_EXECUTABLE)
        set(SUPPORTED 0)
    endif()
    if (PARSED_ARGS_TEST_IS_DPCPP AND WIN32)
        set(SUPPORTED 0)
    endif()
    if (PARSED_ARGS_TEST_IS_DPCPP AND NOT ISPC_INCLUDE_DPCPP_EXAMPLES)
        set(SUPPORTED 0)
    endif()
    if (PARSED_ARGS_TEST_IS_CM)
        set(test_name "${PARSED_ARGS_NAME}_cm")
    else()
        set(test_name "${PARSED_ARGS_NAME}")
    endif()

    list(APPEND ${test_name} ${PARSED_ARGS_UNPARSED_ARGUMENTS} ${PARSED_ARGS_TEST_IS_CM})
    list(JOIN ${test_name} "_" result_test_name)
    if (PARSED_ARGS_REF_IMAGE)
        set (REF_IMAGE_OPT "-ref_image" ${PARSED_ARGS_REF_IMAGE})
    endif()
    if (PARSED_ARGS_RES_IMAGE)
        set (RES_IMAGE_OPT "-res_image" ${PARSED_ARGS_RES_IMAGE})
    endif()
    if (PARSED_ARGS_IMAGE_CMP_TH)
        set (IMAGE_CMP_TH_OPT "-image_cmp_th" ${PARSED_ARGS_IMAGE_CMP_TH})
    endif()
    if (PARSED_ARGS_DPCPP_SPV)
        set (DPCPP_SPV "-dpcpp_spv" ${PARSED_ARGS_DPCPP_SPV})
    endif()


    if (WIN32)
        set (TEST_REL_PATH ${PARSED_ARGS_NAME}/${CMAKE_BUILD_TYPE})
    else()
        set (TEST_REL_PATH ${PARSED_ARGS_NAME})
    endif()
    if (SUPPORTED EQUAL 1)
        add_test(NAME ${result_test_name}
            COMMAND ${Python3_EXECUTABLE} ${TEST_RUNNER} ${REF_IMAGE_OPT} ${RES_IMAGE_OPT} ${IMAGE_CMP_TH_OPT} ${DPCPP_SPV}
            ${TEST_REL_PATH} ${PARSED_ARGS_UNPARSED_ARGUMENTS}
        )
    endif()
endfunction()
