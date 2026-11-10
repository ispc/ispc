#
#  Copyright (c) 2018-2026, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# tests CMakeLists.txt
#
set(LIT_COMMAND "${Python3_EXECUTABLE};${CMAKE_SOURCE_DIR}/utils/lit/lit.py")
# User can specify additional lit arguments with -DLIT_ARGS
# Path to ispc
list(APPEND LIT_ARGS "--path=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}")
# Path to LLVM utils
list(APPEND LIT_ARGS "--path=${LLVM_TOOLS_BINARY_DIR}")
# Path to ispc root directory
list(APPEND LIT_ARGS "-Dispc_build_dir=${CMAKE_BINARY_DIR}")
# Path to tests execution directory
list(APPEND LIT_ARGS "-Dispc_test_exec_root=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/tests")
# LLVM version used to build ISPC
list(APPEND LIT_ARGS "-Dispc_llvm_version_number=${LLVM_VERSION_NUMBER}")
# ISPC enabled targets
list(APPEND LIT_ARGS "-Dx86_enabled=$<IF:$<BOOL:${X86_ENABLED}>,ON,OFF>")
list(APPEND LIT_ARGS "-Darm_enabled=$<IF:$<BOOL:${ARM_ENABLED}>,ON,OFF>")
list(APPEND LIT_ARGS "-Darm32_enabled=$<IF:$<BOOL:${ARM32_ENABLED}>,ON,OFF>")
list(APPEND LIT_ARGS "-Driscv_enabled=$<IF:$<BOOL:${RISCV_ENABLED}>,ON,OFF>")
list(APPEND LIT_ARGS "-Dppc64_enabled=$<IF:$<BOOL:${PPC64_ENABLED}>,ON,OFF>")
list(APPEND LIT_ARGS "-Dwasm_enabled=$<IF:$<BOOL:${WASM_ENABLED}>,ON,OFF>")
list(APPEND LIT_ARGS "-Dxe_enabled=$<IF:$<BOOL:${XE_ENABLED}>,ON,OFF>")
# ISPC enabled OS.
list(APPEND LIT_ARGS "-Dwindows_enabled=$<IF:$<BOOL:${ISPC_WINDOWS_TARGET}>,ON,OFF>")
list(APPEND LIT_ARGS "-Dlinux_enabled=$<IF:$<BOOL:${ISPC_LINUX_TARGET}>,ON,OFF>")
# On Android hosts (Termux) the Linux target uses bionic, see cmake/GenerateBuiltins.cmake.
list(APPEND LIT_ARGS "-Dandroid_host=$<IF:$<STREQUAL:${CMAKE_SYSTEM_NAME},Android>,ON,OFF>")
list(APPEND LIT_ARGS "-Dandroid_enabled=$<IF:$<BOOL:${ISPC_ANDROID_TARGET}>,ON,OFF>")
list(APPEND LIT_ARGS "-Dfreebsd_enabled=$<IF:$<BOOL:${ISPC_FREEBSD_TARGET}>,ON,OFF>")
list(APPEND LIT_ARGS "-Dps_enabled=$<IF:$<BOOL:${ISPC_PS_TARGET}>,ON,OFF>")
list(APPEND LIT_ARGS "-Dmacos_arm_enabled=$<IF:$<AND:$<BOOL:${ISPC_MACOS_TARGET}>,$<BOOL:${ARM_ENABLED}>>,ON,OFF>")
list(APPEND LIT_ARGS "-Dmacos_ios_enabled=$<IF:$<BOOL:${ISPC_IOS_TARGET}>,ON,OFF>")
# FileCheck executable
list(APPEND LIT_ARGS "-Dfile_check_executable=${FILE_CHECK_EXECUTABLE}")
# ISPC library is enabled
list(APPEND LIT_ARGS "-Dispc_lib_enabled=$<IF:$<BOOL:${ISPC_LIBRARY}>,ON,OFF>")
list(APPEND LIT_ARGS "-Dispc_lib_jit_enabled=$<IF:$<BOOL:${ISPC_LIBRARY_JIT}>,ON,OFF>")
# TODO! generic target bc?
set(CHECK_ALL_DEPS ispc ispc-opt ${ISPC_DEPS})
if(TARGET check_isa)
    list(APPEND CHECK_ALL_DEPS check_isa)
endif()
if(TARGET libispc_shared)
    list(APPEND CHECK_ALL_DEPS libispc_shared)
endif()
# Exercise the production C and m4 builtin commands with intentional errors.
if(X86_ENABLED)
    set(BUILTINS_ERROR_TEST_ARCH x86)
elseif(ARM_ENABLED)
    set(BUILTINS_ERROR_TEST_ARCH arm)
elseif(RISCV_ENABLED)
    set(BUILTINS_ERROR_TEST_ARCH riscv)
elseif(PPC64_ENABLED)
    set(BUILTINS_ERROR_TEST_ARCH ppc)
endif()
configure_file(cmake/builtins-compile-error.cmake.in builtins-compile-error.cmake @ONLY)
add_custom_target(check-builtins-compile-error
    COMMAND ${CMAKE_COMMAND} "-DTEST_CONFIG=$<CONFIG>"
        -P "${CMAKE_CURRENT_BINARY_DIR}/builtins-compile-error.cmake"
    COMMENT "Checking builtin compiler and m4 error propagation"
    VERBATIM)
set_target_properties(check-builtins-compile-error PROPERTIES
    EXCLUDE_FROM_DEFAULT_BUILD ON FOLDER "Tests")
add_custom_target(check-all DEPENDS ${CHECK_ALL_DEPS}
    COMMAND ${LIT_COMMAND} ${LIT_ARGS} "${CMAKE_CURRENT_SOURCE_DIR}/lit-tests" "--verbose"
    COMMENT "Running lit tests"
    USES_TERMINAL
    )
set_target_properties(check-all PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)
set_target_properties(check-all PROPERTIES FOLDER "Tests")
add_dependencies(check-all check-builtins-compile-error)

string(REPLACE ";" " " LIT_ARGS_STR "${LIT_ARGS}")
string(REPLACE ";" " " LIT_COMMAND_STR "${LIT_COMMAND}")

add_custom_target(check-one DEPENDS ${CHECK_ALL_DEPS}
    COMMAND bash -c "${LIT_COMMAND_STR} ${LIT_ARGS_STR} \$TEST --verbose"
    COMMENT "Running lit test from $TEST"
    USES_TERMINAL
    VERBATIM
)
set_target_properties(check-one PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)
set_target_properties(check-one PROPERTIES FOLDER "Tests")
