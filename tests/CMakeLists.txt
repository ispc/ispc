#
#  Copyright (c) 2018-2024, Intel Corporation
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
# Path to llvm lit tests
list(APPEND LIT_ARGS "${CMAKE_CURRENT_SOURCE_DIR}/lit-tests")
# Path to tests execution directory
list(APPEND LIT_ARGS "-Dispc_test_exec_root=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/tests")
# LLVM version used to build ISPC
list(APPEND LIT_ARGS "-Dispc_llvm_version_number=${LLVM_VERSION_NUMBER}")
# ISPC enabled targets
list(APPEND LIT_ARGS "-Dx86_enabled=$<IF:$<BOOL:${X86_ENABLED}>,ON,OFF>")
list(APPEND LIT_ARGS "-Darm_enabled=$<IF:$<BOOL:${ARM_ENABLED}>,ON,OFF>")
list(APPEND LIT_ARGS "-Dwasm_enabled=$<IF:$<BOOL:${WASM_ENABLED}>,ON,OFF>")
list(APPEND LIT_ARGS "-Dxe_enabled=$<IF:$<BOOL:${XE_ENABLED}>,ON,OFF>")
# ISPC enabled OS.
list(APPEND LIT_ARGS "-Dwindows_enabled=$<IF:$<BOOL:${ISPC_WINDOWS_TARGET}>,ON,OFF>")
list(APPEND LIT_ARGS "-Dlinux_enabled=$<IF:$<BOOL:${ISPC_LINUX_TARGET}>,ON,OFF>")
list(APPEND LIT_ARGS "-Dps_enabled=$<IF:$<BOOL:${ISPC_PS_TARGET}>,ON,OFF>")
list(APPEND LIT_ARGS "-Dmacos_arm_enabled=$<IF:$<BOOL:${ISPC_MACOS_ARM_TARGET}>,ON,OFF>")
# ISPC opaque pointers
list(APPEND LIT_ARGS "-Dopaque_mode=$<IF:$<BOOL:${ISPC_OPAQUE_PTR_MODE}>,ON,OFF>")
add_custom_target(check-all DEPENDS ispc
    COMMAND ${LIT_COMMAND} ${LIT_ARGS} "--verbose"
    COMMENT "Running lit tests"
    USES_TERMINAL
    )
set_target_properties(check-all PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)
set_target_properties(check-all PROPERTIES FOLDER "Tests")
