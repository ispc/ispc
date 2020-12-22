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
list(APPEND LIT_ARGS "-Dgenx_enabled=$<IF:$<BOOL:${GENX_ENABLED}>,ON,OFF>")
# ISPC enabled OS.
list(APPEND LIT_ARGS "-Dwindows_enabled=$<IF:$<BOOL:${ISPC_WINDOWS_TARGET}>,ON,OFF>")
list(APPEND LIT_ARGS "-Dmacos_arm_enabled=$<IF:$<BOOL:${ISPC_MACOS_ARM_TARGET}>,ON,OFF>")

add_custom_target(check-all DEPENDS ispc
    COMMAND ${LIT_COMMAND} ${LIT_ARGS}
    COMMENT "Running lit tests"
    USES_TERMINAL
    )
set_target_properties(check-all PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)
set_target_properties(check-all PROPERTIES FOLDER "Tests")
