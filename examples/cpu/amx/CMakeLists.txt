#
#  Copyright (c) 2025, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

#
# ispc examples: amx
#

# Check if we're on Linux x86-64 architecture
if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND (ISPC_ARCH STREQUAL "x86-64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")))
    message(STATUS "AMX example requires Linux x86-64 architecture. Skipping AMX example.")
    return()
endif()

# Test for __fp16 support
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
int main() {
    __fp16 a = 1.0;
    __fp16 b = 2.0;
    __fp16 c = a + b;
    return 0;
}
" HAS_FP16_SUPPORT)

if(NOT HAS_FP16_SUPPORT)
    message(STATUS "AMX example requires __fp16 support, but host compiler doesn't support it. Skipping AMX example.")
    return()
endif()

# Check for AMX hardware support on Linux
set(AMX_SUPPORTED FALSE)
set(AMX_DETECTION_METHOD "none")

# First check for hardware AMX support via CPUID
include(CheckCSourceRuns)
set(CMAKE_REQUIRED_LIBRARIES "")

check_c_source_runs("
    #include <cpuid.h>
    #include <stdio.h>

    int check_amx_tile_support() {
        unsigned int eax, ebx, ecx, edx;
        // Check CPUID.(EAX=7,ECX=0):EDX[bit 24] for AMX-TILE
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            return (edx & (1 << 24)) != 0;
        }
        return 0;
    }

    int check_amx_fp16_support() {
        unsigned int eax, ebx, ecx, edx;
        // Check CPUID.(EAX=7,ECX=1):EAX[bit 21] for AMX-FP16
        if (__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) {
            return (eax & (1 << 21)) != 0;
        }
        return 0;
    }

    int main() {
        if (check_amx_tile_support() && check_amx_fp16_support()) {
            return 0; // Success
        }
        return 1; // Failure
    }
" HAVE_AMX_SUPPORT)

if(HAVE_AMX_SUPPORT)
    message(STATUS "Detected AMX hardware support via CPUID (AMX-TILE and AMX-FP16)")
    set(AMX_SUPPORTED TRUE)
    set(AMX_DETECTION_METHOD "hardware_cpuid")
else()
    message(STATUS "AMX hardware support not detected via CPUID")
    # Fallback to Intel SDE if hardware support is not available
    find_program(SDE_EXECUTABLE
        NAMES sde
        PATHS ENV SDE_PATH
        PATH_SUFFIXES bin
        DOC "Intel SDE (Software Development Emulator) for AMX emulation"
    )

    if(SDE_EXECUTABLE)
        message(STATUS "Found Intel SDE: ${SDE_EXECUTABLE} - will use for AMX emulation")
        set(AMX_SUPPORTED TRUE)
        set(AMX_DETECTION_METHOD "sde")
    else()
        set(AMX_SUPPORTED FALSE)
    endif()
endif()

if(NOT AMX_SUPPORTED)
    message(STATUS "Neither AMX hardware nor Intel SDE found. Skipping AMX example.")
    return()
endif()

message(STATUS "AMX support detected via: ${AMX_DETECTION_METHOD}")

if(AMX_SUPPORTED)
    set (ISPC_SRC_NAME "amx_matmul")
    set (TARGET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/amx_matmul.cpp)
    set (ISPC_FLAGS -O2 --target=avx512spr-x32 --cpu=gnr --enable-llvm-intrinsics --pic)
    set (ISPC_IA_TARGETS "avx512spr-x32")

    add_ispc_example(NAME "amx"
                  ISPC_IA_TARGETS ${ISPC_IA_TARGETS}
                  ISPC_SRC_NAME ${ISPC_SRC_NAME}
                  TARGET_SOURCES ${TARGET_SOURCES}
                  ISPC_FLAGS ${ISPC_FLAGS})

    # Add compile definitions for matrix dimensions
    target_compile_definitions(amx PRIVATE
        MAT_SIZE_M=128
        MAT_SIZE_N=128
        MAT_SIZE_K=128)

    # Add SDE path if available
    if(SDE_EXECUTABLE)
        get_filename_component(SDE_PATH "${SDE_EXECUTABLE}" DIRECTORY)
        get_filename_component(SDE_PATH "${SDE_PATH}" DIRECTORY)
        target_compile_definitions(amx PRIVATE SDE_PATH="${SDE_PATH}")
    endif()

    # Handle testing when SDE emulation is required
    if(AMX_DETECTION_METHOD STREQUAL "sde")
        # Disable the default test since it cannot run on non-AMX hardware
        set_tests_properties(amx PROPERTIES DISABLED TRUE)

        # Add a new test that uses SDE emulation
        add_test(NAME amx_sde
                 COMMAND ${SDE_EXECUTABLE} -gnr -- $<TARGET_FILE:amx>
                 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        set_tests_properties(amx_sde PROPERTIES
            TIMEOUT 300
            LABELS "cpu_example"
        )
        message(STATUS "Test 'amx' disabled, 'amx_sde' configured to use SDE emulation")
    endif()

    # Add run target that uses the best available execution method
    if(AMX_DETECTION_METHOD STREQUAL "sde")
        # Use SDE emulation if that's how we detected support
        add_custom_target(run-amx
            COMMAND ${SDE_EXECUTABLE} -gnr -- $<TARGET_FILE:amx>
            DEPENDS amx
            COMMENT "Running AMX example with Intel SDE emulation"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        message(STATUS "Added run target: 'run-amx' (will use SDE emulation)")
    else()
        # Use native execution if we detected hardware support
        add_custom_target(run-amx
            COMMAND $<TARGET_FILE:amx>
            DEPENDS amx
            COMMENT "Running AMX example on native hardware"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        message(STATUS "Added run target: 'run-amx' (will use native hardware)")
    endif()
endif()
