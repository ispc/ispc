// This test verifies that ISPC defines all the predefined macros documented at
// https://ispc.github.io/ispc.html#the-preprocessor
//
// This test uses empty input with -E -dM to list all predefined macros.
// We use CHECK-DAG to allow macros to appear in any order.

// RUN: %{ispc} --target=avx2 --enable-llvm-intrinsics -o - -E -dM - < /dev/null | FileCheck %s
// REQUIRES: X86_ENABLED

// Basic ISPC macro
// CHECK-DAG: #define ISPC 1

// Target-specific macros (at least one should be defined for each compilation)
// CHECK-DAG: #define ISPC_TARGET_AVX2 1

// Pointer size (should be either 32 or 64)
// CHECK-DAG: #define ISPC_POINTER_SIZE {{(32|64)}}

// Version macros
// CHECK-DAG: #define ISPC_MAJOR_VERSION
// CHECK-DAG: #define ISPC_MINOR_VERSION
// CHECK-DAG: #define LLVM_VERSION_MAJOR
// CHECK-DAG: #define LLVM_VERSION_MINOR

// Mathematics constant
// CHECK-DAG: #define PI 3.1415926535

// Target width and element width
// CHECK-DAG: #define TARGET_WIDTH 8
// CHECK-DAG: #define TARGET_ELEMENT_WIDTH 4

// Feature detection macros
// CHECK-DAG: #define ISPC_UINT_IS_DEFINED 1
// CHECK-DAG: #define ISPC_ATTRIBUTE_SUPPORTED 1

// FP64 support (should be defined for most targets)
// CHECK-DAG: #define ISPC_TARGET_HAS_FP64_SUPPORT 1
// CHECK-DAG: #define ISPC_FP64_SUPPORTED 1

// NOTE: ISPC_TARGET_HAS_FP16_SUPPORT and ISPC_FP16_SUPPORTED are only defined for targets with full FP16 support (e.g., avx512spr)
// NOTE: ISPC_LLVM_INTRINSICS_ENABLED is only defined when --enable-llvm-intrinsics is used
// CHECK-DAG: #define ISPC_LLVM_INTRINSICS_ENABLED 1

// Signed integer min/max macros
// CHECK-DAG: #define INT8_MIN
// CHECK-DAG: #define INT16_MIN
// CHECK-DAG: #define INT32_MIN
// CHECK-DAG: #define INT64_MIN
// CHECK-DAG: #define INT8_MAX
// CHECK-DAG: #define INT16_MAX
// CHECK-DAG: #define INT32_MAX
// CHECK-DAG: #define INT64_MAX

// Unsigned integer max macros
// CHECK-DAG: #define UINT8_MAX
// CHECK-DAG: #define UINT16_MAX
// CHECK-DAG: #define UINT32_MAX
// CHECK-DAG: #define UINT64_MAX

// Floating point min/max macros
// CHECK-DAG: #define F16_MIN
// CHECK-DAG: #define FLT_MIN
// CHECK-DAG: #define DBL_MIN
// CHECK-DAG: #define F16_MAX
// CHECK-DAG: #define FLT_MAX
// CHECK-DAG: #define DBL_MAX

// Standard predefined macros (__FILE__, __LINE__, __DATE__, __TIME__)
// Note: These are not printed by -dM when reading from stdin
