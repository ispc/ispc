// This test verifies that ISPC defines all the predefined macros documented at
// https://ispc.github.io/ispc.html#the-preprocessor
//
// This test uses empty input with -E -dM to list all predefined macros.
// We use CHECK-DAG to allow macros to appear in any order.

// RUN: %{ispc} -o - -E -dM - < /dev/null | FileCheck %s

// Basic ISPC macro
// CHECK-DAG: #define ISPC 1

// Target-specific macros (at least one should be defined for each compilation)
// CHECK-DAG: #define ISPC_TARGET_{{(NEON|SSE2|SSE4|AVX|AVX2|AVX512SKX|AVX512ICL|AVX512SPR|AVX10_2)}}

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
// CHECK-DAG: #define TARGET_WIDTH
// CHECK-DAG: #define TARGET_ELEMENT_WIDTH

// Feature detection macros
// CHECK-DAG: #define ISPC_UINT_IS_DEFINED 1
// CHECK-DAG: #define ISPC_ATTRIBUTE_SUPPORTED 1

// FP64 support (should be defined for most targets)
// CHECK-DAG: #define ISPC_FP64_SUPPORTED

// NOTE: ISPC_FP16_SUPPORTED is not defined as a macro, but ISPC_TARGET_HAS_HALF is used instead
// NOTE: ISPC_LLVM_INTRINSICS_ENABLED is only defined when --enable-llvm-intrinsics is used

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
// Note: The actual macros are F16_MIN/F16_MAX, not FLT16_MIN/FLT16_MAX
// CHECK-DAG: #define F16_MIN
// CHECK-DAG: #define FLT_MIN
// CHECK-DAG: #define DBL_MIN
// CHECK-DAG: #define F16_MAX
// CHECK-DAG: #define FLT_MAX
// CHECK-DAG: #define DBL_MAX

// Standard predefined macros (__FILE__, __LINE__, __DATE__, __TIME__)
// Note: These are not printed by -dM when reading from stdin
