// RUN: %{ispc} -DTYPE_BOOL --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=BOOL
// RUN: %{ispc} -DTYPE_INT8 --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=INT8
// RUN: %{ispc} -DTYPE_UINT8 --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=INT8
// RUN: %{ispc} -DTYPE_INT16 --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=INT16
// RUN: %{ispc} -DTYPE_UINT16 --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=INT16
// RUN: %{ispc} -DTYPE_INT --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=INT
// RUN: %{ispc} -DTYPE_UINT --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=INT
// RUN: %{ispc} -DTYPE_INT64 --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=INT64
// RUN: %{ispc} -DTYPE_UINT64 --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=INT64
// RUN: %{ispc} -DTYPE_FLOAT16 --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=FLOAT16
// RUN: %{ispc} -DTYPE_FLOAT --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=FLOAT
// RUN: %{ispc} -DTYPE_DOUBLE --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --check-prefix=DOUBLE

// REQUIRES: X86_ENABLED

// BOOL-LABEL: @foo___
// BOOL-NEXT: allocas:
// BOOL-NEXT: ret <4 x i32> <i32 0, i32 -1, i32 0, i32 -1>

// INT8-LABEL: @foo___
// INT8-NEXT: allocas:
// INT8-NEXT: ret <4 x i8> <i8 1, i8 2, i8 3, i8 4>

// INT16-LABEL: @foo___
// INT16-NEXT: allocas:
// INT16-NEXT: ret <4 x i16> <i16 1, i16 2, i16 3, i16 4>

// INT-LABEL: @foo___
// INT-NEXT: allocas:
// INT-NEXT: ret <4 x i32> <i32 1, i32 2, i32 3, i32 4>

// INT64-LABEL: @foo___
// INT64-NEXT: allocas:
// INT64-NEXT: ret <4 x i64> <i64 1, i64 2, i64 3, i64 4>

// FLOAT16-LABEL: @foo___
// FLOAT16-NEXT: allocas:
// FLOAT16-NEXT: ret <4 x half> <half 0xH3C00, half 0xH4000, half 0xH4200, half 0xH4400>

// FLOAT-LABEL: @foo___
// FLOAT-NEXT: allocas:
// FLOAT-NEXT: ret <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>

// DOUBLE-LABEL: @foo___
// DOUBLE-NEXT: allocas:
// DOUBLE-NEXT: ret <4 x double> <double 1.000000e+00, double 2.000000e+00, double 3.000000e+00, double 4.000000e+00>

// BOOL-NOT: Error: Initializer for global variable "y" must be a constant.
// INT8-NOT: Error: Initializer for global variable "y" must be a constant.
// INT16-NOT: Error: Initializer for global variable "y" must be a constant.
// INT-NOT: Error: Initializer for global variable "y" must be a constant.
// INT64-NOT: Error: Initializer for global variable "y" must be a constant.
// FLOAT16-NOT: Error: Initializer for global variable "y" must be a constant.
// FLOAT-NOT: Error: Initializer for global variable "y" must be a constant.
// DOUBLE-NOT: Error: Initializer for global variable "y" must be a constant.

#define FP_INITIALIZER 0.0, (float16)1, (float)2.0, (double)3.0
#define INT_INITIALIZER 0, (int16)1, (uint8)2, (int64)3ul
#define BOOL_INITIALIZER true, (bool)0.0, (bool)1ull, (bool)0

#ifdef TYPE_BOOL
#define TYPE bool
#define INITIALIZER BOOL_INITIALIZER
#endif

#ifdef TYPE_INT8
#define TYPE int8
#define INITIALIZER INT_INITIALIZER
#endif

#ifdef TYPE_UINT8
#define TYPE uint8
#define INITIALIZER INT_INITIALIZER
#endif

#ifdef TYPE_INT16
#define TYPE int16
#define INITIALIZER INT_INITIALIZER
#endif

#ifdef TYPE_UINT16
#define TYPE uint16
#define INITIALIZER INT_INITIALIZER
#endif

#ifdef TYPE_INT
#define TYPE int
#define INITIALIZER INT_INITIALIZER
#endif

#ifdef TYPE_UINT
#define TYPE uint
#define INITIALIZER INT_INITIALIZER
#endif

#ifdef TYPE_INT64
#define TYPE int64
#define INITIALIZER INT_INITIALIZER
#endif

#ifdef TYPE_UINT64
#define TYPE uint64
#define INITIALIZER INT_INITIALIZER
#endif

#ifdef TYPE_FLOAT16
#define TYPE float16
#define INITIALIZER FP_INITIALIZER
#endif

#ifdef TYPE_FLOAT
#define TYPE float
#define INITIALIZER FP_INITIALIZER
#endif

#ifdef TYPE_DOUBLE
#define TYPE double
#define INITIALIZER FP_INITIALIZER
#endif

static const TYPE x = { INITIALIZER };
#ifdef TYPE_BOOL
static const TYPE y = !x;
#else
static const TYPE y = x + 1;
#endif

TYPE foo() {
    return y;
}
