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
// BOOL-NEXT: ret <4 x i32> zeroinitializer

// INT8-LABEL: @foo___
// INT8-NEXT: allocas:
// INT8-NEXT: ret <4 x i8> <i8 2, i8 2, i8 2, i8 2>

// INT16-LABEL: @foo___
// INT16-NEXT: allocas:
// INT16-NEXT: ret <4 x i16> <i16 2, i16 2, i16 2, i16 2>

// INT-LABEL: @foo___
// INT-NEXT: allocas:
// INT-NEXT: ret <4 x i32> <i32 2, i32 2, i32 2, i32 2>

// INT64-LABEL: @foo___
// INT64-NEXT: allocas:
// INT64-NEXT: ret <4 x i64> <i64 2, i64 2, i64 2, i64 2>

// FLOAT16-LABEL: @foo___
// FLOAT16-NEXT: allocas:
// FLOAT16-NEXT: ret <4 x half> <half 0xH4000, half 0xH4000, half 0xH4000, half 0xH4000>

// FLOAT-LABEL: @foo___
// FLOAT-NEXT: allocas:
// FLOAT-NEXT: ret <4 x float> <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00>

// DOUBLE-LABEL: @foo___
// DOUBLE-NEXT: allocas:
// DOUBLE-NEXT: ret <4 x double> <double 2.000000e+00, double 2.000000e+00, double 2.000000e+00, double 2.000000e+00>

// BOOL-NOT: Error: Initializer for global variable "y" must be a constant.
// INT8-NOT: Error: Initializer for global variable "y" must be a constant.
// INT16-NOT: Error: Initializer for global variable "y" must be a constant.
// INT-NOT: Error: Initializer for global variable "y" must be a constant.
// INT64-NOT: Error: Initializer for global variable "y" must be a constant.
// FLOAT16-NOT: Error: Initializer for global variable "y" must be a constant.
// FLOAT-NOT: Error: Initializer for global variable "y" must be a constant.
// DOUBLE-NOT: Error: Initializer for global variable "y" must be a constant.

#ifdef TYPE_BOOL
#define TYPE bool
#endif

#ifdef TYPE_INT8
#define TYPE int8
#endif

#ifdef TYPE_UINT8
#define TYPE uint8
#endif

#ifdef TYPE_INT16
#define TYPE int16
#endif

#ifdef TYPE_UINT16
#define TYPE uint16
#endif

#ifdef TYPE_INT
#define TYPE int
#endif

#ifdef TYPE_UINT
#define TYPE uint
#endif

#ifdef TYPE_INT64
#define TYPE int64
#endif

#ifdef TYPE_UINT64
#define TYPE uint64
#endif

#ifdef TYPE_FLOAT16
#define TYPE float16
#endif

#ifdef TYPE_FLOAT
#define TYPE float
#endif

#ifdef TYPE_DOUBLE
#define TYPE double
#endif

static const TYPE x = { 1 };
#ifdef TYPE_BOOL
static const TYPE y = !x;
#else
static const TYPE y = x + 1;
#endif

TYPE foo() {
    return y;
}
