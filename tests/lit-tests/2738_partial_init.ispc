// RUN: not %{ispc} -DTYPE_BOOL --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s
// RUN: not %{ispc} -DTYPE_INT8 --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s
// RUN: not %{ispc} -DTYPE_UINT8 --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s
// RUN: not %{ispc} -DTYPE_INT16 --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s
// RUN: not %{ispc} -DTYPE_UINT16 --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s
// RUN: not %{ispc} -DTYPE_INT --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s
// RUN: not %{ispc} -DTYPE_UINT --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s
// RUN: not %{ispc} -DTYPE_INT64 --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s
// RUN: not %{ispc} -DTYPE_UINT64 --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s
// RUN: not %{ispc} -DTYPE_FLOAT16 --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s
// RUN: not %{ispc} -DTYPE_FLOAT --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s
// RUN: not %{ispc} -DTYPE_DOUBLE --target=avx2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK: Error: Initializer list for {{.*}} must have 4 elements

#define FP_INITIALIZER 0.0, 1.0
#define INT_INITIALIZER 0, 1
#define BOOL_INITIALIZER true, false

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
