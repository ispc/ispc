// Defining a struct with the name of a non-struct type must be reported
// without a crash, with and without members and attributes.
// See https://github.com/ispc/ispc/issues/3963

// RUN: not %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -o %t.o 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-NOT: FATAL ERROR
// CHECK: 16:8: Error: Type "I" is not a struct type!
// CHECK: 17:8: Error: Type "I" is not a struct type!
// CHECK: 18:8: Error: Type "I" is not a struct type!
// CHECK: 19:8: Error: Type "I" is not a struct type!
// CHECK-NOT: FATAL ERROR
typedef int I;
struct I { int a; };
struct I { int a; } __attribute__((aligned(16)));
struct I {};
struct I {} __attribute__((aligned(16)));
