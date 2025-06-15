// This test checks that the sizeof for the reference to an array properly 
// returns 28 here (size of the array rather than the size of the reference).

// RUN: %{ispc} --target=host --emit-llvm-text --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: ret i32 28
// CHECK-NOT: Error:
// CHECK-NOT: FATAL ERROR:

uniform int foo(uniform int (&a)[7]) {
    return sizeof(a);
}

