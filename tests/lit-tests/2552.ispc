// This test checks that this code produces neither any unreachable code nor
// causes any crash one functions requesting the addresses of variables that 
// are `const` and also set to a compile-time constant (even when the later
// address is dereferenced).

// RUN: %{ispc} %s --emit-llvm-text --nowrap --nostdlib --target=host -g -o - 2>&1 | FileCheck %s

// CHECK-NOT: unreachable
// CHECK-NOT: Error:
// CHECK-NOT: FATAL ERROR:

uniform int boo() {
    const uniform int a = 2;
    uniform const int * uniform const p = &a;
    uniform int x = a + *p;
    return x;
}
