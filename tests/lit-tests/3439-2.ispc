// This test checks that the compiler reports no errors when setting a 
// dereferenced pointer comming from a function call (it must be an lvalue).

// RUN: %{ispc} --target=host --emit-llvm-text --nowrap -g %s -o - 2>&1 | FileCheck %s

// CHECK-NOT: Error:
// CHECK-NOT: FATAL ERROR:

uniform int val = 0;
uniform int val2 = 1;

static uniform int *uniform foo() {
    return &val;
}

static void bar() {
    *foo() = val2;
}

extern "C" uniform int main() {
    bar();
    return val;
}
