// RUN: %{ispc} --target=host --nowrap %s -o t.o 2>&1 | FileCheck %s

// Check that the deprecated attribute is working correctly for functions returning pointers

// CHECK: Warning: Function "f(uniform int32 * uniform * varying)" is deprecated.

__attribute__((deprecated)) uniform int** f(uniform int** p) { }

void foo() {
    uniform int** p;
    f(p);
}
