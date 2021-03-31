// Test to check 'alloca()' implementation.

// RUN: not %{ispc} %s --target=host --nowrap 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK: 9:12: Error: "alloca()" cannot have an argument of type "varying float".
void *uniform foo0(float size) {
    return alloca(size);
}

// CHECK: 14:12: Error: "alloca()" cannot have an argument of type "uniform int32 * uniform".
void *uniform foo1(uniform int *uniform size) {
    return alloca(size);
}
