// This tests checks that EEI is used (instead of GEP) to get unique value in foreach_unique statement for Xe target.
// RUN: %{ispc} %s --target=gen9-x8 -O0  --emit-llvm-text --no-discard-value-names --nowrap -o - | FileCheck %s

// REQUIRES: XE_ENABLED

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int count = 0;
    float a = aFOO[programIndex];
    foreach_unique (ua in a)
// CHECK: foreach_find_next
// CHECK-NOT: %unique_value_vec = getelementptr
// CHECK: %unique_value = extractelement
        ++count;
    RET[programIndex] = count;
}