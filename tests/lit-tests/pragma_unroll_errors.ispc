// Test to check '#pragma nounroll' functionality for different loop statemnts.

// RUN: not %{ispc} %s --target=avx2-i32x8 --nowrap 2>&1 | FileCheck %s
// REQUIRES: X86_ENABLED

// CHECK: Warning: Undefined #pragma.
// CHECK: Error: '#pragma unroll()' invalid value '0'; must be positive.
// CHECK: Error: Illegal '#Pragma' - expected a loop to follow '#pragma unroll/nounroll'.

#pragma whaaat!!
extern void goo_for(uniform int);
void foo_for() {
#pragma unroll(0)
    for (uniform int iter1 = 0; iter1 < 45; iter1++) {
#pragma nounroll
        goo_for(iter1);
    }   
}
