// Test to check '#pragma unroll/nounroll' functionality for different loop statemnts.

// RUN: not %{ispc} %s --target=host --nostdlib --nowrap 2>&1 | FileCheck %s

// CHECK: Warning: unknown pragma ignored.
// CHECK: Error: '#pragma unroll()' invalid value '0'; must be positive.
// CHECK: Error: Illegal pragma - expected a loop to follow '#pragma unroll/nounroll'.

#pragma whaaat!!
extern void goo_for(uniform int);
void foo_for() {
#pragma unroll(0)
    for (uniform int iter1 = 0; iter1 < 45; iter1++) {
#pragma nounroll
        goo_for(iter1);
    }   
}
