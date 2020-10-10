// Test to check '#pragma unroll/nounroll' functionality for different loop statemnts.

// RUN: not %{ispc} %s --target=host --nostdlib --nowrap 2>&1 | FileCheck %s

// CHECK: Warning: extra tokens at end of '#pragma nounroll'.
// CHECK: Error: Multiple '#pragma unroll/nounroll' directives used.

extern void goo_for(uniform int);
void foo_for() {
#pragma nounroll 9
#pragma unroll
    for (uniform int iter1 = 0; iter1 < 45; iter1++) {
        goo_for(iter1);
    }   
}
