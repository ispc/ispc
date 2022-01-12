// Check that prefetchw instruction is generated.
// RUN: %{ispc} %s --emit-asm --cpu=broadwell --target=avx2-i32x8 -o - | FileCheck %s -check-prefix=CHECK_PREFETCHW
// RUN: %{ispc} %s --emit-asm --target=avx512skx-i32x8 -o - | FileCheck %s -check-prefix=CHECK_PREFETCHW

// REQUIRES: X86_ENABLED

void test(uniform float input[], uniform float out[]) {
//CHECK_PREFETCHW-COUNT-3: prefetchw
    prefetchw_l1(input);
    prefetchw_l2(input);
    prefetchw_l3(input);
    
    uniform int64 a[programCount];
    int64 *ptr = &(a[programIndex]);
//CHECK_PREFETCHW-COUNT-24: prefetchw
    prefetchw_l1(ptr);
    prefetchw_l2(ptr);
    prefetchw_l3(ptr);
}
