// Xe platforms do not support write prefetch.
// Check that prefetch/prefetchw instructions are not generated.
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --emit-llvm-text --nowrap -o - | FileCheck -check-prefix=CHECK_NO_PREFETCH %s
// RUN: %{ispc} %s --target=xelp-x16 --arch=xe64 --emit-llvm-text --nowrap -o - | FileCheck -check-prefix=CHECK_NO_PREFETCH %s
// RUN: %{ispc} %s --target=xehpg-x16 --arch=xe64 --emit-llvm-text --nowrap -o - | FileCheck -check-prefix=CHECK_NO_PREFETCH %s

// REQUIRES: GENX12_PLUS_ENABLED

// CHECK_NO_PREFETCH-NOT: .prefetch
// CHECK_NO_PREFETCH-NOT: .prefetchw

task void test(uniform float input[], uniform float out[]) {
    prefetchw_l1(input);
    prefetchw_l2(input);
    prefetchw_l3(input);

    uniform int64 a[programCount];
    int64 *ptr = &(a[programIndex]);

    prefetchw_l1(ptr);
    prefetchw_l2(ptr);
    prefetchw_l3(ptr);
}
