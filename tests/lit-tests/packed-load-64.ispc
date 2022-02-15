// This test checks 'packed_load_active' implementation for int64 on avx512skx-x8.
// A LLVM patch(https://reviews.llvm.org/D91294) is required for 'packed_load_active'
// and 'packed_store_active' on avx512skx-x8 for llvm 11.0. This test will fail
// without that patch. But the test is valid for later llvm versions too.
// In case of failure, run the following runtime test to confirm it's a
// legitimate ISPC bug(remove `--wrap-exe="sde -skx -- "` if sde is not
// required for avx512skx-x8 support) :
// python3 run_tests.py --target=avx512skx-x8 --wrap-exe="sde -skx -- " ./tests/packed-load-64-1.ispc

// RUN: %{ispc} %s --emit-asm --nowrap --target=avx512skx-x8 -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK: kshiftlb
// CHECK-NEXT: kshiftrb
// CHECk-NEXT: kmovb
// CHECk-NEXT: kshiftrb
void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform unsigned int64 a[programCount];
    #pragma ignore warning(perf)
    a[programIndex] = aFOO[programIndex];
    unsigned int64 aa;
    packed_load_active(a, &aa);
    #pragma ignore warning(perf)
    RET[programIndex] = aa;
}
