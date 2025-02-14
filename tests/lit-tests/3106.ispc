// RUN: %{ispc} -O2 --target=avx512icl-x4 --emit-asm --x86-asm-syntax=intel %s -o - 2>&1 | FileCheck %s

// REQUIRE: X86_ENABLED

// XFAIL: *

// CHECK-LABEL: max___uni_3C_3_3E_uni_3C_3_3E_:
// CHECK-NEXT:# %bb.0:
// CHECK-NEXT: vpmaxsd xmm0, xmm0, xmm1
// CHECK-NEXT: ret
uniform int<3> max(uniform int<3> a, uniform int<3> b) {
    uniform int<3> r;
    foreach (i = 0 ... 3) {
        r[i] = max(a[i], b[i]);
    }
    return r;
}
