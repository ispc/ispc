// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --emit-asm -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// The goal of this test is to check that code generation for both versions is the same.
// We check only the key instructions to ensure that shuffle() library call is expanded
// as if the argument is a constant.

// Effectively it checks that LoopFullUnrollPass happens before ReplaceStdlibShiftPass.

// CHECK-LABEL: gang_scan___vyi
// CHECK: vpblendd
// CHECK: vpaddd
// CHECK: vpblendd
// CHECK: vpaddd
// CHECK: vperm2i128
// CHECK: vpaddd
// CHECK: retq

int32 gang_scan(int32 v) {
    #pragma unroll
    for (uniform int s = 1; s < programCount; s *= 2) {
        v += shift(v, -s);
    }
    return v;
}

// CHECK-LABEL: gang_scan_manual_unroll___vyi
// CHECK: vpblendd
// CHECK: vpaddd
// CHECK: vpblendd
// CHECK: vpaddd
// CHECK: vperm2i128
// CHECK: vpaddd
// CHECK: retq
int32 gang_scan_manual_unroll(int32 v) {
    v += shift(v, -1);
    v += shift(v, -2);
    v += shift(v, -4);
    return v;
}
