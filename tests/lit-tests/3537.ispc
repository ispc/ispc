// This test checks that varying control flow generates efficient code on NEON targets.
// The regression in v1.26.0 produced inefficient codegen for varying conditionals.
// https://github.com/ispc/ispc/issues/3537

// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x8 --emit-asm -o - | FileCheck %s

// REQUIRES: ARM_ENABLED

// CHECK-LABEL: square_even___vyi
// Check that umaxv is used for efficient any() check instead of scalar extraction
// CHECK: umaxv
// Check that bit/bif are used for efficient blending
// CHECK: bi{{t|f}}

varying int square_even(varying int num) {
    if ((num & 1) == 0) {
        return num * num;
    } else {
        return num;
    }
}
