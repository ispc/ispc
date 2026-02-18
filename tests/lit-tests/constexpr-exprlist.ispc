// REQUIRES: X86_ENABLED && X86_64_HOST
// RUN: %{ispc} %s --target=generic-i32x4 --nostdlib --emit-llvm-text -o - | FileCheck %s

constexpr varying int Lanes = {0, 1, 2, 3};
constexpr varying int Lanes2 = Lanes + 5;

varying int foo() { return Lanes2; }

constexpr varying int make() {
    varying int v = {10, 11, 12, 13};
    return v + 2;
}

constexpr varying int Made = make();

varying int bar() { return Made; }

// CHECK: ret <4 x i32> <i32 5, i32 6, i32 7, i32 8>
// CHECK: ret <4 x i32> <i32 12, i32 13, i32 14, i32 15>
