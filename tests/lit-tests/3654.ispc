// Test that unary negation of enum values doesn't trigger an assertion failure.
// This is a regression test for GitHub issue #3654.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s

enum codes {
    E_FAILED0 = 1,
    E_FAILED1 = 2,
    E_FAILED2 = 3,
};

// CHECK-LABEL: @func___
// CHECK: ret <{{[0-9]+}} x i32> {{splat \(i32 -1\)|<.*i32 -1.*>}}
int func() {
    return -E_FAILED0;
}

// CHECK-LABEL: @func2___
// CHECK: ret <{{[0-9]+}} x i32> {{splat \(i32 -2\)|<.*i32 -2.*>}}
int func2() {
    return -E_FAILED1;
}

// CHECK-LABEL: @func3___
// CHECK: ret <{{[0-9]+}} x i32> {{splat \(i32 9\)|<.*i32 9.*>}}
int func3() {
    return -E_FAILED0 + 10;
}
