// Test that foreach ranges work with macros adjacent to ... operator (no space).
// This is a regression test for issue #3809 where "0...MAXI" failed to parse
// when MAXI is a #define macro, but "0 ... MAXI" worked.
//
// RUN: %{ispc} %s --target=host --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s

#define MAXI 5
#define HEX_MAX 0xA

// CHECK-LABEL: @test___
export void test(uniform int output[]) {
    // CHECK: call void @__masked_store
    foreach (i = 0...MAXI) {
        output[i] = i;
    }
}

// CHECK-LABEL: @test_with_spaces___
export void test_with_spaces(uniform int output[]) {
    // CHECK: call void @__masked_store
    foreach (i = 0 ... MAXI) {
        output[i] = i;
    }
}

// CHECK-LABEL: @test_hex___
export void test_hex(uniform int output[]) {
    // CHECK: call void @__masked_store
    foreach (i = 0x0...HEX_MAX) {
        output[i] = i;
    }
}

// CHECK-LABEL: @test_preserve_comments___
export void test_preserve_comments(uniform int output[]) {
    // This comment has 0...MAXI in it and should not be rewritten.
    /* Block comment: 0...MAXI */

    // CHECK: call void @__masked_store
    foreach (i = 0...MAXI) {
        output[i] = i;
    }
}

// CHECK-LABEL: @test_line_continuation___
export void test_line_continuation(uniform int output[]) {
    // This is a continued comment \
    // 0...MAXI should stay in the comment.

    // CHECK: call void @__masked_store
    foreach (i = 0...MAXI) {
        output[i] = i;
    }
}
