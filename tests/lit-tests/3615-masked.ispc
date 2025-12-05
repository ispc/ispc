// Test that popcnt respects mask in masked control flow
// RUN: %{ispc} --emit-llvm-text -O0 --target=avx512icl-x8 %s -o - | FileCheck %s
// REQUIRES: X86_ENABLED

// CHECK-LABEL: @test_popcnt_int32_masked
// CHECK: call {{.*}}@llvm.ctpop.v8i32
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> zeroinitializer

export void test_popcnt_int32_masked(uniform int input[], uniform int result[]) {
    varying int value = input[programIndex];
    varying int count;

    // popcnt under masked control flow
    if (programIndex < programCount / 2) {
        count = popcnt(value);
    } else {
        count = 0;
    }
    result[programIndex] = count;
}

// CHECK-LABEL: @test_popcnt_int64_masked
// CHECK: call {{.*}}@llvm.ctpop.v8i64
// CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> zeroinitializer

export void test_popcnt_int64_masked(uniform int64 input[], uniform int result[]) {
    varying int64 value = input[programIndex];
    varying int count;

    // popcnt under masked control flow
    if (programIndex < programCount / 2) {
        count = popcnt(value);
    } else {
        count = 0;
    }
    result[programIndex] = count;
}
