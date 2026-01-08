// This test checks that masked-off lanes do not loose initialization values.
//
// RUN: %{ispc} --target=sse2-i32x8 --emit-llvm-text --nowrap %s -o - | FileCheck %s
// REQUIRES: X86_ENABLED

struct Foo {
    varying float y;
};

// Verify that masked-off lanes (1,2,4,5,7) correctly get -150 (bit pattern -1021968384)
// CHECK-LABEL: @f_fu
// CHECK: shufflevector {{.*}}, <8 x i32> <i32 {{.*}}, i32 -1021968384, i32 -1021968384, i32 {{.*}}, i32 -1021968384, i32 -1021968384, i32 {{.*}}, i32 -1021968384>
void f_fu(uniform float RET[], uniform float aFOO[]) {
    Foo struct_1;
    struct_1.y = aFOO[programIndex];

    Foo struct_2;
    struct_2.y = -150;

    if (programIndex % 3 == 0) {
        struct_2.y = struct_1.y;
    }

    RET[programIndex] = struct_2.y;
}
