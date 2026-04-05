// This test checks that dead code with varying variables does not leave
// redundant control flow (branches, phi nodes) in the final IR.
// https://github.com/ispc/ispc/issues/2802

// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-LABEL: @test___
// CHECK-NOT: br i1
// CHECK-NOT: phi
// CHECK: add <[[WIDTH:.*]] x i8>
// CHECK-NEXT: ret <[[WIDTH]] x i8>

uint8 test(uniform uint8<TARGET_WIDTH> x)
{
    uint8 y = x[0];
    return x[programIndex]+1;
}

// CHECK-LABEL: @test_unmasked___
// CHECK-NOT: br i1
// CHECK-NOT: phi
// CHECK: add <[[WIDTH]] x i8>
// CHECK-NEXT: ret <[[WIDTH]] x i8>

uint8 test_unmasked(uniform uint8<TARGET_WIDTH> x)
{
    uint8 _x;
    unmasked
    {
        _x = x[programIndex]+1;
    }
    return x[programIndex]+1;
}
