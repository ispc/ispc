// RUN: %{ispc} --target=host --nowrap --nostdlib --emit-llvm-text %s -o - | FileCheck %s

// CHECK-NOT: FATAL ERROR: Unhandled signal sent to process

varying float test_orig(uniform bool bCond, uniform float* uniform * uniform ptrA, uniform float* uniform* uniform ptrB)
{
    return (bCond ? *ptrA : *ptrB)[programIndex];
}

// CHECK-LABEL: @test(
// CHECK-NEXT: allocas:
// CHECK-NEXT:   [[SELECT:%.*]] = select i1 %bCond, {{.*}} %a, {{.*}} %b
// CHECK-NEXT:   [[PTR:%.*]] = getelementptr i{{(32|8)}}, {{.*}} [[SELECT]], i64 {{(4|16)}}
// CHECK-NEXT:   [[LOAD:%.*]] = load i32, {{.*}} [[PTR]]
// CHECK-NEXT:   ret i32 [[LOAD]]
// CHECK-NEXT: }
export uniform int test(uniform bool bCond, uniform int* uniform a, uniform int* uniform b)
{
    return (bCond ? a : b)[4];
}
