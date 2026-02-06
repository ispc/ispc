// This test ensures that ispc compiler generates optimal 64-bit computation for
// an unsigned loop induction variable. We also check that nsw flag is not incorrectly
// set on binop operating on the induction variable.

// REQUIRES: X86_ENABLED
// The fix for the test was reverted due to stability issues on neon target.
// XFAIL: *

// RUN: %{ispc} %s --emit-llvm-text --target=host --no-discard-value-names -o - -O1 --nowrap | FileCheck %s
// RUN: %{ispc} %s --target=avx2 -O1 --emit-asm --x86-asm-syntax=intel --nostdlib --nowrap -o - 2>&1 | FileCheck %s --check-prefix=ASM

// CHECK-LABEL: define void @fill(
// CHECK:      for_loop:
// CHECK-NEXT:   [[IV:%.*]] = phi i64 [ 0, [[PH:%.*]] ], [ [[IV_NEXT:%.*]], %for_loop ]
// CHECK-NEXT:   [[SHL:%.*]] = shl{{( nuw)?}}{{( nsw)?}} i64 [[IV]], 2
// CHECK-NEXT:   [[AND:%.*]] = and i64 [[SHL]], [[UINT_MAX:[0-9]+]]
// CHECK-NEXT:   [[PTR:%.*]] = getelementptr i8, ptr %buffer, i64 [[AND]]
// CHECK-NEXT:   store <{{[0-9]+}} x i32> {{.*}}, ptr [[PTR]], align 4
// CHECK-NEXT:   [[IV_NEXT:%.*]] = add nuw nsw i64 [[IV]], {{[0-9]+}}

// ASM-LABEL: fill:
// ASM:       LBB1_2:
// ASM-NEXT:         mov     [[R1:.*]], [[R2:.*]]
// ASM-NEXT:         and     [[R1]], -32
// ASM-NEXT:         vmovups ymmword ptr [[[R3:.*]] + [[R4:.*]]], ymm0
// ASM-NEXT:         add     [[R5:.*]], 8
// ASM-NEXT:         add     [[R6:.*]], 32

export void fill(uint32* uniform buffer, uniform int count) {
    for (uniform uint i = 0; i < count; i += programCount) {
        buffer[i + programIndex] = 7;
    }
}