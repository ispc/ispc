// This test ensures that ispc compiler generates optimal 64-bit computation for
// an unsigned loop induction variable. We also check that nsw flag is not incorrectly
// set on binop operating on the induction variable.

// REQUIRES: X86_ENABLED
// RUN: %{ispc} %s --emit-llvm-text --target=host --no-discard-value-names -o - -O1 --nowrap | FileCheck %s
// RUN: %{ispc} %s --target=avx2 -O1 --emit-asm --x86-asm-syntax=intel --nostdlib --nowrap -o - 2>&1 | FileCheck %s --check-prefix=ASM

// CHECK-LABEL: define void @fill(
// CHECK:      for_loop:
// CHECK-NEXT:   %indvars.iv = phi i64 [ 0, %for_loop.lr.ph ], [ %indvars.iv.next, %for_loop ]
// CHECK-NEXT:   %shl_i_load4_broadcast5_.elt0 = shl i64 %indvars.iv, 2
// CHECK-NEXT:   %offset_cast.elt0 = and i64 %shl_i_load4_broadcast5_.elt0, 4294967264
// CHECK-NEXT:   %ptr = getelementptr i8, ptr %buffer, i64 %offset_cast.elt0
// CHECK-NEXT:   store <{{[0-9]+}} x i32> splat (i32 7), ptr %ptr, align 4
// CHECK-NEXT:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, {{[0-9]+}}
// CHECK-NEXT:   %less_i_load_count_load = icmp samesign ult i64 %indvars.iv.next, %0
// CHECK-NEXT:   br i1 %less_i_load_count_load, label %for_loop, label %for_exit,

// ASM-LABEL: fill:
// ASM:       .LBB1_2:
// ASM-NEXT:          mov     esi, ecx
// ASM-NEXT:          and     esi, -32
// ASM-NEXT:          vmovups ymmword ptr [rdi + rsi], ymm0
// ASM-NEXT:          add     rdx, 8
// ASM-NEXT:          add     rcx, 32

export void fill(uint32* uniform buffer, uniform int count) {
    for (uniform uint i = 0; i < count; i += programCount) {
        buffer[i + programIndex] = 7;
    }
}
