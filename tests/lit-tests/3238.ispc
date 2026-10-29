// This test ensures that ispc compiler generates optimal 64-bit computation for
// an unsigned loop induction variable: the offset of the unsigned index is
// computed with nuw and stays zero extended, so the loop has one 64-bit
// induction variable and no sign extension (cdqe/movsxd) of a 32-bit one.

// REQUIRES: X86_ENABLED

// RUN: %{ispc} %s --emit-llvm-text --target=sse4-i32x4 --arch=x86-64 -o - -O1 --nowrap | FileCheck %s
// RUN: %{ispc} %s --emit-llvm-text --target=avx2-i32x8 --arch=x86-64 -o - -O1 --nowrap | FileCheck %s
// RUN: %{ispc} %s --emit-llvm-text --target=avx512skx-x16 --arch=x86-64 -o - -O1 --nowrap | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86-64 -O1 --emit-asm --x86-asm-syntax=intel --nostdlib --nowrap -o - 2>&1 | FileCheck %s --check-prefix=ASM

// CHECK-LABEL: define void @fill(
// CHECK:      for_loop:
// CHECK-NEXT:   [[IV:%.*]] = phi i64 [ 0, [[PH:%.*]] ], [ [[IV_NEXT:%.*]], %for_loop ]
// CHECK-NEXT:   [[SHL:%.*]] = shl{{( nuw)?}}{{( nsw)?}} i64 [[IV]], 2
// CHECK-NEXT:   [[PTR:%.*]] = getelementptr i8, ptr %buffer, i64 [[SHL]]
// CHECK-NEXT:   store <{{[0-9]+}} x i32> {{.*}}, ptr [[PTR]], align 4
// CHECK-NEXT:   [[IV_NEXT]] = add nuw nsw i64 [[IV]], {{[0-9]+}}

// ASM-LABEL: fill:
// ASM-NOT:     {{cdqe|movsxd}}
// ASM:         vmovups ymmword ptr [{{[a-z0-9]+}} + 4*[[R:[a-z0-9]+]]], ymm0
// ASM-NEXT:    add [[R]], 8
// ASM-NEXT:    cmp [[R]], {{[a-z0-9]+}}
// ASM-NOT:     {{cdqe|movsxd}}
// ASM:         ret

export void fill(uint32* uniform buffer, uniform int count) {
    for (uniform uint i = 0; i < count; i += programCount) {
        buffer[i + programIndex] = 7;
    }
}
