// RUN: %{ispc} --pic --target=avx512skx-x8 %s --x86-asm-syntax=intel --emit-asm -o - | FileCheck %s

// REQUIRES: !MACOS_HOST && X86_ENABLED

// CHECK-LABEL: RoundToInt___vyf:
// CHECK-NEXT: # %bb.0:
// CHECK-NEXT: vroundps        ymm0, {{.*}}, 8
// CHECK-NEXT: ret
float RoundToInt(float x)
{
    return round(x);
}
