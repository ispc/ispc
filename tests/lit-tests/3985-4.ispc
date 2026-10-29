// Regression test for https://github.com/ispc/ispc/issues/3985
// On a 32-bit target the byte offset of an unsigned index is full width and
// isn't extended, so it has no nuw: it may wrap, like the pointer it is added
// to. Signed indices keep nsw.

// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86 --nostdlib --nowrap -O0 --emit-llvm-text -o - | FileCheck %s

// macOS doesn't support 32-bit x86.
// REQUIRES: X86_ENABLED && !MACOS_HOST

// CHECK-LABEL: define void @gather_unsigned
// CHECK: %mul__i_load = mul <8 x i32> splat (i32 4), %i_load
void gather_unsigned(uniform float p[], uniform unsigned int32 idx[], uniform float r[]) {
    unsigned int32 i = idx[programIndex];
    r[programIndex] = p[i];
}

// CHECK-LABEL: define void @gather_signed
// CHECK: %mul__i_load = mul nsw <8 x i32> splat (i32 4), %i_load
void gather_signed(uniform float p[], uniform int32 idx[], uniform float r[]) {
    int32 i = idx[programIndex];
    r[programIndex] = p[i];
}
