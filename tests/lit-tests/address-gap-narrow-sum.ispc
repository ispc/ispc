// An int64 sum the program computes itself and then converts to a varying
// pointer. Its IR is the same as the frontend's nested indexing: a 64-bit add
// of two sign-extended 32-bit terms. With --addressing=32 ImproveMemoryOps
// narrows that sum without proving its range: for x = INT32_MAX and y = 1
// the byte offset is +2^31, which a scale-1 dword gather sign-extends to -2^31.
// Keep the full sum until arithmetic legality is established. The frontend
// integer-to-pointer boundary (#3983) also needs to preserve provenance.
// Ordinary nested indexing has a related arithmetic failure without such a
// cast (address-gap-nested-sum.ispc); the boundary alone cannot fix that case.
//
// RUN: %{ispc} %s --target=avx2-i32x8 --addressing=32 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s
// REQUIRES: X86_ENABLED
// XFAIL: *

// CHECK-LABEL: @wide_sum___
// CHECK-NOT: @llvm.x86.avx2.gather.d.ps.256(
// CHECK: @llvm.x86.avx2.gather.q.ps.256(
// CHECK-NOT: @llvm.x86.avx2.gather.d.ps.256(
void wide_sum(uniform float *uniform base, uniform int x[], uniform int y[], uniform float out[]) {
    int64 bytes = (int64)x[programIndex] + (int64)y[programIndex];
    uniform float *varying p = (uniform float *varying)((uniform int64)base + bytes);
    out[programIndex] = *p;
}
