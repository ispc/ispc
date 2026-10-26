// Pins how the element size reaches the address of a hardware gather or
// scatter: as the scale immediate of the instruction, or as a shift of the
// index before it. The checks describe today's output; each case also gives
// the ideal output in an "Ideal:" comment, and the change that reaches it
// updates the checks of that case.
// Related issues: #1256, #1531.
//
// With --addressing=32 the shift is visible in the IR: `shl nsw` of the index
// and a gather with scale 1. With --addressing=64 the IR has the same shift on
// the 64-bit index, which the x86 backend folds into the scale, so those runs
// check the asm (for Linux, which fixes the register of the first argument
// and the symbol names).
//
// The ideal is legal in both modes: the shift is `shl nsw`, so
// sext(i << 2) == sext(i) * 4.

// RUN: %{ispc} %s --target=avx2-i32x8 --addressing=32 --nostdlib --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefix=IR32
// RUN: %{ispc} %s --target=avx512skx-x16 --addressing=32 --nostdlib --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefixes=IR32,IR32SKX
// RUN: %{ispc} %s --target=avx2-i32x8 --addressing=64 --target-os=linux --nostdlib --nowrap --emit-asm -o - \
// RUN:   | FileCheck %s --check-prefix=ASM64
// RUN: %{ispc} %s --target=avx512skx-x16 --addressing=64 --target-os=linux --nostdlib --nowrap --emit-asm -o - \
// RUN:   | FileCheck %s --check-prefixes=ASM64,ASM64SKX

// REQUIRES: X86_ENABLED && LINUX_ENABLED

// Ideal: scale 4, no shift (vgatherdps (%rdi,%ymm,4)) with --addressing=32;
// with --addressing=64 = today, but dword (see gather-index-width.ispc).
// IR32-LABEL: @g_f___
// IR32: shl nsw <{{[0-9]+}} x i32> {{%.*}}, splat (i32 2)
// IR32: call {{.*}}@llvm.x86.{{avx2.gather.d.ps.256|avx512.mask.gather.dps.512}}({{.*}}, {{i8|i32}} 1)
// ASM64-LABEL: {{^}}g_f___
// ASM64-NOT: vgather{{.*}}(%rdi,%{{[yz]}}mm{{[0-9]+}})
// ASM64: vgatherqps {{.*}}(%rdi,%{{[yz]}}mm{{[0-9]+}},4)
// ASM64-NOT: vgather{{.*}}(%rdi,%{{[yz]}}mm{{[0-9]+}})
void g_f(uniform float *uniform f, uniform int *uniform i, uniform float *uniform r) { r[programIndex] = f[i[programIndex]]; }

// Ideal: scale 8, no shift.
// IR32-LABEL: @g_d___
// IR32: shl nsw <{{[0-9]+}} x i32> {{%.*}}, splat (i32 3)
// IR32: call {{.*}}@llvm.x86.{{avx2.gather.d.pd.256|avx512.mask.gather.dpd.512}}({{.*}}, {{i8|i32}} 1)
// ASM64-LABEL: {{^}}g_d___
// ASM64-NOT: vgather{{.*}}(%rdi,%{{[yz]}}mm{{[0-9]+}})
// ASM64: vgatherqpd {{.*}}(%rdi,%{{[yz]}}mm{{[0-9]+}},8)
// ASM64-NOT: vgather{{.*}}(%rdi,%{{[yz]}}mm{{[0-9]+}})
void g_d(uniform double *uniform f, uniform int *uniform i, uniform double *uniform r) {
    r[programIndex] = f[i[programIndex]];
}

// Ideal: scale 8, no shift.
// IR32-LABEL: @g_i64___
// IR32: shl nsw <{{[0-9]+}} x i32> {{%.*}}, splat (i32 3)
// IR32: call {{.*}}@llvm.x86.{{avx2.gather.d.q.256|avx512.mask.gather.dpq.512}}({{.*}}, {{i8|i32}} 1)
// ASM64-LABEL: {{^}}g_i64___
// ASM64-NOT: vpgather{{.*}}(%rdi,%{{[yz]}}mm{{[0-9]+}})
// ASM64: vpgatherqq {{.*}}(%rdi,%{{[yz]}}mm{{[0-9]+}},8)
// ASM64-NOT: vpgather{{.*}}(%rdi,%{{[yz]}}mm{{[0-9]+}})
void g_i64(uniform int64 *uniform f, uniform int *uniform i, uniform int64 *uniform r) {
    r[programIndex] = f[i[programIndex]];
}

// AVX2 has no scatter instruction, so the scatter is checked on AVX-512 only.
// Ideal: scale 4, no shift.
// IR32-LABEL: @s_f___
// IR32SKX: shl nsw <16 x i32> {{%.*}}, splat (i32 2)
// IR32SKX: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512({{.*}}, i32 1)
// ASM64-LABEL: {{^}}s_f___
// ASM64SKX-NOT: vscatter{{.*}}(%rdi,%zmm{{[0-9]+}})
// ASM64SKX: vscatterqps {{.*}}(%rdi,%zmm{{[0-9]+}},4)
// ASM64SKX-NOT: vscatter{{.*}}(%rdi,%zmm{{[0-9]+}})
void s_f(uniform float *uniform f, uniform int *uniform i, uniform float *uniform r) { f[i[programIndex]] = r[programIndex]; }

// The nested table lookup from #1531 (examples/cpu/noise). The table is not
// static, so its contents are unknown here. Both lookups use scale 1; the
// inner one is also qword (see gather-index-width.ispc).
// Ideal: dword with scale 4 for both lookups.
// IR32-LABEL: @lookup___
// IR32: call {{.*}}@llvm.x86.{{avx2.gather.q.d.256|avx512.mask.gather.qpi.512}}({{.*}}, {{i8|i32}} 1)
// IR32: shl nsw <{{[0-9]+}} x i32> {{%.*}}, splat (i32 2)
// IR32: call {{.*}}@llvm.x86.{{avx2.gather.d.d.256|avx512.mask.gather.dpi.512}}({{.*}}, {{i8|i32}} 1)
// ASM64-LABEL: {{^}}lookup___
uniform int perm[512];
void lookup(uniform int out[], uniform int xs[], uniform int ys[]) {
    int x = xs[programIndex] & 255, y = ys[programIndex] & 255;
    out[programIndex] = perm[perm[x] + y];
}

// Original reproducer from #1256 (exported, so it has an internal copy too).
// Ideal: as g_f.
// IR32-LABEL: @gather___
// ASM64-LABEL: {{^}}gather___
// IR32-LABEL: @gather(
// IR32: shl nsw <{{[0-9]+}} x i32> {{%.*}}, splat (i32 2)
// IR32: call {{.*}}@llvm.x86.{{avx2.gather.d.ps.256|avx512.mask.gather.dps.512}}({{.*}}, {{i8|i32}} 1)
// ASM64-LABEL: {{^}}gather:
// ASM64-NOT: vgather{{.*}}(%rdi,%{{[yz]}}mm{{[0-9]+}})
// ASM64: vgatherqps {{.*}}(%rdi,%{{[yz]}}mm{{[0-9]+}},4)
// ASM64-NOT: vgather{{.*}}(%rdi,%{{[yz]}}mm{{[0-9]+}})
export void gather(float *uniform f, varying int *uniform i, varying float *uniform r) {
  *r = f[*i];
}
