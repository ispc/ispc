// Pins how the element size reaches the address of a hardware gather or
// scatter: as the scale immediate of the instruction, or as a shift of the
// index before it. The checks describe today's output during the structural
// rewrite. TODOs identify missed optimizations; the change implementing one
// updates its checks.
// Related issues: #1256, #1531.
//
// With --addressing=32 the shift is visible in the IR: `shl nsw` of the index
// and a gather with scale 1. With --addressing=64 the IR has the same shift on
// the 64-bit index, which the x86 backend folds into the scale, so those runs
// check the asm (for Linux, which fixes the register of the first argument
// and the symbol names).
//
// For the defined addresses in these signed-index cases, `shl nsw` permits
// moving the element-size shift into the hardware scale:
// sext(i << 2) == sext(i) * 4. This does not justify moving an unflagged
// wrapping shift across an extension.
//
// The sse4 run (FAC) covers targets without a gather instruction, where the
// scale goes to the factored pseudo call; it checks the record cases below.

// RUN: %{ispc} %s --target=avx2-i32x8 --addressing=32 --nostdlib --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefix=IR32
// RUN: %{ispc} %s --target=avx512skx-x16 --addressing=32 --nostdlib --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefixes=IR32,IR32SKX
// RUN: %{ispc} %s --target=avx2-i32x8 --addressing=64 --target-os=linux --nostdlib --nowrap --emit-asm -o - \
// RUN:   | FileCheck %s --check-prefix=ASM64
// RUN: %{ispc} %s --target=avx512skx-x16 --addressing=64 --target-os=linux --nostdlib --nowrap --emit-asm -o - \
// RUN:   | FileCheck %s --check-prefixes=ASM64,ASM64SKX
// RUN: %{ispc} %s --target=sse4-i32x4 --addressing=32 --nostdlib --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefix=FAC

// REQUIRES: X86_ENABLED && LINUX_ENABLED

// TODO(#1256): use scale 4 without a separate shift with --addressing=32;
// with --addressing=64 retain scale 4 and select dword indices
// (see gather-index-width.ispc).
// IR32-LABEL: @g_f___
// IR32: shl nsw <{{[0-9]+}} x i32> {{%.*}}, splat (i32 2)
// IR32: call {{.*}}@llvm.x86.{{avx2.gather.d.ps.256|avx512.mask.gather.dps.512}}({{.*}}, {{i8|i32}} 1)
// ASM64-LABEL: {{^}}g_f___
// ASM64-NOT: vgather{{.*}}(%rdi,%{{[yz]}}mm{{[0-9]+}})
// ASM64: vgatherqps {{.*}}(%rdi,%{{[yz]}}mm{{[0-9]+}},4)
// ASM64-NOT: vgather{{.*}}(%rdi,%{{[yz]}}mm{{[0-9]+}})
void g_f(uniform float *uniform f, uniform int *uniform i, uniform float *uniform r) { r[programIndex] = f[i[programIndex]]; }

// TODO(#1256): use scale 8 without a separate index shift with --addressing=32.
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

// TODO(#1256): use scale 8 without a separate index shift with --addressing=32.
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
// TODO(#1256): use scale 4 without a separate index shift with --addressing=32.
// IR32-LABEL: @s_f___
// IR32SKX: shl nsw <16 x i32> {{%.*}}, splat (i32 2)
// IR32SKX: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512({{.*}}, i32 1)
// ASM64-LABEL: {{^}}s_f___
// ASM64SKX-NOT: vscatter{{.*}}(%rdi,%zmm{{[0-9]+}})
// ASM64SKX: vscatterqps {{.*}}(%rdi,%zmm{{[0-9]+}},4)
// ASM64SKX-NOT: vscatter{{.*}}(%rdi,%zmm{{[0-9]+}})
void s_f(uniform float *uniform f, uniform int *uniform i, uniform float *uniform r) { f[i[programIndex]] = r[programIndex]; }

// The nested table lookup from #1531 (examples/cpu/noise). The table is not
// static, so its contents are unknown here. Both lookups use scale 1 and
// dword indices with --addressing=32 (see gather-index-width.ispc).
// TODO(#1531): select scale 4 for both lookups, preserving the original
// arithmetic and its no-wrap conditions.
// IR32-LABEL: @lookup___
// IR32: call {{.*}}@llvm.x86.{{avx2.gather.d.d.256|avx512.mask.gather.dpi.512}}({{.*}}, {{i8|i32}} 1)
// IR32: shl nsw <{{[0-9]+}} x i32> {{%.*}}, splat (i32 2)
// IR32: call {{.*}}@llvm.x86.{{avx2.gather.d.d.256|avx512.mask.gather.dpi.512}}({{.*}}, {{i8|i32}} 1)
// ASM64-LABEL: {{^}}lookup___
uniform int perm[512];
void lookup(uniform int out[], uniform int xs[], uniform int ys[]) {
    int x = xs[programIndex] & 255, y = ys[programIndex] & 255;
    out[programIndex] = perm[perm[x] + y];
}

// Records whose size is not a power of two. The offset is a `mul` by the
// record size, and ImproveMemoryOps pulls the largest of 2, 4 and 8 that
// divides it into the scale: i * 20 becomes (i * 5) with scale 4. For these
// signed indices a power-of-two size arrives as a `shl` (g_f above), and a
// constant field offset stops extraction (rec_z_load). The separate unsigned
// nuw path can also extract a shift. On sse4 (no gather instruction) the scale
// goes to the factored pseudo call, so the check there is the `mul` by 5 instead of 20.
struct Rec {
    float x, y[3], z;
};

// IR32-LABEL: @rec_x_store___
// IR32: mul {{(nsw )?}}<{{[0-9]+}} x i32> {{%.*}}, splat (i32 5)
// IR32SKX: call void @llvm.x86.avx512.mask.scatter.dps.512({{.*}}, i32 4)
// FAC-LABEL: @rec_x_store___
// FAC: mul {{(nsw )?}}<4 x i32> {{%.*}}, splat (i32 5)
void rec_x_store(uniform Rec recs[], int i, float v) { recs[i].x = v; }

// IR32-LABEL: @rec_x_load___
// IR32: mul {{(nsw )?}}<{{[0-9]+}} x i32> {{%.*}}, splat (i32 5)
// IR32: call {{.*}}@llvm.x86.{{avx2.gather.d.ps.256|avx512.mask.gather.dps.512}}({{.*}}, {{i8|i32}} 4)
// FAC-LABEL: @rec_x_load___
// FAC: mul {{(nsw )?}}<4 x i32> {{%.*}}, splat (i32 5)
void rec_x_load(uniform Rec recs[], int i, uniform float r[]) { r[programIndex] = recs[i].x; }

// i * 20 + 16: the add of the field offset stops the scale extraction. The
// gather is also qword: when ImproveMemoryOps runs, the 16 is still a
// broadcast shuffle rather than a constant, so the offset is not accepted as
// 32-bit.
// TODO(#1256): factor i * 20 + 16 into a dword index i * 5, scale 4, and
// displacement 16, proving equivalence across the arithmetic widths.
// IR32-LABEL: @rec_z_load___
// IR32: mul {{(nsw )?}}<{{[0-9]+}} x i32> {{%.*}}, splat (i32 20)
// IR32: call {{.*}}@llvm.x86.{{avx2.gather.q.ps.256|avx512.mask.gather.qps.512}}({{.*}}, {{i8|i32}} 1)
// FAC-LABEL: @rec_z_load___
// FAC: mul {{(nsw )?}}<4 x i32> {{%.*}}, splat (i32 20)
void rec_z_load(uniform Rec recs[], int i, uniform float r[]) { r[programIndex] = recs[i].z; }

// A runtime stride: the index is i * stride, so only the element size can be
// a scale.
// TODO(#1256): move the element-size shift into scale 4, retaining i * stride
// and its arithmetic semantics in the index.
// IR32-LABEL: @strided___
// IR32: call {{.*}}@llvm.x86.{{avx2.gather.d.ps.256|avx512.mask.gather.dps.512}}({{.*}}, {{i8|i32}} 1)
void strided(uniform float p[], int i, uniform int stride, uniform float r[]) { r[programIndex] = p[i * stride]; }

// Original reproducer from #1256 (exported, so it has an internal copy too).
// The scale/index-width TODO at g_f also applies to both copies here.
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
