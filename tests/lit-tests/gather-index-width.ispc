// Pins the index width (dword or qword) of hardware gathers and scatters for
// each index type, in both addressing modes. The checks describe today's
// output; each case also gives the ideal output in an "Ideal:" comment, and
// the change that reaches it updates the checks of that case.
// Related issues: #1581 (unsigned index -> qword gather), #1531 (`x & 255`).

// RUN: %{ispc} %s --target=avx2-i32x8 --addressing=32 --nostdlib --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefix=A32 -DGD=llvm.x86.avx2.gather.d.ps.256 -DGQ=llvm.x86.avx2.gather.q.ps.256
// RUN: %{ispc} %s --target=avx2-i32x8 --addressing=64 --nostdlib --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefix=A64 -DGD=llvm.x86.avx2.gather.d.ps.256 -DGQ=llvm.x86.avx2.gather.q.ps.256
// RUN: %{ispc} %s --target=avx512skx-x16 --addressing=32 --nostdlib --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefixes=A32,SKX32 -DGD=llvm.x86.avx512.mask.gather.dps.512 -DGQ=llvm.x86.avx512.mask.gather.qps.512
// RUN: %{ispc} %s --target=avx512skx-x16 --addressing=64 --nostdlib --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefixes=A64,SKX64 -DGD=llvm.x86.avx512.mask.gather.dps.512 -DGQ=llvm.x86.avx512.mask.gather.qps.512

// REQUIRES: X86_ENABLED

// AVX2 has no scatter instruction, so scatters are checked on AVX-512 only.
// The SKX prefixes match the scatter intrinsics by their full names:
// llvm.x86.avx512.mask.scatter.dps.512 and llvm.x86.avx512.mask.scatter.qps.512.

#define G(NAME, IT)                                                                                                    \
    void NAME(uniform float *uniform base, uniform IT *uniform idx, uniform float *uniform out) {                     \
        IT i = idx[programIndex];                                                                                      \
        out[programIndex] = base[i];                                                                                   \
    }
#define S(NAME, IT)                                                                                                    \
    void NAME(uniform float *uniform base, uniform IT *uniform idx, uniform float *uniform val) {                     \
        IT i = idx[programIndex];                                                                                      \
        base[i] = val[programIndex];                                                                                   \
    }

// Signed 8- and 16-bit indices.
// Ideal: dword in both modes. With --addressing=64 the hardware computes
// base + sext(i32 index) * scale in 64 bits, which is exact.
// A32-LABEL: @g_s8___
// A32-NOT: call {{.*}}@[[GQ]](
// A32: call {{.*}}@[[GD]](
// A32-NOT: call {{.*}}@[[GQ]](
// A64-LABEL: @g_s8___
// A64-NOT: call {{.*}}@[[GD]](
// A64: call {{.*}}@[[GQ]](
// A64-NOT: call {{.*}}@[[GD]](
G(g_s8, int8)

// Unsigned 8- and 16-bit indices are qword in both modes today: InstCombine
// turns the offset into `zext nneg`, which is read as "may be >= 2^31".
// Ideal: dword in both modes (the byte offset is below 2^18).
// A32-LABEL: @g_u8___
// A32-NOT: call {{.*}}@[[GD]](
// A32: call {{.*}}@[[GQ]](
// A32-NOT: call {{.*}}@[[GD]](
// A64-LABEL: @g_u8___
// A64-NOT: call {{.*}}@[[GD]](
// A64: call {{.*}}@[[GQ]](
// A64-NOT: call {{.*}}@[[GD]](
G(g_u8, uint8)

// Ideal: dword in both modes.
// A32-LABEL: @g_s16___
// A32-NOT: call {{.*}}@[[GQ]](
// A32: call {{.*}}@[[GD]](
// A32-NOT: call {{.*}}@[[GQ]](
// A64-LABEL: @g_s16___
// A64-NOT: call {{.*}}@[[GD]](
// A64: call {{.*}}@[[GQ]](
// A64-NOT: call {{.*}}@[[GD]](
G(g_s16, int16)

// Ideal: dword in both modes.
// A32-LABEL: @g_u16___
// A32-NOT: call {{.*}}@[[GD]](
// A32: call {{.*}}@[[GQ]](
// A32-NOT: call {{.*}}@[[GD]](
// A64-LABEL: @g_u16___
// A64-NOT: call {{.*}}@[[GD]](
// A64: call {{.*}}@[[GQ]](
// A64-NOT: call {{.*}}@[[GD]](
G(g_u16, uint16)

// Ideal: dword in both modes. With --addressing=64 that needs the scale
// immediate (see gather-scale.ispc), so that the index is not pre-shifted.
// A32-LABEL: @g_s32___
// A32-NOT: call {{.*}}@[[GQ]](
// A32: call {{.*}}@[[GD]](
// A32-NOT: call {{.*}}@[[GQ]](
// A64-LABEL: @g_s32___
// A64-NOT: call {{.*}}@[[GD]](
// A64: call {{.*}}@[[GQ]](
// A64-NOT: call {{.*}}@[[GD]](
G(g_s32, int32)

// Ideal: qword unless the index is proved to be small enough. A dword gather
// sign-extends its index, which gives a wrong address once the byte offset is
// 2^31 or more.
// A32-LABEL: @g_u32___
// A32-NOT: call {{.*}}@[[GD]](
// A32: call {{.*}}@[[GQ]](
// A32-NOT: call {{.*}}@[[GD]](
// A64-LABEL: @g_u32___
// A64-NOT: call {{.*}}@[[GD]](
// A64: call {{.*}}@[[GQ]](
// A64-NOT: call {{.*}}@[[GD]](
G(g_u32, uint32)

// With --addressing=32 the 64-bit index is truncated to 32 bits.
// Ideal: = today.
// A32-LABEL: @g_s64___
// A32-NOT: call {{.*}}@[[GQ]](
// A32: call {{.*}}@[[GD]](
// A32-NOT: call {{.*}}@[[GQ]](
// A64-LABEL: @g_s64___
// A64-NOT: call {{.*}}@[[GD]](
// A64: call {{.*}}@[[GQ]](
// A64-NOT: call {{.*}}@[[GD]](
G(g_s64, int64)

// Ideal: = today.
// A32-LABEL: @g_u64___
// A32-NOT: call {{.*}}@[[GD]](
// A32: call {{.*}}@[[GQ]](
// A32-NOT: call {{.*}}@[[GD]](
// A64-LABEL: @g_u64___
// A64-NOT: call {{.*}}@[[GD]](
// A64: call {{.*}}@[[GQ]](
// A64-NOT: call {{.*}}@[[GD]](
G(g_u64, uint64)

// The #1531 noise pattern: a masked int index is qword for the same reason as
// the unsigned 8- and 16-bit indices.
// Ideal: dword in both modes.
// A32-LABEL: @g_and255___
// A32-NOT: call {{.*}}@[[GD]](
// A32: call {{.*}}@[[GQ]](
// A32-NOT: call {{.*}}@[[GD]](
// A64-LABEL: @g_and255___
// A64-NOT: call {{.*}}@[[GD]](
// A64: call {{.*}}@[[GQ]](
// A64-NOT: call {{.*}}@[[GD]](
void g_and255(uniform float *uniform base, uniform int *uniform idx, uniform float *uniform out) {
    int i = idx[programIndex] & 255;
    out[programIndex] = base[i];
}

// Scatters follow the gathers: dword where the gather is dword, qword
// otherwise. Ideal: as for the gathers.
// A32-LABEL: @s_s8___
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX32: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// A64-LABEL: @s_s8___
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX64: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
S(s_s8, int8)

// A32-LABEL: @s_u8___
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX32: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// A64-LABEL: @s_u8___
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX64: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
S(s_u8, uint8)

// A32-LABEL: @s_s16___
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX32: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// A64-LABEL: @s_s16___
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX64: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
S(s_s16, int16)

// A32-LABEL: @s_u16___
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX32: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// A64-LABEL: @s_u16___
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX64: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
S(s_u16, uint16)

// A32-LABEL: @s_s32___
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX32: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// A64-LABEL: @s_s32___
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX64: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
S(s_s32, int32)

// A32-LABEL: @s_u32___
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX32: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// A64-LABEL: @s_u32___
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX64: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
S(s_u32, uint32)

// A32-LABEL: @s_s64___
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX32: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// A64-LABEL: @s_s64___
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX64: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
S(s_s64, int64)

// A32-LABEL: @s_u64___
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX32: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// A64-LABEL: @s_u64___
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX64: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
S(s_u64, uint64)

// The #1531 noise pattern as a scatter. Ideal: dword in both modes.
// A32-LABEL: @s_and255___
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX32: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX32-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// A64-LABEL: @s_and255___
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
// SKX64: call {{.*}}@llvm.x86.avx512.mask.scatter.qps.512(
// SKX64-NOT: call {{.*}}@llvm.x86.avx512.mask.scatter.dps.512(
void s_and255(uniform float *uniform base, uniform int *uniform idx, uniform float *uniform val) {
    int i = idx[programIndex] & 255;
    base[i] = val[programIndex];
}

// Original reproducer from #1581 (the 16-byte Vecf4 record): avx2-i32x8
// emits 8 vgatherqps for each unsigned index type.
// Ideal: 4 vgatherdps for the uint16 index; the uint32 index follows g_u32.
// A32-LABEL: @ShotgunGatherScatterTest32___
// A32-NOT: call {{.*}}@[[GD]](
// A32: call {{.*}}@[[GQ]](
// A32-NOT: call {{.*}}@[[GD]](
// A32-LABEL: @ShotgunGatherScatterTest16___
// A32-NOT: call {{.*}}@[[GD]](
// A32: call {{.*}}@[[GQ]](
// A32-NOT: call {{.*}}@[[GD]](
typedef float<4> Vecf4;

unmasked void ShotgunGatherScatterTest32(uniform Vecf4 *uniform points, uniform unsigned int32 *uniform indices,
                                         uniform unsigned int32 count) {
    for (uniform unsigned int32 i = 0; i < count; i += programCount) {
        varying unsigned int32 pointIndex0 = indices[i + programIndex];

        varying Vecf4 p0 = points[pointIndex0];

        points[pointIndex0] = p0;
    }
}

unmasked void ShotgunGatherScatterTest16(uniform Vecf4 *uniform points, uniform unsigned int16 *uniform indices,
                                         uniform unsigned int32 count) {
    for (uniform unsigned int32 i = 0; i < count; i += programCount) {
        varying unsigned int16 pointIndex0 = indices[i + programIndex];

        varying Vecf4 p0 = points[pointIndex0];

        points[pointIndex0] = p0;
    }
}
