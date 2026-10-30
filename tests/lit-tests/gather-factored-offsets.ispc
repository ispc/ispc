// Pins how ImproveMemoryOps splits gather offsets on targets without a gather
// instruction (sse4). There a gather becomes a "factored" pseudo call with a
// varying offset, a 2/4/8 scale and a constant offset. With --addressing=32
// its 32-bit variant is used when the varying part is 32-bit (the constant part
// is assumed to fit, see the FIXME in ImproveMemoryOps). The checks describe
// today's output; each case also gives the ideal output in an "Ideal:" comment,
// and the change that reaches it updates the checks of that case.

// RUN: %{ispc} %s --target=sse4-i32x4 --addressing=32 --nostdlib --nowrap --emit-llvm-text -o - 2> %t.warn \
// RUN:   | FileCheck %s --check-prefix=FAC
// RUN: FileCheck --input-file=%t.warn %s --check-prefix=WARN
// RUN: %{ispc} %s --target=sse4-i32x4 --addressing=32 --wrap-signed-int --nostdlib --nowrap --emit-llvm-text -o - \
// RUN:   | FileCheck %s --check-prefix=WRAP

// REQUIRES: X86_ENABLED

// An array of varying structs indexed by a varying index plus 1: the offset
// is idx * 16 plus a different constant per lane ({16, 20, 24, 28}). The
// constant part is split off, so the varying part stays 32-bit and each lane
// extends its own 32-bit offset. Without the split the whole offset is a
// 64-bit vector.
// Ideal: = today.
// FAC-LABEL: @vstruct___
// FAC-NOT: sext <4 x i32> {{%.*}} to <4 x i64>
// FAC: extractelement <4 x i32>
// FAC-NOT: sext <4 x i32> {{%.*}} to <4 x i64>
struct Foo {
    float f;
};
void vstruct(Foo foo[], int idx, uniform float r[]) { r[programIndex] = foo[idx + 1].f; }

// The index comes from a uniform table, so it becomes constant only after
// other optimizations. A later ImproveMemoryOps run moves it from the varying
// to the constant offset, which lets GatherCoalescePass replace the gather
// with scalar loads.
// Ideal: = today.
// FAC-LABEL: @table_index___
// WARN: gather-factored-offsets.ispc:[[@LINE+6]]:{{[0-9]+}}: Performance Warning: Coalesced gather into 4 loads (4 x 1-wide).
void table_index(uniform float aFOO[], uniform float r[]) {
    float a = aFOO[programIndex];
    float foo[5] = {0, a, 0, 2 * a, 3};
    uniform int offset[4] = {1, 2, 3, 4};
    int x = offset[programIndex & 0x3];
    r[programIndex] = foo[x];
}

// A varying global added to the index keeps the gather. ImproveMemoryOps
// splits (programIndex + zero) << 2: programIndex << 2 = {0, 4, 8, 12} becomes
// the constant offset, and the varying offset is a new zero << 2, which gets
// the nsw flag of the original shift unless --wrap-signed-int is given.
// Ideal: the varying offset zero with scale 4 instead of the shift (see
// gather-scale.ispc), with the same treatment of overflow.
// FAC-LABEL: @plus_zero___
// FAC: shl nsw <4 x i32> {{%.*}}, splat (i32 2)
// WRAP-LABEL: @plus_zero___
// WRAP-NOT: shl nsw
// WRAP: shl <4 x i32> {{%.*}}, splat (i32 2)
// WRAP-NOT: shl nsw
int zero = 0;
void plus_zero(uniform float a[], uniform float r[]) { r[programIndex] = a[programIndex + zero]; }
