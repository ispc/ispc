// Pins today's code for #304 (and #330, the same kernel on targets with a
// gather instruction): two loads from attr[] at offsets (j & ~3) and
// (j & ~3) + 1. Every 4-lane block reads the same two elements.
// Today: sse4-i32x4 needs no gather (one index per 4-lane block becomes a
// broadcast); avx2-i32x8 gathers with constant offsets (vgatherdps);
// avx512skx-x16 emits 2 gathers. Both AVX targets warn twice.
// TODO(#304, #330): replace repeated reads with scalar loads and broadcasts
// on the AVX targets too, removing the gather warnings. Only load an element
// when a lane reading it is active.

// RUN: %{ispc} %s --target=sse4-i32x4 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=SSE4
// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=AVX2
// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=SKX
// RUN: %{ispc} %s --target=sse4-i32x4 --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -o /dev/null 2> %t.avx2
// RUN: FileCheck %s --input-file=%t.avx2 --check-prefixes=WARN
// RUN: FileCheck %s --input-file=%t.avx2 --check-prefix=COUNT2
// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --nowrap -o /dev/null 2> %t.avx512skx
// RUN: FileCheck %s --input-file=%t.avx512skx --check-prefixes=WARN
// RUN: FileCheck %s --input-file=%t.avx512skx --check-prefix=COUNT2

// REQUIRES: X86_ENABLED

// The warning checks match in any order; the COUNT prefix checks that
// there are no other warnings.
// COUNT2-COUNT-2: Performance Warning
// COUNT2-NOT: Performance Warning

// NOWARN-NOT: Performance Warning

// SSE4-LABEL: @computePatchHull___
// SSE4-NOT: call {{.*}}@llvm.x86.{{.*}}gather
// SSE4: call <4 x float> @blossom___
// SSE4-NOT: call {{.*}}@llvm.x86.{{.*}}gather

// AVX2-LABEL: @computePatchHull___
// AVX2: call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> poison, ptr {{.*}}%attr, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 16,
// AVX2: call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> poison, ptr {{.*}}%attr, <8 x i32> <i32 4, i32 4, i32 4, i32 4, i32 20,
// AVX2-NOT: call {{.*}}@llvm.x86.avx2.gather.q.ps.256(

// SKX-LABEL: @computePatchHull___
// SKX: call <16 x float> @llvm.x86.avx512.mask.gather.dps.512(
// SKX: call <16 x float> @llvm.x86.avx512.mask.gather.dps.512(
// SKX-NOT: call {{.*}}@llvm.x86.avx512.mask.gather

extern unmasked float blossom(float, float);

unmasked void
computePatchHull(uniform float attr[]) {
    uniform float ubcp[16];
    foreach (j = 0 ... 16) {
        // WARN-DAG: 304.ispc:[[#@LINE+1]]:21: Performance Warning: Gather required to load value.
        float cp0 = attr[(j & ~3)];
        // WARN-DAG: 304.ispc:[[#@LINE+1]]:21: Performance Warning: Gather required to load value.
        float cp1 = attr[(j & ~3)+1];

        ubcp[j] =  blossom(cp0,cp1);
    }
}
