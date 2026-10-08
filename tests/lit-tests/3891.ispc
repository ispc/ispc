// Regression test for #3891: on AVX-512 the masked load of hits[i] into tmp
// must stay above the masked store to hits[i]. LLVM 22.1 (llvm/llvm-project#215453)
// sank the load below the store, so the swap degenerated to cur = cur.
// Fixed in LLVM 23.
// RUN: %{ispc} %s --target=avx512skx-x16 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED
// XFAIL: LLVM_22_0+ && !LLVM_23_0+

// CHECK-LABEL: define float @sortInsert(
// CHECK: safe_if_run_true:
// CHECK-NOT: store
// CHECK: call <16 x float> @llvm.masked.load.v16f32.p0(ptr {{[^%]*}}%[[PTR:[^,]+]],
// CHECK: store <16 x float> {{.*}}, ptr %[[PTR]]
// CHECK-NOT: @llvm.masked.load
// CHECK: br label

export uniform float sortInsert(uniform float seed) {
    float hits[programCount];
    for (uniform int i = 0; i < programCount; i++)
        hits[i] = seed;

    float cur = seed + programIndex;

    // lane n runs n iterations, so the trip count varies per lane
    for (uint32 i = 0; i < programIndex; i++) {
        if (abs(hits[i]) > abs(cur)) {
            const float tmp = hits[i];
            hits[i] = cur;
            cur = tmp;
        }
    }

    return reduce_add(cur);
}
