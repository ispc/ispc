// Test for issue #3544: Backward stores should be optimized to contiguous store + shuffle
// instead of scatter operations.
//
// This test verifies that backward memory access patterns like dst[size-1-i] are
// transformed into a contiguous vector store with a shuffle (reverse) rather than
// generating scatter operations. The optimization applies to full vector iterations
// (all-on mask). Partial iterations at the end still use gather/scatter.

// RUN: %{ispc} %s --target=avx2-i32x8 --nowrap --emit-llvm-text -o - 2>&1 | FileCheck %s
// REQUIRES: X86_ENABLED

// Verify no performance warnings are generated (scatter/gather optimized away)
// CHECK-NOT: Performance Warning
// CHECK-NOT: Gather required
// CHECK-NOT: Scatter required

// CHECK-LABEL: @backward_store___
// The main loop should use shuffle + store (no scatter)
// CHECK: foreach_full_body:
// CHECK: shufflevector <8 x i32> {{%[a-zA-Z0-9_.]+}}, <8 x i32> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
// CHECK: store <8 x i32>

// Test backward store pattern - should optimize (unmasked)
unmasked void backward_store(uniform const int src[],
                             uniform int dst[],
                             uniform int size) {
    foreach (i = 0 ... size)
        dst[size - 1 - i] = src[i];
}

// CHECK-LABEL: @backward_load___
// The main loop should use load + shuffle (no gather)
// CHECK: foreach_full_body:
// CHECK: load <8 x i32>
// CHECK: shufflevector <8 x i32> {{%[a-zA-Z0-9_.]+}}, <8 x i32> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>

// Test backward load pattern - should optimize (unmasked)
unmasked void backward_load(uniform const int src[],
                            uniform int dst[],
                            uniform int size) {
    foreach (i = 0 ... size)
        dst[i] = src[size - 1 - i];
}

// CHECK-LABEL: @backward_store_float___
// CHECK: foreach_full_body:
// CHECK: shufflevector <8 x float> {{%[a-zA-Z0-9_.]+}}, <8 x float> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
// CHECK: store <8 x float>

// Test with float type
unmasked void backward_store_float(uniform const float src[],
                                   uniform float dst[],
                                   uniform int size) {
    foreach (i = 0 ... size)
        dst[size - 1 - i] = src[i];
}
