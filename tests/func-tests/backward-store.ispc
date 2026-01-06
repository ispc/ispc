// Test for fix #3544: backward store optimization
// Tests that dst[size-1-i] = src[i] pattern produces correct results.

#include "test_static.isph"

// Use unmasked + noinline to ensure the optimization triggers
// (all-on mask + prevents inlining/unrolling)
unmasked noinline void do_backward_store(uniform float src[],
                                          uniform float dst[],
                                          uniform int size) {
    foreach (i = 0 ... size) {
        dst[size - 1 - i] = src[i];
    }
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    // Use 4x programCount to test multiple vector iterations
    uniform int size = 4 * programCount;
    uniform float src[64];
    uniform float dst[64];

    // Initialize source array: 1, 2, 3, ... size
    for (uniform int i = 0; i < size; ++i) {
        src[i] = (float)(i + 1);
    }

    // Backward store: dst[size-1-i] = src[i]
    do_backward_store(src, dst, size);

    // Return actual results for comparison
    RET[programIndex] = dst[programIndex];
}

task void result(uniform float RET[]) {
    // Expected: dst[i] = src[size-1-i] = (size-1-i) + 1 = size - i
    // For i = programIndex: expected = 4 * programCount - programIndex
    uniform int size = 4 * programCount;
    RET[programIndex] = (float)(size - programIndex);
}
