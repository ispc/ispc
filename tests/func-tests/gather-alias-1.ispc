#include "test_static.isph"
// rule: skip on arch=xe64
// Two loads that the compiler can coalesce (one base, constant offsets) with
// a store between them through a pointer that aliases the base at run time.
// The second load must see the store, so the loads must not be merged into
// one access before it.

task void f_f(uniform float RET[], uniform float aFOO[]) {
    // q is aFOO at run time; the compiler cannot prove it.
    uniform float *uniform q = aFOO[0] == 1 ? aFOO : RET;
    int err = 0;
    // Every block of 4 lanes reads elements 4g and 4g + 1 (the #304 shape),
    // with a varying store to element 4g + 1 between the reads.
    foreach (j = 0 ... 64) {
        #pragma ignore warning(perf)
        float a = aFOO[(j & ~3)];
        #pragma ignore warning(perf)
        q[(j & ~3) + 1] = -a;
        #pragma ignore warning(perf)
        float b = aFOO[(j & ~3) + 1];
        err += a != (j & ~3) + 1 || b != -a ? 1 : 0;
    }
    // Adjacent elements 2i and 2i + 1 with a uniform store between the
    // reads, to the element the last lane reads second.
    foreach (i = 32 ... 32 + programCount) {
        #pragma ignore warning(perf)
        float a = aFOO[2 * i];
        q[2 * (32 + programCount - 1) + 1] = 1000;
        #pragma ignore warning(perf)
        float b = aFOO[2 * i + 1];
        err += a != 2 * i + 1 ? 1 : 0;
        err += b != (i == 32 + programCount - 1 ? 1000 : 2 * i + 2) ? 1 : 0;
    }
    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
