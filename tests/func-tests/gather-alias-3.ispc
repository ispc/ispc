#include "test_static.isph"
// rule: skip on arch=xe64
// Loads that the compiler can coalesce (one base, constant offsets) around an
// atomic that changes an element the second load reads, and loads from two
// different bases next to each other. The second load must see the atomic,
// and loads from different bases must not be merged.

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int data[2 * programCount + 2];
    uniform int other[2 * programCount + 2];
    for (uniform int i = 0; i < 2 * programCount + 2; ++i) {
        data[i] = (int)aFOO[i % programCount] + i;
        other[i] = -data[i];
    }
    int err = 0;
    // Adjacent elements 2i and 2i + 1 with an atomic between the reads, to the
    // element the last lane reads second.
    #pragma ignore warning(perf)
    int a = data[2 * programIndex];
    atomic_add_global(&data[2 * (programCount - 1) + 1], 1000);
    #pragma ignore warning(perf)
    int b = data[2 * programIndex + 1];
    #pragma ignore warning(perf)
    int e = (int)aFOO[(2 * programIndex) % programCount] + 2 * programIndex;
    err += a != e ? 1 : 0;
    #pragma ignore warning(perf)
    e = (int)aFOO[(2 * programIndex + 1) % programCount] + 2 * programIndex + 1;
    err += b != e + (programIndex == programCount - 1 ? 1000 : 0) ? 1 : 0;
    // The same offsets from two bases, interleaved.
    #pragma ignore warning(perf)
    int c = data[2 * programIndex];
    #pragma ignore warning(perf)
    int d = other[2 * programIndex + 1];
    #pragma ignore warning(perf)
    int f = data[2 * programIndex + 1];
    err += c != a ? 1 : 0;
    err += d != -e ? 1 : 0;
    err += f != b ? 1 : 0;
    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
