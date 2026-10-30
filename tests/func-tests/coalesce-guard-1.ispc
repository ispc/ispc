#include "test_static.isph"
// rule: skip on arch=xe64
// Gathers with a uniform base and constant offsets, which can be coalesced
// into vector loads (today only 32-bit ones, on targets without hardware
// gathers), whose last element is the last element of aFOO.
// The harness puts aFOO at the end of its pages, so a coalesced load that
// reads past the elements the gang uses faults. Here the elements the gang
// reads are contiguous (no gaps); coalesce-guard-2 has gaps.

#define BYTES 1024

// The value of element k of aFOO, viewed as T.
template <typename T> inline T val(int k) { return (T)((k * 37 + 11) & 0x3f); }

template <typename T> int check(uniform float aFOO[]) {
    uniform int cap = BYTES / sizeof(uniform T);
    uniform T *uniform p = (uniform T * uniform) aFOO;
    foreach (k = 0 ... cap)
        p[k] = val<T>(k);

    int err = 0;
    // Two adjacent elements per lane.
    uniform int first = cap - 2 * programCount;
    uniform T *uniform q = p + first;
    #pragma ignore warning(perf)
    T a = q[2 * programIndex];
    #pragma ignore warning(perf)
    T b = q[2 * programIndex + 1];
    err += a != val<T>(first + 2 * programIndex) || b != val<T>(first + 2 * programIndex + 1) ? 1 : 0;
    // Four adjacent elements per lane, for 32-bit elements (for 64-bit ones
    // 4 * 64 elements do not fit in aFOO).
    if (sizeof(uniform T) == 4) {
        first = cap - 4 * programCount;
        q = p + first;
        for (uniform int c = 0; c < 4; ++c) {
            #pragma ignore warning(perf)
            T v = q[4 * programIndex + c];
            err += v != val<T>(first + 4 * programIndex + c) ? 1 : 0;
        }
    }
    return err;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int err = check<int32>(aFOO) + check<float>(aFOO) + check<int64>(aFOO) + check<double>(aFOO);
    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
