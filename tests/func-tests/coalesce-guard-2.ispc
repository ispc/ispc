#include "test_static.isph"
// rule: skip on arch=xe64
// Gathers with a uniform base and constant offsets, which can be coalesced
// into vector loads (today only 32-bit ones, on targets without hardware
// gathers), whose last element is the last element of aFOO.
// The harness puts aFOO at the end of its pages, so a coalesced load that
// reads past the elements the gang uses faults. Here the elements the gang
// reads have gaps: a load that covers a whole group of elements, including
// the unused ones after the last element read, faults (coalesce-guard-1 has
// no gaps).

#define BYTES 1024

// The value of element k of aFOO, viewed as T.
template <typename T> inline T val(int k) { return (T)((k * 37 + 11) & 0x3f); }

template <typename T> int check(uniform float aFOO[]) {
    uniform int cap = BYTES / sizeof(uniform T);
    uniform T *uniform p = (uniform T * uniform) aFOO;
    foreach (k = 0 ... cap)
        p[k] = val<T>(k);

    int err = 0;
    // Blocks of 4 adjacent lanes read 4 adjacent elements, and the blocks are
    // S elements apart, so S - 4 elements after each block are unused. The
    // last element the gang reads is the last element of aFOO.
    uniform int S = sizeof(uniform T) == 4 ? 16 : 8;
    uniform int span = ((programCount - 1) >> 2) * S + ((programCount - 1) & 3) + 1;
    uniform int first = cap - span;
    uniform T *uniform q = p + first;
    #pragma ignore warning(perf)
    T a = q[(programIndex >> 2) * S + (programIndex & 3)];
    err += a != val<T>(first + (programIndex >> 2) * S + (programIndex & 3)) ? 1 : 0;
    // Every third element, or every second one where 3 * programCount
    // elements do not fit in aFOO.
    uniform int stride = 3 * (programCount - 1) + 1 <= cap ? 3 : 2;
    first = cap - stride * (programCount - 1) - 1;
    q = p + first;
    #pragma ignore warning(perf)
    T b = q[stride * programIndex];
    err += b != val<T>(first + stride * programIndex) ? 1 : 0;
    // Elements (j & ~3) and (j & ~3) + 1 of a foreach loop (the #304 shape):
    // every block of 4 lanes reads the first two of 4 elements, and the last
    // element read is the last element of aFOO.
    first = cap - 4 * ((programCount + 3) / 4) + 2;
    q = p + first;
    foreach (j = 0 ... 4 * ((programCount + 3) / 4)) {
        #pragma ignore warning(perf)
        T u = q[(j & ~3)];
        #pragma ignore warning(perf)
        T w = q[(j & ~3) + 1];
        err += u != val<T>(first + (j & ~3)) || w != val<T>(first + (j & ~3) + 1) ? 1 : 0;
    }
    return err;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int err = check<int32>(aFOO) + check<float>(aFOO) + check<int64>(aFOO) + check<double>(aFOO);
    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
