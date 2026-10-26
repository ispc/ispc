#include "test_static.isph"
// rule: skip on arch=xe64
// Masked gathers with negative indices, and a masked load, whose inactive
// lanes point into the guard page right before vint (the harness starts vint
// right after an inaccessible page), for 8- to 64-bit elements. The active
// lanes read the first elements of vint.

#define BYTES 1024

// The value of element k of vint, viewed as T.
template <typename T> inline T val(int k) { return (T)((k * 37 + 11) & 0x3f); }

template <typename T> int check(uniform int vint[], uniform int one, int r) {
    uniform int cap = BYTES / sizeof(uniform T);
    uniform T *uniform p = (uniform T * uniform) vint;
    foreach (k = 0 ... cap)
        p[k] = val<T>(k);
    // The base is element K; active lanes use indices in [-K, 0), inactive
    // ones indices below -K, which are in the guard page.
    uniform int K = 2 * programCount;
    uniform T *uniform base = p + K;

    int err = 0;
    // All lanes off (today the code branches around the access when no lane
    // is on), every third lane on, only the first lane on.
    for (uniform int c = 0; c < 3; ++c) {
        bool on = c == 0 ? r >= programCount : c == 1 ? r % 3 == 0 : r == 0;
        int i = on ? -1 - 2 * r : -K - 1 - r;
        if (on) {
            #pragma ignore warning(perf)
            err += base[i] != val<T>(K + i) ? 1 : 0;
        }
    }
    // The tail of a foreach loop that walks down to element 0: its last
    // iteration has 2 lanes on, and the lanes past the end of the loop point
    // before vint. j ^ one (one is 1 at run time) is a permutation of [0, m)
    // (m is even) that the compiler cannot turn into vector loads.
    uniform int m = programCount + 2;
    uniform T *uniform top = p + m - 1;
    foreach (j = 0 ... m) {
        int t = -(j ^ one);
        #pragma ignore warning(perf)
        err += top[t] != val<T>(m - 1 + t) ? 1 : 0;
    }
    // A masked contiguous load: lanes below d (3 at run time) are off and
    // point before vint.
    uniform int d = 3 * one;
    if (programIndex >= d)
        err += p[programIndex - d] != val<T>(programIndex - d) ? 1 : 0;
    return err;
}

task void f_fi(uniform float RET[], uniform float aFOO[], uniform int vint[]) {
    int r = (int)aFOO[programIndex] - 1;
    uniform int one = (uniform int)aFOO[0];
    int err = check<int8>(vint, one, r) + check<int16>(vint, one, r) + check<int32>(vint, one, r) +
              check<float>(vint, one, r) + check<int64>(vint, one, r) + check<double>(vint, one, r);
    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
