#include "test_static.isph"
// rule: skip on arch=xe64
// Masked scatters with negative indices, and masked stores, whose inactive
// lanes point into the guard page right before vint2 (the harness starts
// vint2 right after an inaccessible page), for 8- to 64-bit elements. The
// active lanes write the first elements of vint2; the test checks them and
// that no other element changed.

#define BYTES 1024

// The value of element k of vint2, viewed as T, before the stores.
template <typename T> inline T val(int k) { return (T)((k * 37 + 11) & 0x3f); }

template <typename T> void fill(uniform T p[]) {
    uniform int cap = BYTES / sizeof(uniform T);
    foreach (k = 0 ... cap)
        p[k] = val<T>(k);
}

// The number of elements of p that are wrong, given that exactly the
// elements [lo, hi) were written with val() + 1.
template <typename T> uniform int wrong(uniform T p[], uniform int lo, uniform int hi) {
    uniform int cap = BYTES / sizeof(uniform T);
    uniform int bad = 0;
    foreach (k = 0 ... cap)
        bad += reduce_add(p[k] != (k >= lo && k < hi ? (T)(val<T>(k) + 1) : val<T>(k)) ? 1 : 0);
    return bad;
}

template <typename T> int check(uniform int vint2[], uniform int one, int r) {
    uniform T *uniform p = (uniform T * uniform) vint2;
    // The base is element K; active lanes use indices in [-K, 0), inactive
    // ones indices below -K, which are in the guard page.
    uniform int K = 2 * programCount;
    uniform T *uniform base = p + K;

    int err = 0;
    // Scatters: all lanes off (today the code branches around the access when
    // no lane is on), every third lane on, only the first lane on.
    for (uniform int c = 0; c < 3; ++c) {
        fill(p);
        bool on = c == 0 ? r >= programCount : c == 1 ? r % 3 == 0 : r == 0;
        int i = on ? -1 - 2 * r : -K - 1 - r;
        if (on) {
            #pragma ignore warning(perf)
            base[i] = (T)(val<T>(K + i) + 1);
        }
        uniform int changed = 0;
        uniform int cap = BYTES / sizeof(uniform T);
        foreach (k = 0 ... cap)
            changed += reduce_add(p[k] != val<T>(k) ? 1 : 0);
        err += changed != reduce_add(on ? 1 : 0) ? 1 : 0;
        if (on) {
            #pragma ignore warning(perf)
            err += base[i] != (T)(val<T>(K + i) + 1) ? 1 : 0;
        }
    }
    // Scatters in the tail of a foreach loop that walks down to element 0:
    // its last iteration has 2 lanes on, and the lanes past the end of the
    // loop point before vint2. j ^ 1 is a permutation of [0, m) (m is even)
    // that the compiler cannot turn into a vector store.
    uniform int m = programCount + 2;
    uniform T *uniform top = p + m - 1;
    fill(p);
    foreach (j = 0 ... m) {
        int t = -(j ^ 1);
        #pragma ignore warning(perf)
        top[t] = (T)(val<T>(m - 1 + t) + 1);
    }
    err += wrong(p, 0, m) != 0 ? 1 : 0;
    // A store in reverse order that ends at element 0 (a scatter or per-lane
    // stores): the lanes past the end of the loop point before vint2.
    fill(p);
    foreach (j = 0 ... m)
        top[-j] = (T)(val<T>(m - 1 - j) + 1);
    err += wrong(p, 0, m) != 0 ? 1 : 0;
    // A masked contiguous store: lanes below d (3 at run time) are off and
    // point before vint2.
    fill(p);
    uniform int d = 3 * one;
    if (programIndex >= d)
        p[programIndex - d] = (T)(val<T>(programIndex - d) + 1);
    err += wrong(p, 0, programCount - d) != 0 ? 1 : 0;
    fill(p);
    return err;
}

task void f_di(uniform float RET[], uniform double aFOO[], uniform int vint2[]) {
    int r = (int)aFOO[programIndex] - 1;
    uniform int one = (uniform int)aFOO[0];
    int err = check<int8>(vint2, one, r) + check<int16>(vint2, one, r) + check<int32>(vint2, one, r) +
              check<float>(vint2, one, r) + check<int64>(vint2, one, r) + check<double>(vint2, one, r);
    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
