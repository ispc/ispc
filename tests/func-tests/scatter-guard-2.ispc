#include "test_static.isph"
// rule: skip on arch=xe64
// Masked scatters of 32- and 64-bit elements (native scatters on targets
// that have them) whose inactive lanes point into the guard page right after
// vdouble: the harness puts vdouble (2048 bytes) at the end of its pages, so
// storing to such a lane faults. The active lanes write the last elements of
// vdouble; the test checks them and that no other element changed.

#define BYTES 2048

// The value of element k of vdouble, viewed as T, before the scatters.
template <typename T> inline T val(int k) { return (T)((k * 37 + 11) & 0x3f); }

// Index for gang lane r: an element near the end of vdouble if the lane is
// active, an element in the guard page after vdouble if it is not.
template <typename T> inline int index(bool active, int r) {
    uniform int cap = BYTES / sizeof(uniform T);
    return active ? cap - 1 - 2 * r : cap + r;
}

template <typename T> void fill(uniform T p[]) {
    uniform int cap = BYTES / sizeof(uniform T);
    foreach (k = 0 ... cap)
        p[k] = val<T>(k);
}

// 1 in every lane if the elements of p differ from val() in other than
// `expected` places, or if the lane is on and its element is not v.
template <typename T> int verify(uniform T p[], bool on, int i, T v, uniform int expected) {
    uniform int cap = BYTES / sizeof(uniform T);
    uniform int changed = 0;
    foreach (k = 0 ... cap)
        changed += reduce_add(p[k] != val<T>(k) ? 1 : 0);
    int err = changed != expected ? 1 : 0;
    if (on) {
        #pragma ignore warning(perf)
        err += p[i] != v ? 1 : 0;
    }
    fill(p);
    return err;
}

template <typename T> int check(uniform double vdouble[], int r) {
    uniform T *uniform p = (uniform T * uniform) vdouble;
    uniform int cap = BYTES / sizeof(uniform T);
    fill(p);

    int err = 0;
    // All lanes off (today the code branches around the access when no lane
    // is on), every third lane on, only the first lane on.
    for (uniform int c = 0; c < 3; ++c) {
        bool on = c == 0 ? r >= programCount : c == 1 ? r % 3 == 0 : r == 0;
        int i = index<T>(on, r);
        T v = (T)(val<T>(i) + 1);
        if (on) {
            #pragma ignore warning(perf)
            p[i] = v;
        }
        err += verify(p, on, i, v, reduce_add(on ? 1 : 0));
    }
    // The tail of a foreach loop over the last m elements: its last iteration
    // has 2 lanes on. The index j ^ 1 is a permutation of [0, m) (m is even)
    // that the compiler cannot turn into a vector store, and the lanes past
    // the end of the loop point past the end of vdouble.
    uniform int m = programCount + 2;
    foreach (j = 0 ... m) {
        int t = cap - m + (j ^ 1);
        #pragma ignore warning(perf)
        p[t] = (T)(val<T>(t) + 1);
    }
    uniform int changed = 0;
    foreach (k = 0 ... cap)
        changed += reduce_add(p[k] != (k >= cap - m ? (T)(val<T>(k) + 1) : val<T>(k)) ? 1 : 0);
    err += changed != 0 ? 1 : 0;
    fill(p);
    return err;
}

task void f_du(uniform float RET[], uniform double vdouble[], uniform double b) {
    int r = (int)vdouble[programIndex] - 1;
    int err = check<int32>(vdouble, r) + check<int64>(vdouble, r) + check<float>(vdouble, r) +
              check<double>(vdouble, r);
    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
