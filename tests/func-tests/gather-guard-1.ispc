#include "test_static.isph"
// rule: skip on arch=xe64
// Masked gathers of 8- and 16-bit elements (emulated: no x86 target has a
// gather of these sizes) whose inactive lanes point into the guard page
// right after aFOO: the harness puts aFOO at the end of its pages, so
// loading such a lane faults. The active lanes read the last elements of
// aFOO.

#define BYTES 1024

// The value of element k of aFOO, viewed as T.
template <typename T> inline T val(int k) { return (T)((k * 37 + 11) & 0x3f); }

// Index for gang lane r: an element near the end of aFOO if the lane is
// active, an element in the guard page after aFOO if it is not.
template <typename T> inline int index(bool active, int r) {
    uniform int cap = BYTES / sizeof(uniform T);
    return active ? cap - 1 - 2 * r : cap + r;
}

template <typename T> int check(uniform float aFOO[], int r) {
    uniform int cap = BYTES / sizeof(uniform T);
    uniform T *uniform p = (uniform T * uniform) aFOO;
    foreach (k = 0 ... cap)
        p[k] = val<T>(k);

    int err = 0;
    // All lanes off. Today the code branches around the access when no lane
    // is on; this guards against a change that drops the branch.
    bool on = r >= programCount;
    int i = index<T>(on, r);
    if (on) {
        #pragma ignore warning(perf)
        err += p[i] != val<T>(i) ? 1 : 0;
    }
    // Every third lane on.
    on = r % 3 == 0;
    i = index<T>(on, r);
    if (on) {
        #pragma ignore warning(perf)
        err += p[i] != val<T>(i) ? 1 : 0;
    }
    // Only the first lane on.
    on = r == 0;
    i = index<T>(on, r);
    if (on) {
        #pragma ignore warning(perf)
        err += p[i] != val<T>(i) ? 1 : 0;
    }
    // The tail of a foreach loop over the last m elements: its last iteration
    // has 2 lanes on. The index j ^ 1 is a permutation of [0, m) (m is even),
    // so the access stays a gather in the partial iteration (on targets
    // without gathers, full iterations may be coalesced into vector loads),
    // and the lanes past the end of the loop point past the end of aFOO.
    uniform int m = programCount + 2;
    foreach (j = 0 ... m) {
        int t = cap - m + (j ^ 1);
        #pragma ignore warning(perf)
        err += p[t] != val<T>(t) ? 1 : 0;
    }
    return err;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int r = (int)aFOO[programIndex] - 1;
    int err = check<int8>(aFOO, r) + check<int16>(aFOO, r);
    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
