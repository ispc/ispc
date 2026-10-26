#include "test_static.isph"
// rule: skip on arch=xe64
// Contiguous stores in the tail of a foreach loop that ends at the last
// element of aFOO, for 8-, 16-, 32- and 64-bit elements. The harness puts
// aFOO at the end of its pages, so a full-width vector store in the tail
// iteration faults: the tail must use a masked store (or narrower stores).
// The test also checks that the elements before the loop do not change.

#define BYTES 1024

// The value of element k of aFOO, viewed as T, before the stores.
template <typename T> inline T val(int k) { return (T)((k * 37 + 11) & 0x3f); }

template <typename T> int check(uniform float aFOO[]) {
    uniform int cap = BYTES / sizeof(uniform T);
    uniform T *uniform p = (uniform T * uniform) aFOO;

    int err = 0;
    // Tails of 1, 2, programCount / 2 + 1 and programCount - 1 lanes, after
    // up to two full iterations.
    for (uniform int t = 0; t < 4; ++t) {
        uniform int tail = t == 0 ? 1 : t == 1 ? 2 : t == 2 ? programCount / 2 + 1 : programCount - 1;
        if (tail < 1 || tail >= programCount)
            continue;
        foreach (k = 0 ... cap)
            p[k] = val<T>(k);
        uniform int full = min((uniform int)2, (cap - tail) / programCount);
        uniform int s = cap - full * programCount - tail;
        foreach (i = s ... cap)
            p[i] = (T)(val<T>(i) + 1);
        foreach (k = 0 ... cap)
            err += p[k] != (k >= s ? (T)(val<T>(k) + 1) : val<T>(k)) ? 1 : 0;
    }
    return err;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int err = check<int8>(aFOO) + check<int16>(aFOO) + check<int32>(aFOO) + check<float>(aFOO) +
              check<int64>(aFOO) + check<double>(aFOO);
    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
