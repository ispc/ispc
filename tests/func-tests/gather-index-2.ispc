#include "test_static.isph"
// rule: skip on arch=xe64
// Gathers with signed and unsigned 16-bit indices, for every element type:
// the hardware gather sign-extends its 32-bit index, so an unsigned index
// must be zero-extended before it is used. The unsigned indices are >= 2^15,
// where the two extensions differ, and the signed ones are negative too.

#define POOL 65536
static uniform int64 pool[POOL];

// A hash of k that every bit of k changes, so that a wrong index gives a
// wrong value.
static inline int64 hash(int64 k) { return ((k * 0x9E3779B1ll) >> 16) & 0x3fff; }

// The value of element k of the pool, viewed as T.
template <typename T> inline T val(int64 k) { return (T)hash(k); }

template <typename T> void fill() {
    uniform T *uniform mem = (uniform T * uniform)&pool[0];
    foreach (k = 0 ... POOL)
        mem[k] = val<T>(k);
}

// 1 in each active lane whose gathered value is wrong.
template <typename T, typename I> int check(uniform int64 first, I idx) {
    uniform T *uniform base = (uniform T * uniform)&pool[0] + first;
    #pragma ignore warning(perf)
    T g = base[idx];
    return g != val<T>(first + (int64)idx) ? 1 : 0;
}

// Indices for gang lane r: the extremes of the index range in lanes 0 and 1,
// the rest spread over it.
template <typename T> int run(int r) {
    int err = 0;
    fill<T>();
    for (uniform int round = 0; round < 2; ++round) {
        int off = (r * 4099 + round * 977) % 32768;
        uint16 u = r == 0 ? 65535 : r == 1 ? 32768 : (uint16)(32768 + off);
        int16 s = r == 0 ? -32768 : r == 1 ? 32767 : (int16)(off - 16384);
        // All lanes on.
        err += check<T, uint16>(0, u);
        err += check<T, int16>(32768, s);
        // About two thirds of the lanes on.
        if (r % 3 != 1) {
            err += check<T, uint16>(0, u);
            err += check<T, int16>(32768, s);
        }
        // Only the last lane on.
        if (r == programCount - 1) {
            err += check<T, uint16>(0, u);
            err += check<T, int16>(32768, s);
        }
    }
    return err;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int r = (int)aFOO[programIndex] - 1;
    int err = run<int8>(r) + run<int16>(r) + run<int32>(r) + run<int64>(r) + run<float>(r) + run<double>(r);
    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
