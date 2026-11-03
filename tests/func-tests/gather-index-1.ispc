#include "test_static.isph"
// rule: skip on arch=xe64
// Gathers with unsigned 32-bit indices, for every element type. The indices
// stay below 2^17 here (gather-index-overflow-1 and -2 cover larger byte
// offsets), so both a dword and a qword gather give the right address.

#define POOL 131072
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

// Indices for gang lane r: 0 and the largest index in lanes 0 and 1, the
// rest spread over the pool.
template <typename T> int run(int r) {
    int err = 0;
    fill<T>();
    for (uniform int round = 0; round < 2; ++round) {
        int off = (r * 40961 + round * 977) % 131072;
        uint32 u = r == 0 ? 131071 : r == 1 ? 0 : (uint32)off;
        // All lanes on.
        err += check<T, uint32>(0, u);
        // About two thirds of the lanes on.
        if (r % 3 != 1) {
            err += check<T, uint32>(0, u);
        }
        // Only the last lane on.
        if (r == programCount - 1) {
            err += check<T, uint32>(0, u);
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
