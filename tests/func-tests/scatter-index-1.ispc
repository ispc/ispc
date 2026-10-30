#include "test_static.isph"
// rule: skip on arch=xe64
// Scatters with unsigned 32-bit and signed and unsigned 16-bit indices, for
// every element type. As for gathers, a hardware scatter sign-extends its
// 32-bit index, so the unsigned 16-bit indices are >= 2^15, and the signed
// ones are negative too. Each lane writes its own element; the test checks
// the written elements and that no other element changed.

#define POOL 65536
static uniform int64 pool[POOL];

// A hash of k that every bit of k changes, so that a wrong index gives a
// wrong value.
static inline int64 hash(int64 k) { return ((k * 0x9E3779B1ll) >> 16) & 0x3fff; }

// The value of element k of the pool, viewed as T, before the scatter.
template <typename T> inline T val(int64 k) { return (T)hash(k); }

template <typename T> void fill() {
    uniform T *uniform mem = (uniform T * uniform)&pool[0];
    foreach (k = 0 ... POOL)
        mem[k] = val<T>(k);
}

// Scatters to base[idx] in the active lanes, where base is element `first`
// of the pool, and returns 1 in each active lane if its element is wrong
// afterwards or if the number of changed elements is not the number of
// active lanes. The indices of the lanes must differ.
template <typename T, typename I> int check(uniform int64 first, I idx) {
    uniform T *uniform mem = (uniform T * uniform)&pool[0];
    uniform T *uniform base = mem + first;
    T v = (T)(hash(first + (int64)idx) + 1);
    #pragma ignore warning(perf)
    base[idx] = v;
    uniform int active = popcnt(lanemask());
    // Count over the whole gang: the caller may run with some lanes off.
    uniform int changed;
    unmasked {
        int changed_lanes = 0;
        foreach (k = 0 ... POOL)
            changed_lanes += mem[k] != val<T>(k) ? 1 : 0;
        changed = reduce_add(changed_lanes);
    }
    #pragma ignore warning(perf)
    int bad = mem[first + (int64)idx] != v || changed != active ? 1 : 0;
    // Restore the pool for the next scatter.
    fill<T>();
    return bad;
}

// Indices for gang lane r: the extremes of each index range in lanes 0 and
// 1, the rest spread over it; all different.
template <typename T> int run(int r) {
    int err = 0;
    fill<T>();
    for (uniform int round = 0; round < 2; ++round) {
        int off = (r * 4099 + round * 977) % 32768;
        uint32 w = r == 0 ? 65535 : r == 1 ? 0 : (uint32)(off * 2 + 1);
        uint16 u = r == 0 ? 65535 : r == 1 ? 32768 : (uint16)(32768 + off);
        int16 s = r == 0 ? -32768 : r == 1 ? 32767 : (int16)(off - 16384);
        // All lanes on.
        err += check<T, uint32>(0, w);
        err += check<T, uint16>(0, u);
        err += check<T, int16>(32768, s);
        // About two thirds of the lanes on.
        if (r % 3 != 1) {
            err += check<T, uint32>(0, w);
            err += check<T, uint16>(0, u);
            err += check<T, int16>(32768, s);
        }
        // Only the last lane on.
        if (r == programCount - 1) {
            err += check<T, uint32>(0, w);
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
