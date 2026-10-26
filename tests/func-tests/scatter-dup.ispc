#include "test_static.isph"
// Scatters where several active lanes store to the same element. Which
// active lane's value is stored is undefined, but it must be the value of
// one of the active lanes that store there: never an inactive lane's value,
// and never a mix of the bytes of several values. Elements that no active
// lane stores to keep their value.

#define SLOTS 5

// The value lane r stores; every byte differs between lanes.
template <typename T> inline T lane_val(int r) { return (T)0; }
template <> inline int8 lane_val<int8>(int r) { return (int8)(r + 1); }
template <> inline int32 lane_val<int32>(int r) { return (r + 1) * 0x01010101; }
template <> inline int64 lane_val<int64>(int r) { return (int64)(r + 1) * 0x0101010101010101; }
template <> inline float lane_val<float>(int r) { return floatbits((r + 1) * 0x01010101); }

template <typename T> int check(int r) {
    uniform T slot[SLOTS];
    uniform T init = extract(lane_val<T>(100), 0);
    int err = 0;
    // Lane r stores to slot r % 3, so slots 3 and 4 are never stored to.
    // Masks: all lanes on, every second lane on, lanes with r % 3 == 0 off.
    for (uniform int c = 0; c < 3; ++c) {
        for (uniform int k = 0; k < SLOTS; ++k)
            slot[k] = init;
        bool on = c == 0 ? true : c == 1 ? r % 2 == 0 : r % 3 != 0;
        int s = r % 3;
        if (on) {
            #pragma ignore warning(perf)
            slot[s] = lane_val<T>(r);
        }
        // Check every slot against the lanes that are on and store to it.
        for (uniform int k = 0; k < SLOTS; ++k) {
            uniform T v = slot[k];
            uniform bool stored = any(on && s == k);
            uniform bool match = any(on && s == k && lane_val<T>(r) == v);
            err += stored ? (match ? 0 : 1) : (v == init ? 0 : 1);
        }
    }
    return err;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int r = (int)aFOO[programIndex] - 1;
    RET[programIndex] = check<int8>(r) + check<int32>(r) + check<int64>(r) + check<float>(r);
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
