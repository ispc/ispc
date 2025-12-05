// Test popcnt with int64 under masked control flow
#include "test_static.isph"

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int64 value = ((int64)programIndex + 1) << 32 | (programIndex + 1);
    int count;
    
    // popcnt on int64 under masked control flow
    if ((programIndex & 1) == 0) {
        count = popcnt(value);
    } else {
        count = -1;
    }
    
    RET[programIndex] = count;
}

static uniform int manual_popcnt(uniform int64 v) {
    uniform int count = 0;
    while (v != 0) {
        if (v & 1)
            ++count;
        v >>= 1;
    }
    return count;
}

task void result(uniform float RET[]) {
    for (uniform int i = 0; i < programCount; ++i) {
        if ((i & 1) == 0) {
            // Even indices: popcnt of value
            uniform int64 value = ((uniform int64)i + 1) << 32 | (i + 1);
            RET[i] = manual_popcnt(value);
        } else {
            // Odd indices: should be -1 (inactive in the if condition)
            RET[i] = -1;
        }
    }
}
