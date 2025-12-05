// Test that popcnt respects mask in masked control flow
#include "test_static.isph"

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int value = programIndex + 1;
    int count;
    
    // Test 1: popcnt under if condition - first half gets popcnt, second half gets 0
    if (programIndex < programCount / 2) {
        count = popcnt(value);
    } else {
        count = 0;
    }
    
    RET[programIndex] = count;
}

static uniform int manual_popcnt(uniform int v) {
    uniform int count = 0;
    while (v != 0) {
        if (v & 1)
            ++count;
        v >>= 1;
    }
    return count;
}

task void result(uniform float RET[]) {
    // First half should have popcnt values
    for (uniform int i = 0; i < programCount / 2; ++i) {
        RET[i] = manual_popcnt(i + 1);
    }
    // Second half should be 0 (inactive lanes in the if condition)
    for (uniform int i = programCount / 2; i < programCount; ++i) {
        RET[i] = 0;
    }
}
