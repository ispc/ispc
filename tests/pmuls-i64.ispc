#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int64 a_max = 0x7FFFFFFFFFFFFFFF, a_min = 0x8000000000000000; // max and min signed int64
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_mul(a_max, (uniform int64) b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_mul(a_min, (uniform int64) b);
    }
    else {
        RET[programIndex] = saturating_mul((uniform int64) b,
                                           (uniform int64) b);
    } 
}

task void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform int64) 0x7FFFFFFFFFFFFFFF; // max signed int64
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform int64) 0x8000000000000000; // min signed int64
    }
    else {
        RET[programIndex] = (uniform int64) 25;
    } 
}
