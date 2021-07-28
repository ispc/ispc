#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int8 a_max = 0x7F, a_min = -0x80; // max and min signed int8
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_mul(a_max, (uniform int8) b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_mul(a_min, (uniform int8) b);
    }
    else {
        RET[programIndex] = saturating_mul((uniform int8) b,
                                           (uniform int8) b);
    } 
}

task void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform int8) 0x7F; // max signed int8
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform int8) -0x80; // min signed int8
    }
    else {
        RET[programIndex] = (uniform int8) 25;
    } 
}
