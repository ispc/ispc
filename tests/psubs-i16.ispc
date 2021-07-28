#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int16 a_max = 32767, a_min = -32768; // max and min signed int16
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_sub(a_min, b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_sub(a_max, -b);
    }
    else {
        RET[programIndex] = saturating_sub(a_max, b);
    } 
}

task void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform int16) -32768;
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform int16) 32767;
    }
    else {
        RET[programIndex] = (uniform int16) 32762;
    } 
}
