#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int16 a_max = 0x7FFF, a_min = -0x8000; // max and min signed int16
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_div(a_max, (uniform int16) b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_div(a_min, (uniform int16) b);
    }
    else {
        RET[programIndex] = saturating_div(a_min, (uniform int16) -1);
    } 
}

task void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform int16) 6553.4; // max / 5
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform int16) -6553.6; // min / 5 
    }
    else {
        RET[programIndex] = (uniform int16) 0x7FFF; // max signed int16
    } 
}
