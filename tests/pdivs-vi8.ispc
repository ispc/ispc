#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying int8 a_max = 0x7F, a_min = -0x80; // max and min signed int8
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_div(a_max, (varying int8) b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_div(a_min, (varying int8) b);
    }
    else {
        RET[programIndex] = saturating_div(a_min, (varying int8) -1);
    } 
}

task void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (varying int8) 25.4; // max / 5
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (varying int8) -25.6; // min / 5 
    }
    else {
        RET[programIndex] = (varying int8) 0x7F; // max signed int8
    } 
}
