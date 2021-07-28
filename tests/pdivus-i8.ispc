#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int8 a_max = 0xFF, a_min = 0; // max and min unsigned int8
    if (programIndex % 2 == 0) {
        RET[programIndex] = saturating_div(a_min, (uniform unsigned int8) b);
    }
    else {
        RET[programIndex] = saturating_div(a_max, (uniform unsigned int8) b);
    } 
}

task void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (uniform unsigned int8) 0; // min unsigned int8
    }
    else {
        RET[programIndex] = (uniform unsigned int8) 0x33; // max unsigned int8 / 5
    } 
}
