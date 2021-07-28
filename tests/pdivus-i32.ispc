#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int32 a_max = 0xFFFFFFFF, a_min = 0; // max and min unsigned int32
    if (programIndex % 2 == 0) {
        RET[programIndex] = saturating_div(a_min, (uniform unsigned int32) b);
    }
    else {
        RET[programIndex] = saturating_div(a_max, (uniform unsigned int32) b);
    } 
}

task void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (uniform unsigned int32) 0; // min unsigned int32
    }
    else {
        RET[programIndex] = (uniform unsigned int32) 0x33333333; // max unsigned int32 / 5
    } 
}
