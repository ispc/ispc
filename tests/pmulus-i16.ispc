#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int16 a_max = 0xFFFF, a_min = 0; // max and min unsigned int16
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_mul(a_max, (uniform unsigned int16) b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_mul(a_min, (uniform unsigned int16) -b);
    }
    else {
        RET[programIndex] = saturating_mul((uniform unsigned int16) b,
                                           (uniform unsigned int16) b);
    } 
}

task void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform unsigned int16) 0xFFFF; // max unsigned int16
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform unsigned int16) 0; // min unsigned int16
    }
    else {
        RET[programIndex] = (uniform unsigned int16) 25;
    } 
}
