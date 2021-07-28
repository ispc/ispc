#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int8 a_max = 0xFF, a_min = 0; // max and min unsigned int8
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_mul(a_max, (uniform unsigned int8) b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_mul(a_min, (uniform unsigned int8) -b);
    }
    else {
        RET[programIndex] = saturating_mul((uniform unsigned int8) b,
                                           (uniform unsigned int8) b);
    } 
}

task void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform unsigned int8) 0xFF; // max unsigned int8
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform unsigned int8) 0; // min unsigned int8
    }
    else {
        RET[programIndex] = (uniform unsigned int8) 25;
    } 
}
