#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int8 a_max = 255, a_min = 0; // max and min unsigned int8
    if (programIndex % 2 == 0) {
        RET[programIndex] = saturating_add(a_max, b);
    }
    else {
        RET[programIndex] = saturating_add(a_min, b);
    } 
}

task void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (uniform unsigned int8) 255;
    }
    else {
        RET[programIndex] = (uniform unsigned int8) 5;
    } 
}
