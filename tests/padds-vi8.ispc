#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying int8 a_max = 127, a_min = -128; // max and min signed int8
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_add(a_max, b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_add(a_min, -b);
    }
    else {
        RET[programIndex] = saturating_add(a_min, b);
    } 
}

task void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (varying int8) 127;
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (varying int8) -128;
    }
    else {
        RET[programIndex] = (varying int8) -123;
    } 
}
