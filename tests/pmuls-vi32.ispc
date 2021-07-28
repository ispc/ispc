#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying int32 a_max = 0x7FFFFFFF, a_min = 0x80000000; // max and min signed int32
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_mul(a_max, (varying int32) b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_mul(a_min, (varying int32) b);
    }
    else {
        RET[programIndex] = saturating_mul((varying int32) b,
                                           (varying int32) b);
    } 
}

task void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (varying int32) 0x7FFFFFFF; // max signed int32
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (varying int32) 0x80000000; // min signed int32
    }
    else {
        RET[programIndex] = (varying int32) 25;
    } 
}
