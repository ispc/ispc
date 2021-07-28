#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying unsigned int64 a_max = 0xFFFFFFFFFFFFFFFF, a_min = 0; // max and min unsigned int64
    if (programIndex % 2 == 0) {
        #pragma ignore warning(perf)
        RET[programIndex] = saturating_div(a_min, (varying unsigned int64) b);
    }
    else {
        #pragma ignore warning(perf)
        RET[programIndex] = saturating_div(a_max, (varying unsigned int64) b);
    } 
}

task void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (varying unsigned int64) 0; // min unsigned int64
    }
    else {
        RET[programIndex] = (varying unsigned int64) 0x3333333333333333; // max unsigned int64 / 5
    } 
}
