#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying unsigned int8 a_max = 0xFF, a_min = 0; // max and min unsigned int8
    if (programIndex % 3 == 0) {
        #pragma ignore warning(perf)
        RET[programIndex] = saturating_mul(a_max, (varying unsigned int8) b);
    }
    else if (programIndex % 3 == 1) {
        #pragma ignore warning(perf)
        RET[programIndex] = saturating_mul(a_min, (varying unsigned int8) -b);
    }
    else {
        #pragma ignore warning(perf)
        RET[programIndex] = saturating_mul((varying unsigned int8) b, (varying unsigned int8) b);
    } 
}

task void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (varying unsigned int8) 0xFF; // max unsigned int8
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (varying unsigned int8) 0; // min unsigned int8
    }
    else {
        RET[programIndex] = (varying unsigned int8) 25;
    } 
}
