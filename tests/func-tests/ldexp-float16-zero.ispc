#include "test_static.isph"
// rule: skip on arch=ppc64le
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = 0.0f;
    float16 testVal = ldexp(aFOO[0]-1, -2);
    RET[programIndex] += (testVal == 0.0f16) ? 0 : 1;
    testVal = ldexp(0.0f16, -2);
    RET[programIndex] += (testVal == 0.0f16) ? 0 : 1;
    testVal = ldexp(-0.0f16, -2);
    RET[programIndex] += (testVal == -0.0f16) ? 0 : 1;
    testVal = ldexp(0.0f16, 2);
    RET[programIndex] += (testVal == 0.0f16) ? 0 : 1;
    testVal = ldexp(-0.0f16, 2);
    RET[programIndex] += (testVal == -0.0f16) ? 0 : 1;    
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}

