#include "test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = 0.0f;
    double testVal = ldexp(aFOO[0]-1, -2);
    RET[programIndex] += (testVal == 0.0d) ? 0 : 1;
    testVal = ldexp(0.0d, -2);
    RET[programIndex] += (testVal == 0.0d) ? 0 : 1;
    testVal = ldexp(-0.0d, -2);
    RET[programIndex] += (testVal == -0.0d) ? 0 : 1;
    testVal = ldexp(0.0d, 2);
    RET[programIndex] += (testVal == 0.0d) ? 0 : 1;
    testVal = ldexp(-0.0d, 2);
    RET[programIndex] += (testVal == -0.0d) ? 0 : 1;    
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}

