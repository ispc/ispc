#include "test_static.isph"
// rule: skip on arch=ppc64le
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = 0;

    // Continuation of the test from fmod-float16-varying-1.ispc.
    // It's split into several files to reduce object size with O0.
    // Case 5.1: x is +inf, y is positive
    float16 x = aFOO[programCount / 2] / 0.0;
    float16 y = 0.8;
    float16 testVal = fmod(x, y);
    RET[programIndex] += (isnan(testVal) && (signbits(testVal) == signbits(x))) ? 0 : 1;

    // Case 5.2: x is -inf, y is positive
    x = -aFOO[programCount / 2] / 0.0;
    y = 0.8;
    testVal = fmod(x, y);
    RET[programIndex] += (isnan(testVal) && (signbits(testVal) == signbits(x))) ? 0 : 1;
}

task void result(uniform float RET[]) { RET[0] = 0; }
