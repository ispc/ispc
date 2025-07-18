#include "test_static.isph"
// rule: run on OS=!windows
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2
// rule: skip on target=generic.*

task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = 0;

    // Continuation of the test from fmod-double-varying.ispc.
    // It's split into three files to reduce stack usage with O0.
    // Case 7.1: x is positive, y is inf
    double x = aFOO[programCount / 2];
    double y = 0.8 / 0.0;
    double testVal = fmod(x, y);
    RET[programIndex] += (testVal == x) ? 0 : 1;

    // Case 7.1: x is negative, y is inf
    x = -aFOO[programCount / 2];
    y = 0.8 / 0.0;
    testVal = fmod(x, y);
    RET[programIndex] += (testVal == x) ? 0 : 1;

    // Case 8.1: x is +0.0, y is positive
    x = 0.0;
    y = 0.8;
    testVal = fmod(x, y);
    RET[programIndex] += (testVal == x) ? 0 : 1;

    // Case 8.2: x is -0.0, y is positive
    x = -0.0;
    y = 0.8;
    testVal = fmod(x, y);
    RET[programIndex] += (testVal == x) ? 0 : 1;

    // Case 9.1: x is +0.0, y is 0.0
    x = 0.0;
    y = 0.0;
    testVal = fmod(x, y);
    RET[programIndex] += (isnan(testVal) && (signbits(testVal) == signbits(x))) ? 0 : 1;

    // Case 9.2: x is -0.0, y is 0.0
    x = -0.0;
    y = 0.0;
    testVal = fmod(x, y);
    RET[programIndex] += (isnan(testVal) && (signbits(testVal) == signbits(x))) ? 0 : 1;
}

task void result(uniform float RET[]) { RET[0] = 0; }
