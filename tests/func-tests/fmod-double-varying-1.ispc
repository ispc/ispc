#include "test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = 0;

    // Case 1: x and y are generic positive values
    double x = aFOO[programCount / 2];
    double y = 0.8;
    double testVal = fmod(x, y);
    double baseRes = x - trunc(x / y) * y;
    RET[programIndex] += (abs(baseRes - testVal) < 1e-7) ? 0 : 1;

    // Case 2: x is negative, y is positive
    x = -aFOO[programCount / 2];
    y = 0.8;
    testVal = fmod(x, y);
    baseRes = x - trunc(x / y) * y;
    RET[programIndex] += (abs(baseRes - testVal) < 1e-7) ? 0 : 1;

    // Case 3: x is positive, y is negative
    x = aFOO[programCount / 2];
    y = -0.8;
    testVal = fmod(x, y);
    baseRes = x - trunc(x / y) * y;
    RET[programIndex] += (abs(baseRes - testVal) < 1e-7) ? 0 : 1;

    // Case 4: x and y are negative
    x = -aFOO[programCount / 2];
    y = -0.8;
    testVal = fmod(x, y);
    baseRes = x - trunc(x / y) * y;
    RET[programIndex] += (abs(baseRes - testVal) < 1e-7) ? 0 : 1;
}

task void result(uniform float RET[]) { RET[0] = 0; }
