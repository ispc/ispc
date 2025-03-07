#include "test_static.isph"

task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = 0;

    // Case 1: x and y are generic positive values
    float x = aFOO[programCount / 2];
    float y = 0.8;
    float testVal = fmod(x, y);
    float baseRes = x - trunc(x / y) * y;
    RET[programIndex] += ((abs(baseRes - testVal) < 1e-6) || abs((baseRes-testVal)/baseRes) < 1e-5) ? 0 : 1;

    // Case 2: x is negative, y is positive
    x = -aFOO[programCount / 2];
    y = 0.8;
    testVal = fmod(x, y);
    baseRes = x - trunc(x / y) * y;
    RET[programIndex] += ((abs(baseRes - testVal) < 1e-6) || abs((baseRes-testVal)/baseRes) < 1e-5) ? 0 : 1;

    // Case 3: x is positive, y is negative
    x = aFOO[programCount / 2];
    y = -0.8;
    testVal = fmod(x, y);
    baseRes = x - trunc(x / y) * y;
    RET[programIndex] += ((abs(baseRes - testVal) < 1e-6) || abs((baseRes-testVal)/baseRes) < 1e-5) ? 0 : 1;

    // Case 4: x and y are negative
    x = -aFOO[programCount / 2];
    y = -0.8;
    testVal = fmod(x, y);
    baseRes = x - trunc(x / y) * y;
    RET[programIndex] += ((abs(baseRes - testVal) < 1e-6) || abs((baseRes-testVal)/baseRes) < 1e-5) ? 0 : 1;
}

task void result(uniform float RET[]) { RET[0] = 0; }
