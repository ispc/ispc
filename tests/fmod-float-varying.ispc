#include "../test_static.isph"

task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = 0;

    // Case 1: x and y are generic positive values
    float x = aFOO[programCount / 2];
    float y = 0.8;
    float testVal = fmod(x, y);
    float baseRes = x - trunc(x * rcp(y)) * y;
    RET[programIndex] += (abs(baseRes - testVal) < 1e-6) ? 0 : 1;

    // Case 2: x is negative, y is positive
    x = -aFOO[programCount / 2];
    y = 0.8;
    testVal = fmod(x, y);
    baseRes = x - trunc(x * rcp(y)) * y;
    RET[programIndex] += (abs(baseRes - testVal) < 1e-6) ? 0 : 1;

    // Case 3: x is positive, y is negative
    x = aFOO[programCount / 2];
    y = -0.8;
    testVal = fmod(x, y);
    baseRes = x - trunc(x * rcp(y)) * y;
    RET[programIndex] += (abs(baseRes - testVal) < 1e-6) ? 0 : 1;

    // Case 4: x and y are negative
    x = -aFOO[programCount / 2];
    y = -0.8;
    testVal = fmod(x, y);
    baseRes = x - trunc(x * rcp(y)) * y;
    RET[programIndex] += (abs(baseRes - testVal) < 1e-6) ? 0 : 1;

    // Case 5.1: x is +inf, y is positive
    x = aFOO[programCount / 2] / 0.0;
    y = 0.8;
    testVal = fmod(x, y);
    RET[programIndex] += (isnan(testVal) && (signbits(testVal) == signbits(x))) ? 0 : 1;

    // Case 5.2: x is -inf, y is positive
    x = -aFOO[programCount / 2] / 0.0;
    y = 0.8;
    testVal = fmod(x, y);
    RET[programIndex] += (isnan(testVal) && (signbits(testVal) == signbits(x))) ? 0 : 1;

    // Case 6: x is positive, y is 0.0
    x = aFOO[programCount / 2];
    y = 0.0;
    testVal = fmod(x, y);
    RET[programIndex] += (isnan(testVal) && (signbits(testVal) == signbits(x))) ? 0 : 1;

    // Case 7.1: x is positive, y is inf
    x = aFOO[programCount / 2];
    y = 0.8 / 0.0;
    testVal = fmod(x, y);
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
