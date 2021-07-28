#include "../test_static.isph"

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float error = 0;
    RET[programIndex] = 0;
    float deltaLow = 0.2;
    float deltaHigh = 0.8;
    float testVal;
    float resVal;
    float baseVal;
    float checkVal = 0;

    // Case 1: Covers range  -(programCount/2 -1) < val < +(programCount/2) including -1 < val < +1 with deltaLow.
    // So, this covers +0.0, -0.0, -inf < val < 0 with decimal greater than 0.5, 0 < val < +inf with decimal less than 0.5
    // For eg. for a 8 wide target, this will check values : -2.8, -1.8, -0.8, 0.2, 1.2, 2.2, 3.2, 4.2
    baseVal = aFOO[programIndex] - (programCount / 2);
    testVal = baseVal + deltaLow;
    resVal = ceil(testVal);
    checkVal = baseVal + 1;
    if (checkVal != resVal) {
        error++;
    }

    // Case 2: Covers range  -(programCount/2 -1) < val < +(programCount/2) including -1 < val < +1 with deltaHigh.
    // So, this covers +0.0, -0.0, -inf < val < 0 with decimal less than 0.5, 0 < val < +inf with decimal greater than 0.5
    // For eg. for a 8 wide target, this will check values : -2.2, -1.2, -0.2, 0.8, 1.8, 2.8, 3.8, 4.8
    baseVal = aFOO[programIndex] - (programCount / 2);
    testVal = baseVal + deltaHigh;
    resVal = ceil(testVal);
    checkVal = baseVal + 1;
    if (checkVal != resVal) {
        error++;
    }

    // Case 3: Value +inf.
    baseVal = aFOO[programIndex];
    testVal = baseVal / 0.0;
    resVal = ceil(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 4: Value -inf.
    baseVal = -aFOO[programIndex];
    testVal = baseVal / 0.0;
    resVal = ceil(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 5: Value -nan.
    baseVal = -aFOO[programIndex];
    testVal = sqrt(baseVal);
    resVal = ceil(testVal);
    if (!isnan(resVal) || (signbits(testVal) != signbits(resVal))) {
        error++;
    }

    // Case 6: Value +nan.
    testVal = testVal * -1;
    resVal = ceil(testVal);
    if (!isnan(resVal) || (signbits(testVal) != signbits(resVal))) {
        error++;
    }

    // Case 7: Value +0.
    testVal = +0.0;
    resVal = ceil(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 8: Value -0.
    testVal = -0.0;
    resVal = ceil(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Check if any case fails.
    RET[programIndex] = error;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
