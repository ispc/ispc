#include "test_static.isph"
// rule: run on OS=!windows
// rule: skip on arch=wasm32
// rule: skip on arch=wasm64
// rule: skip on target=generic.*

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float error = 0;
    RET[programIndex] = 0;
    float16 deltaLow = 0.2f16;
    float16 deltaHigh = 0.8f16;
    float16 testVal;
    float16 resVal;
    float16 baseVal;
    float16 checkVal = 0.0f16;

    // Case 1: Covers range  -(programCount/2 -1) < val < +(programCount/2) including -1 < val < +1 with deltaLow.
    // So, this covers +0.0, -0.0, -inf < val < 0 with decimal greater than 0.5, 0 < val < +inf with decimal less than 0.5
    // For eg. for a 8 wide target, this will check values : -2.8, -1.8, -0.8, 0.2, 1.2, 2.2, 3.2, 4.2
    baseVal = aFOO[programIndex] - (programCount / 2);
    testVal = baseVal + deltaLow;
    resVal = round(testVal);
    checkVal = baseVal;
    if (checkVal != resVal) {
        error++;
    }

    // Case 2: Covers range  -(programCount/2 -1) < val < +(programCount/2) including -1 < val < +1 with deltaHigh.
    // So, this covers +0.0, -0.0, -inf < val < 0 with decimal less than 0.5, 0 < val < +inf with decimal greater than 0.5
    // For eg. for a 8 wide target, this will check values : -2.2, -1.2, -0.2, 0.8, 1.8, 2.8, 3.8, 4.8
    baseVal = aFOO[programIndex] - (programCount / 2);
    testVal = baseVal + deltaHigh;
    resVal = round(testVal);
    checkVal = baseVal + 1.0f16;
    if (checkVal != resVal) {
        error++;
    }

    // Case 3: Value +inf.
    baseVal = aFOO[programIndex];
    testVal = baseVal / 0.0f16;
    resVal = round(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 4: Value -inf.
    baseVal = -aFOO[programIndex];
    testVal = baseVal / 0.0f16;
    resVal = round(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 5: Value -nan.
    baseVal = -aFOO[programIndex];
    testVal = sqrt(baseVal);
    resVal = round(testVal);
    if (!isnan(resVal) || (signbits(testVal) != signbits(resVal))) {
        error++;
    }

    // Case 6: Value +nan.
    testVal = testVal * -1.0f16;
    resVal = round(testVal);
    if (!isnan(resVal) || (signbits(testVal) != signbits(resVal))) {
        error++;
    }

    // Case 7: Value +0.
    testVal = +0.0f16;
    resVal = round(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 8: Value -0.
    testVal = -0.0f16;
    resVal = round(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 9: Large value near max half (65000.0) - already integral, no rounding needed.
    // This tests the overflow prevention fix where adding magic constant would overflow.
    testVal = 65000.0f16;
    resVal = round(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 10: Negative large value near min half (-65000.0).
    testVal = -65000.0f16;
    resVal = round(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 11: Value exactly at the boundary (1024.0) - already integral.
    testVal = 1024.0f16;
    resVal = round(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 12: Value just below boundary with fractional part (1023.5).
    testVal = 1023.5f16;
    resVal = round(testVal);
    if (1024.0f16 != resVal) {
        error++;
    }

    // Case 13: Large integral value (32768.0 = 2^15).
    testVal = 32768.0f16;
    resVal = round(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Check if any case fails.
    RET[programIndex] = error;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
