#include "test_static.isph"
// rule: skip on target=generic.*
// rule: run on OS=!windows
// rule: skip on arch=ppc64le

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float error = 0;
    RET[programIndex] = 0;
    uniform float16 deltaLow = 0.2f16;
    uniform float16 deltaHigh = 0.8f16;
    uniform float16 testVal;
    uniform float16 resVal;
    uniform float16 baseVal;

    // Case 1: Generic value 1 < val < +inf with deltaLow.
    baseVal = aFOO[programCount - 1];
    testVal = baseVal + deltaLow;
    resVal = round(testVal);
    if (baseVal != resVal) {
        error++;
    }

    // Case 2: Generic value -inf < val < -1 with deltaLow.
    baseVal = -aFOO[programCount - 1];
    testVal = baseVal - deltaLow;
    resVal = round(testVal);
    if (baseVal != resVal) {
        error++;
    }

    // Case 3: Generic value 1 < val < +inf with deltaHigh.
    baseVal = aFOO[programCount - 1];
    testVal = baseVal + deltaHigh;
    resVal = round(testVal);
    if ((baseVal + 1) != resVal) {
        error++;
    }

    // Case 4: Generic value -inf < val < -1 with deltaHigh.
    baseVal = -aFOO[programCount - 1];
    testVal = baseVal - deltaHigh;
    resVal = round(testVal);
    if ((baseVal - 1) != resVal) {
        error++;
    }

    // Range -1 > val > 1 is tested separately since different
    // emulation sequence is used for this on Xe.
    // Case 5: Value 0 < val < +0.5.
    baseVal = 0.0f16;
    testVal = baseVal + deltaLow;
    resVal = round(testVal);
    if (resVal != baseVal) {
        error++;
    }

    // Case 6: Value +0.5 < val < +1.
    baseVal = 0.0f16;
    testVal = baseVal + deltaHigh;
    resVal = round(testVal);
    if (resVal != (baseVal + 1.0f16)) {
        error++;
    }

    // Case 7: Value -0.5 < val < 0.
    baseVal = 0.0f16;
    testVal = baseVal - deltaLow;
    resVal = round(testVal);
    if (resVal != baseVal) {
        error++;
    }

    // Case 8: Value -1 < val < -0.5.
    baseVal = 0.0f16;
    testVal = baseVal - deltaHigh;
    resVal = round(testVal);
    if ((baseVal - 1.0f16) != resVal) {
        error++;
    }

    // Case 9: Value +0.
    testVal = +0.0f16;
    resVal = round(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 10: Value -0.
    testVal = -0.0f16;
    resVal = round(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 11: Value +inf.
    baseVal = aFOO[programCount - 1];
    testVal = baseVal / 0.0f16;
    resVal = round(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 12: Value -inf.
    baseVal = -aFOO[programCount - 1];
    testVal = baseVal / 0.0f16;
    resVal = round(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 13: Value -nan.
    baseVal = -aFOO[programCount - 1];
    testVal = sqrt(baseVal);
    resVal = round(testVal);
    if (!isnan(resVal) || (signbits(testVal) != signbits(resVal))) {
        error++;
    }

    // Case 14: Value +nan.
    testVal = testVal * -1.0f16;
    resVal = round(testVal);
    if (!isnan(resVal) || (signbits(testVal) != signbits(resVal))) {
        error++;
    }

    // Case 15: Large value near max half (65000.0) - already integral, no rounding needed.
    // This tests the overflow prevention fix where adding magic constant would overflow.
    testVal = 65000.0f16;
    resVal = round(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 16: Negative large value near min half (-65000.0).
    testVal = -65000.0f16;
    resVal = round(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 17: Value exactly at the boundary (1024.0) - already integral.
    testVal = 1024.0f16;
    resVal = round(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 18: Value just below boundary with fractional part (1023.5).
    testVal = 1023.5f16;
    resVal = round(testVal);
    if (1024.0f16 != resVal) {
        error++;
    }

    // Case 19: Large integral value (32768.0 = 2^15).
    testVal = 32768.0f16;
    resVal = round(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Check if any case fails.
    RET[0] = error;
}

task void result(uniform float RET[]) {
    RET[0] = 0;
}
