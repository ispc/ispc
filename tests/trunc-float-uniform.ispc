
export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float error = 0;
    RET[programIndex] = 0;
    uniform float deltaLow = 0.2;
    uniform float deltaHigh = 0.8;
    uniform float testVal;
    uniform float resVal;
    uniform float baseVal;

    // Case 1: Generic value 1 < val < +inf with deltaLow.
    baseVal = aFOO[programCount - 1];
    testVal = baseVal + deltaLow;
    resVal = trunc(testVal);
    if (baseVal != resVal) {
        error++;
    }

    // Case 2: Generic value -inf < val < -1 with deltaLow.
    baseVal = -aFOO[programCount - 1];
    testVal = baseVal - deltaLow;
    resVal = trunc(testVal);
    if (baseVal != resVal) {
        error++;
    }

    // Case 3: Generic value 1 < val < +inf with deltaHigh.
    baseVal = aFOO[programCount - 1];
    testVal = baseVal + deltaHigh;
    resVal = trunc(testVal);
    if (baseVal != resVal) {
        error++;
    }

    // Case 4: Generic value -inf < val < -1 with deltaHigh.
    baseVal = -aFOO[programCount - 1];
    testVal = baseVal - deltaHigh;
    resVal = trunc(testVal);
    if (baseVal != resVal) {
        error++;
    }

    // Range -1 > val > 1 is tested separately since different
    // emulation sequence is used for this on gen.
    // Case 5: Value 0 < val < +0.5.    
    baseVal = 0;
    testVal = baseVal + deltaLow;
    resVal = trunc(testVal);
    if (baseVal != resVal) {
        error++;
    }

    // Case 6: Value +0.5 < val < +1.
    baseVal = 0;
    testVal = baseVal + deltaHigh;
    resVal = trunc(testVal);
    if (baseVal != resVal) {
        error++;
    }

    // Case 7: Value -0.5 < val < 0.
    baseVal = 0;
    testVal = baseVal - deltaLow;
    resVal = trunc(testVal);
    if (baseVal != resVal) {
        error++;
    }

    // Case 8: Value -1 < val < -0.5.
    baseVal = 0;
    testVal = baseVal - deltaHigh;
    resVal = trunc(testVal);
    if (baseVal != resVal) {
        error++;
    }

    // Case 9: Value +0.
    testVal = +0.0;
    resVal = trunc(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 10: Value -0.
    testVal = -0.0;
    resVal = trunc(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 11: Value +inf.
    baseVal = aFOO[programCount - 1];
    testVal = baseVal / 0.0;
    resVal = trunc(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 12: Value -inf.
    baseVal = -aFOO[programCount - 1];
    testVal = baseVal / 0.0;
    resVal = trunc(testVal);
    if (testVal != resVal) {
        error++;
    }

    // Case 13: Value -nan.
    baseVal = -aFOO[programCount - 1];
    testVal = sqrt(baseVal);
    resVal = trunc(testVal);
    if (!isnan(resVal) || (signbits(testVal) != signbits(resVal))) {
        error++;
    }

    // Case 14: Value +nan.
    testVal = testVal * -1;
    resVal = trunc(testVal);
    if (!isnan(resVal) || (signbits(testVal) != signbits(resVal))) {
        error++;
    }

    // Check if any case fails.
    RET[0] = error;
}

export void result(uniform float RET[]) {
    RET[0] = 0;
}
