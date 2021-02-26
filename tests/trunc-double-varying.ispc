// rule: skip on cpu=TGLLP

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float error = 0;
    RET[programIndex] = 0;
    double deltaLow = 0.2d;
    double deltaHigh = 0.8d;
    double testVal;
    double resVal;
    double baseVal;
    double checkVal = 0;

    //Case 1 : Covers range  -(programCount/2 -1) < val < +(programCount/2) including -1 < val < +1 with deltaLow.
    // So, this covers +0.0, -0.0, -inf < val < 0 with decimal greater than 0.5, 0 < val < +inf with decimal less than 0.5
    //For eg. for a 8 wide target, this will check values : -2.999999,-1.999999,-0.999999,0.000001,1.000001,2.000001,3.000001,4.000001
    baseVal = aFOO[programIndex] - (programCount / 2);
    testVal = baseVal + deltaLow;
    resVal = trunc(testVal);
    checkVal = baseVal;
    if (baseVal < 0) {
        checkVal = -abs(baseVal + 1); // -abs() Needed for -0.0
    }
    if (checkVal != resVal) {
        error++;
    }

    //Case 2 : Covers range  -(programCount/2 -1) < val < +(programCount/2) including -1 < val < +1 with deltaHigh.
     // So, this covers +0.0, -0.0, -inf < val < 0 with decimal less than 0.5, 0 < val < +inf with decimal greater than 0.5
    //For eg. for a 8 wide target, this will check values : -2.000001,-1.000001,-0.000001,0.999999,1.999999,2.999999,3.999999,4.999999
    baseVal = aFOO[programIndex] - (programCount / 2);
    testVal = baseVal + deltaHigh;
    resVal = trunc(testVal);
    checkVal = baseVal;
    if (baseVal < 0) {
        checkVal = -abs(baseVal + 1); // -abs() Needed for -0.0
    }
    if (checkVal != resVal) {
        error++;
    }

    //Case 3 : Value +inf.
    baseVal = aFOO[programIndex];
    testVal = baseVal / 0.0;
    resVal = trunc(testVal);
    if (testVal != resVal) {
        error++;
    }

    //Case 4 : Value -inf.
    baseVal = -aFOO[programIndex];
    testVal = baseVal / 0.0;
    resVal = trunc(testVal);
    if (testVal != resVal) {
        error++;
    }

    //Case 5 : Value -nan.
    baseVal = -aFOO[programIndex];
    testVal = sqrt(baseVal);
    resVal = trunc(testVal);
    if (!isnan(resVal) || (signbits(testVal) != signbits(resVal))) {
        error++;
    }

    //Case 6 : Value +nan.
    testVal = testVal * -1;
    resVal = trunc(testVal);
    if (!isnan(resVal) || (signbits(testVal) != signbits(resVal))) {
        error++;
    }

    // Check if any case fails.
    RET[programIndex] = error;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
