// rule: skip on cpu=TGLLP

export void f_f(uniform float RET[], uniform float aFOO[]) {
    double a = aFOO[programIndex];
    // Test parsing of double constants.
    double d1 = 1.0d40;
    double d2 = 1.d40;
    double d3 = 1d40;
    double d4 = .1d41;
    double d5 = 10000000000000000000000000000000000000000.d;
    double d6 = 10000000000000000000000000000000000000000.0d;

    // All the constants should be equal and if it's evaluated as "float",
    // then sqrt will evaluate to +inf.
    if (d1 == d2 && d1 == d3 && d1 == d4 && d1 == d5 && d1 == d6 &&
        ((float)sqrt(d1)) < 2e20) {
        RET[programIndex] = a;
    }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
