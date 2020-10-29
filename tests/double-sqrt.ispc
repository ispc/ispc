// rule: skip on cpu=TGLLP

export void f_f(uniform float RET[], uniform float aFOO[]) {
    double a = aFOO[programIndex];
    if (programIndex & 1) {
        a *= a;
        RET[programIndex] = sqrt(a);
    }
    else
        RET[programIndex] = a;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
