// rule: skip on cpu=TGLLP

export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
    double a = aFOO[programIndex];
    RET[programIndex] = max(3 * a, (double)10.f);
    RET[programCount-1] = max(b, (double)100);
}


export void result(uniform float RET[]) {
    RET[programIndex] = 3 * (1+programIndex);
    RET[0] = 10;
    RET[1] = 10;
    RET[2] = 10;
    RET[programCount-1] = 100;
}
