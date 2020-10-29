// rule: skip on cpu=TGLLP

export void f_duf(uniform float RET[], uniform double aFOO[], uniform float b) {
    double a = aFOO[programIndex];
    if (a <= 2.)
        { RET[programIndex] = 0; return; }
    RET[programIndex] = a;
}

export void result(uniform float RET[]) { 
    RET[programIndex] = 1+programIndex;
    RET[0] = RET[1] = 0;
}

