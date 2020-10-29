// rule: skip on cpu=TGLLP

export void f_duf(uniform float RET[], uniform double aFOO[], uniform float b) {
    double a = aFOO[programIndex];
    if (a+3 > b)
        { RET[programIndex] = 1; return; }
    RET[programIndex] = 0;
}

export void result(uniform float RET[]) { 
    RET[programIndex] = 1;
    RET[0] = RET[1] = 0;
}

