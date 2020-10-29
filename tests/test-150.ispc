// rule: skip on cpu=TGLLP

export void f_f(uniform float RET[], uniform float aFOO[]) {
    double a = aFOO[programIndex]; 
    RET[programIndex] = round(a+(0.50001d));
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex + 2;
}
