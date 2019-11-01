export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    // calculation error 1e-6 is the same as in icc
    RET[programIndex] = (exp(-log(1/a)) - a)/a < 1e-6 ? 1 : 0;
}

export void result(uniform float RET[4]) {
    RET[programIndex] = 1;
}
