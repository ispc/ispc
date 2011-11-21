
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    double a = (1<<programIndex) * 1.5;
    if (programIndex & 1)
        a = -a;
    int exponent;
    frexp(a, &exponent);
    RET[programIndex] = exponent;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
