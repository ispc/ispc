
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    double a = 1 << (programIndex);
    if (programIndex & 1)
        a = -a;
    RET[programIndex] = ldexp(a, 2);
}

export void result(uniform float RET[]) {
    RET[programIndex] = (1 << (programIndex + 2));
    if (programIndex & 1)
        RET[programIndex] = -RET[programIndex];
}
