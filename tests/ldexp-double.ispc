
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    double a = 1 << (programIndex % 28);
    if (programIndex & 1)
        a = -a;
    RET[programIndex] = ldexp(a, 2);
}

export void result(uniform float RET[]) {
    int pi = programIndex % 28;
    RET[programIndex] = (1 << (pi + 2));
    if (programIndex & 1)
        RET[programIndex] = -RET[programIndex];
}
