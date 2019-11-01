export void f_f(uniform float RET[], uniform float aFOO[]) {
    int64 a = aFOO[programIndex];
    int64 b = -2. * a;
    RET[programIndex] = max(a,b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1 * (1 + programIndex);
}
