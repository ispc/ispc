export void f_f(uniform float RET[], uniform float aFOO[]) {
    int a = aFOO[programIndex];
    varying int * uniform b = &a;
    RET[programIndex] = *b;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
