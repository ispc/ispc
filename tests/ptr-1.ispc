export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int a = 1;
    uniform int * uniform b = &a;
    RET[programIndex] = *b;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
