export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform b = aFOO;
    RET[programIndex] = *b;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
