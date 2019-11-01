export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform b = aFOO;
    RET[programIndex] = b[programIndex];
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
