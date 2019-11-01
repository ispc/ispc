export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = (programCount < 10000) ? 1 : 0;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
