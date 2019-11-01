export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * varying b = aFOO;
    uniform float * uniform c = &aFOO[10];
    RET[programIndex] = c - b;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 10;
}
