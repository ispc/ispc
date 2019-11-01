export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * varying b = &aFOO[10];
    uniform float * uniform c = aFOO;
    RET[programIndex] = (b > c) ? 10 : 0;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 10;
}
