export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform b = aFOO;
    b += 5;
    RET[programIndex] = b - aFOO;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 5;
}
