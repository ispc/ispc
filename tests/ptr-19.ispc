export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform b = aFOO;
    b += 10;
    int8 index = -5;
    RET[programIndex] = b[index];
}

export void result(uniform float RET[]) {
    RET[programIndex] = 6;
}
