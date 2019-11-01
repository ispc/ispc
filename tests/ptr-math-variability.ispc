export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform ptr = aFOO;
    RET[programIndex] = *(ptr + programIndex) - 1;
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
