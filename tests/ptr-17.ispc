void inc(int * uniform v) {
    ++*v;
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform b = aFOO;
    b[programIndex] = programCount - programIndex;
    RET[programIndex] = aFOO[programIndex];
}

export void result(uniform float RET[]) {
    RET[programIndex] = programCount - programIndex;
}
